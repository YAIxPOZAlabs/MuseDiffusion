"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os

def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='folder where model checkpoint exist')
    parser.add_argument('--step', type=int, default=100,
                        help='ddim step, if not using ddim, should be same as diffusion step')
    parser.add_argument('--out_dir', type=str, default='./generation_outputs/',
                        help='output directory to store generated midi')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size to run decode')
    parser.add_argument('--use_ddim_reverse', type=bool, default=False,
                        help='choose forward process as ddim or not')
    parser.add_argument('--top_p', type=int, default=0,
                        help='range of the noise added to input, should be set between 0 and 1 (0=no restriction)')
    parser.add_argument('--split', type=str, default='valid',
                        help='dataset type used in sampling')
    parser.add_argument('--clamp_step', type=int, default=0,
                        help='in clamp_first mode, choose end clamp step, otherwise starting clamp step')
    parser.add_argument('--sample_seed', type=int, default=105,
                        help='random seed for sampling')
    parser.add_argument('--clip_denoised', type=bool, default=False,
                        help='denoising 시 clipping 진행여부')
    args = parser.parse_args(argv)

    if not args.model_path:  # Try to get latest model_path
        def get_latest_model_path(base_path):
            candidates = filter(os.path.isdir, os.listdir(base_path))
            candidates_join = (os.path.join(base_path, x) for x in candidates)
            candidates_sort = sorted(candidates_join, key=os.path.getmtime, reverse=True)
            if not candidates_sort:
                return
            ckpt_path = candidates_sort[0]
            candidates = filter(os.path.isfile, os.listdir(ckpt_path))
            candidates_join = (os.path.join(ckpt_path, x) for x in candidates if x.endswith('.pt'))
            candidates_sort = sorted(candidates_join, key=os.path.getmtime, reverse=True)
            if not candidates_sort:
                return
            return candidates_sort[0]

        model_path = get_latest_model_path("diffusion_models")
        if model_path is None:
            raise argparse.ArgumentTypeError("You should specify --model_path: no trained model in ./diffusion_models")
        args.model_path = model_path

    return args


def print_credit():  # Optional
    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        try:
            from MuseDiffusion.utils.etc import credit
            credit()
        except ImportError:
            pass


def main(args):
    ######################################### Setup #####################################
    # Ensure no_grad()
    import torch as th
    if th.is_grad_enabled():
        return th.no_grad()(main)(args)

    # Import dependencies
    import time
    from io import StringIO
    from contextlib import redirect_stdout
    from functools import partial
    from tqdm.auto import tqdm

    # Import everything
    from MuseDiffusion.config import TrainSettings
    from MuseDiffusion.data import load_data_music
    from MuseDiffusion.models.diffusion.rounding import denoised_fn_round
    from MuseDiffusion.utils import dist_util, logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all
    from MuseDiffusion.utils.decode_util import SequenceToMidi

    # Setup everything
    dist_util.setup_dist()
    world_size = dist_util.get_world_size()
    rank = dist_util.get_rank()
    dev = dist_util.dev()

    # Prepare output directory
    model_base_name = os.path.basename(os.path.split(args.model_path)[0])
    model_detailed_name = os.path.split(args.model_path)[1].split('.pt')[0]
    out_path = os.path.join(args.out_dir, model_base_name, model_detailed_name + ".samples")
    log_path = os.path.join(args.out_dir, model_base_name, model_detailed_name + ".logs")
    logger.configure(log_path, format_strs=["stdout", "log"], log_suffix="-" + model_detailed_name)
    if rank == 0:
        os.makedirs(out_path, exist_ok=True)
    else:
        logger.set_level(logger.DISABLED)
    dist_util.barrier()  # Sync

    # Reload train configurations from model folder
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    logger.log(f"### Loading training config from {config_path} ... ")
    training_args = TrainSettings.parse_file(config_path)
    training_args_dict = training_args.dict()
    training_args_dict.pop('batch_size')
    args.__dict__.update(training_args_dict)
    dist_util.barrier()  # Sync
    #########################################################################################

    ##################################### Initialization ####################################
    logger.log("### Creating model and diffusion... ")
    model, diffusion = create_model_and_diffusion(**training_args.dict())

    # Reload model weight from model folder
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    dist_util.barrier()  # Sync

    # Count and log total params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}. ")

    # Load embedding from model, used for reverse process
    model_emb = th.nn.Embedding(
        num_embeddings=args.vocab_size,
        embedding_dim=args.hidden_dim,
        padding_idx=0,
        _weight=model.word_embedding.weight.clone().cpu()
    )

    # Freeze weight and set
    model.eval().requires_grad_(False).to(dev)
    model_emb.eval().requires_grad_(False).to(dev)
    dist_util.barrier()  # Sync

    # Make cudnn deterministic
    seed_all(args.sample_seed, deterministic=True)

    # Prepare dataloader
    data_loader = load_data_music(
        split=args.split,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_corruption=args.use_corruption,
        corr_available=args.corr_available,
        corr_max=args.corr_max,
        corr_p=args.corr_p,
        use_bucketing=args.use_bucketing,
        seq_len=args.seq_len,
        deterministic=True,
        loop=False,
        num_preprocess_proc=1,
    )
    dist_util.barrier()  # Sync

    # Set up forward and sample functions
    # forward_fn = diffusion.q_sample if not args.use_ddim_reverse else diffusion.ddim_reverse_sample
    if args.step == args.diffusion_steps:
        args.use_ddim = False
        step_gap = 1
        sample_fn = diffusion.p_sample_loop
    else:
        args.use_ddim = True
        step_gap = args.diffusion_steps // args.step
        sample_fn = diffusion.ddim_sample_loop

    # Run sample loop
    logger.log(f"### Sampling on {args.split} ... ")
    start_t = time.time()
    iterator = tqdm(enumerate(data_loader), total=len(data_loader)) if rank == 0 else enumerate(data_loader)

    ##########################################################################################
    # Inference Loop
    for batch_index, cond in iterator:
        if batch_index % world_size != rank:
            continue

        input_ids_x = cond['input_ids']
        input_ids_mask_ori = cond['input_mask']

        x_start = model.get_embeds(input_ids_x.to(dev))
        input_ids_mask = th.broadcast_to(input_ids_mask_ori.to(dev).unsqueeze(dim=-1), x_start.shape)
        model_kwargs = cond

        # noise = th.randn_like(x_start)

        # if args.use_ddim_reverse:
        #     noise = x_start
        #     timestep = th.zeros((args.batch_size, ), device=dev, dtype=th.long)
        #     for i in range(args.diffusion_steps):
        #         timestep.fill_(i)
        #         noise = diffusion.ddim_reverse_sample(model, noise, t=timestep, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, )["sample"]
        # else:  
        # 일단 ddim reverse는 안쓰는게 나아 보임. 
        # 추가로 noising을 얼마나 줄지를 결정해야됨.
        noising_t = args.diffusion_steps
        timestep = th.full((args.batch_size, 1), noising_t - 1, device=dev)
        noise = diffusion.q_sample(x_start.unsqueeze(-1), timestep, mask=input_ids_mask)

        x_noised = th.where(input_ids_mask == 0, x_start, noise.squeeze(-1))

        samples = sample_fn(
            model=model,
            shape=(x_start.shape[0], args.seq_len, args.hidden_dim),
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

        # #################################################################################### #
        #                                        Decode                                        #
        #                          Convert Note Sequence To Midi file                          #
        # #################################################################################### #

        decoder = SequenceToMidi()

        sample = samples[-1]  # Sample last step

        reshaped_x_t = sample
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        sample_tokens = th.argmax(logits, dim=-1)

        out = StringIO()
        try:
            with redirect_stdout(out):
                # SequenceToMidi.save_tokens(
                #     input_ids_x.cpu().numpy(),
                #     sample_tokens.cpu().squeeze(-1).numpy(),
                #     output_dir=out_path,
                #     batch_index=batch_index
                # )
                decoder(
                    sequences=sample_tokens.cpu().numpy(),  # input_ids_x.cpu().numpy() - 원래 토큰으로 할 때
                    input_ids_mask_ori=input_ids_mask_ori,
                    output_dir=out_path,
                    batch_index=batch_index,
                    batch_size=args.batch_size
                )
        finally:
            logs_per_batch = out.getvalue()
            with open(os.path.join(log_path, f"batch{batch_index}.txt"), "wt") as fp:
                fp.write(logs_per_batch)
            # Print logs sequentially
            for i in range(world_size):
                if i == rank:
                    print(logs_per_batch)
                dist_util.barrier()

    # Sync each distributed node with dummy barrier() call
    rem = len(data_loader) % world_size
    if rem and rank >= rem:
        for _ in range(world_size):
            dist_util.barrier()

    # Log final result
    logger.log(f'### Total takes {time.time() - start_t:.2f}s .....')
    logger.log(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    arg = parse_args()
    print_credit()
    main(arg)
