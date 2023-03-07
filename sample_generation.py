import os


def parse_args(argv=None):
    from MuseDiffusion.config import SamplingSettings
    return SamplingSettings.from_argv(argv)


def print_credit():  # Optional
    from MuseDiffusion.utils.etc import credit
    credit()


def main(args):

    # Ensure no_grad()
    import torch as th
    if th.is_grad_enabled():
        return th.no_grad()(main)(args)

    # Import dependencies
    import time
    from io import StringIO
    from contextlib import redirect_stdout
    from functools import partial

    # Import everything
    from MuseDiffusion.config import TrainSettings
    from MuseDiffusion.models.rounding import denoised_fn_round
    from MuseDiffusion.utils import dist_util, logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all
    from MuseDiffusion.utils.decode_util import SequenceToMidi

    # Setup everything
    dist_util.setup_dist()
    world_size = dist_util.get_world_size()
    rank = dist_util.get_rank()
    dev = dist_util.dev()

    #Prepare output directory
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

    # Initialize model and diffusion
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

    # Encode input meta
    from MuseDiffusion.utils.decode_util import MetaToSequence
    from sample_meta_generator import get_meta

    encoded_meta = MetaToSequence().execute(get_meta())
    encoded_meta = th.tensor(encoded_meta, device=dev)
    dist_util.barrier()  # Sync

    # Set up forward and sample functions
    if args.step == args.diffusion_steps:
        step_gap = 1
        sample_fn = diffusion.p_sample_loop
    else:
        step_gap = args.diffusion_steps // args.step
        sample_fn = diffusion.ddim_sample_loop

    # Run sample loop
    logger.log(f"### Sampling on {args.split} ... ")
    start_t = time.time()

    input_ids_mask_ori = th.ones(args.batch_size, args.seq_len, device=dev)
    input_ids_mask_ori[:, :len(encoded_meta) + 1] = 0

    input_ids_x = th.zeros(args.batch_size, args.seq_len, device=dev, dtype=th.int)
    input_ids_x[:, :len(encoded_meta)] = encoded_meta

    x_start = model.get_embeds(input_ids_x)
    input_ids_mask = th.broadcast_to(input_ids_mask_ori.unsqueeze(dim=-1), x_start.shape)
    model_kwargs = {'input_ids': input_ids_x, 'input_mask': input_ids_mask_ori}

    trial = 0
    batch_index = 0  # TODO: Fix it
    while True:
        try:
            logger.log("\n### Trial: %s" % trial)
            noise = th.randn_like(x_start)  # randn_like: device will be same as x_start
            x_noised = th.where(input_ids_mask == 0, x_start, noise)

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
                        input_ids_mask_ori=input_ids_mask_ori.cpu().numpy(),
                        output_dir=out_path,
                        batch_index=batch_index,
                        batch_size=args.batch_size
                    )
            finally:
                logs_per_batch = out.getvalue()
                with open(os.path.join(log_path, f"batch{batch_index}.txt"), "wt") as fp:
                    fp.write(logs_per_batch)
                print(logs_per_batch)

            # Log final result
            logger.log(f'### Total takes {time.time() - start_t:.2f}s .....')
            logger.log(f'### Written the decoded output to {out_path}')
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.log("### Trial %s has been failed due to %s." % (trial, type(e).__name__))


if __name__ == "__main__":
    arg = parse_args()
    print_credit()
    main(arg)
