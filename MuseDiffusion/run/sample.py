# python3 MuseDiffusion/run/sample.py
import torch
from MuseDiffusion.config import GenerationSettings, ModificationSettings


def create_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as Df
    parser = ArgumentParser(formatter_class=Df)
    subparsers = parser.add_subparsers(title="sampling-mode", dest="mode", required=True,
                                       help="available sampling mode. "
                                            "type each mode followed up by --help, to show full usage. "
                                            "available modes are 'generation'(=gen) and 'modification'(=mod).")
    GenerationSettings.to_argparse(subparsers.add_parser("generation", aliases=["gen"], formatter_class=Df))
    ModificationSettings.to_argparse(subparsers.add_parser("modification", aliases=["mod"], formatter_class=Df))
    return parser


@torch.no_grad()
def main(namespace):

    # Create config from parsed argument namespace
    klass = {"generation": GenerationSettings, "modification": ModificationSettings}[namespace.__dict__.pop("mode")]
    args: "ModificationSettings|GenerationSettings" = klass.from_argparse(namespace)

    # Import dependencies
    import os
    import time
    from io import StringIO
    from contextlib import redirect_stdout
    from functools import partial
    from tqdm.auto import tqdm

    # Import everything
    from MuseDiffusion.config import TrainSettings
    from MuseDiffusion.data import load_data_music, infinite_loader_from_single
    from MuseDiffusion.models.rounding import denoised_fn_round
    from MuseDiffusion.utils import dist_util, logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all
    from MuseDiffusion.utils.decode_util import batch_decode_seq2seq, batch_decode_generate, MetaToBatch

    # Credit
    try: from MuseDiffusion.utils.credit_printer import credit; credit()  # NOQA
    except Exception: pass  # NOQA

    # Setup distributed
    dist_util.setup_dist()
    world_size = dist_util.get_world_size()
    rank = dist_util.get_rank()
    dev = dist_util.dev()

    # Prepare output directory
    model_base_name = os.path.basename(os.path.split(args.model_path)[0])
    model_detailed_name = os.path.split(args.model_path)[1].split('.pt')[0]
    out_path = os.path.join(args.out_dir, model_base_name, model_detailed_name + ".samples")
    log_path = os.path.join(args.out_dir, model_base_name, model_detailed_name + ".log")
    if rank == 0:
        logger.configure(out_path, format_strs=["stdout"])
        os.makedirs(out_path, exist_ok=True)
    else:
        logger.configure(out_path, format_strs=[])
        logger.set_level(logger.DISABLED)
    dist_util.barrier()  # Sync

    # Reload train configurations from model folder
    logger.log(f"### Loading training config from {args.model_config_json} ... ")
    training_args = TrainSettings.parse_file(args.model_config_json)
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
    model_emb = torch.nn.Embedding(
        num_embeddings=training_args.vocab_size,
        embedding_dim=training_args.hidden_dim,
        padding_idx=0,
        _weight=model.word_embedding.weight.clone().cpu()
    )

    # Freeze weight and set
    model.eval().requires_grad_(False).to(dev)
    model_emb.eval().requires_grad_(False).to(dev)
    dist_util.barrier()  # Sync

    # Make cudnn deterministic
    seed_all(args.sample_seed, deterministic=True)

    # Set up forward and sample functions
    if args.step == training_args.diffusion_steps:
        step_gap = 1
        sample_fn = diffusion.p_sample_loop
    else:
        step_gap = training_args.diffusion_steps // args.step
        sample_fn = diffusion.ddim_sample_loop

    # Prepare dataloader and fn
    if args.__GENERATE__:
        batch = MetaToBatch.execute(args.midi_meta.dict(), args.batch_size, training_args.seq_len)
        data_loader = infinite_loader_from_single(batch)
        midi_decode_fn = batch_decode_generate
    else:
        for name in ['use_corruption', 'corr_available', 'corr_max', 'corr_p', 'corr_kwargs']:
            if getattr(args, name) is None:
                setattr(args, name, getattr(training_args, name))
        data_loader = load_data_music(
            split=args.split,
            batch_size=args.batch_size,
            data_dir=training_args.data_dir,
            use_corruption=args.use_corruption,
            corr_available=args.corr_available,
            corr_max=args.corr_max,
            corr_p=args.corr_p,
            corr_kwargs=args.corr_kwargs,
            use_bucketing=training_args.use_bucketing,
            seq_len=training_args.seq_len,
            deterministic=True,
            loop=False,
            num_preprocess_proc=1,
        )
        midi_decode_fn = batch_decode_seq2seq
    iterator = enumerate(data_loader)
    if rank == 0:
        iterator = tqdm(iterator, total=float('inf') if args.__GENERATE__ else len(data_loader))
    dist_util.barrier()  # Sync

    # Run sample loop
    logger.log(f"### Sampling on {'META' if args.__GENERATE__ else args.split} ... ")
    total_valid_count = torch.tensor(0, device=dev)
    generation_done = False  # for generation mode
    start_t = time.time()

    for batch_index, cond in iterator:
        if batch_index % world_size != rank:
            continue
        if args.__GENERATE__ and generation_done:
            break

        input_ids_x = cond['input_ids'].to(dev)
        input_ids_mask_ori = cond['input_mask'].to(dev)

        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = torch.broadcast_to(input_ids_mask_ori.unsqueeze(dim=-1), x_start.shape)
        model_kwargs = cond

        if args.__GENERATE__:
            noising_t = None
            noise = torch.randn_like(x_start)  # randn_like: device will be same as x_start
            x_noised = torch.where(input_ids_mask == 0, x_start, noise)
        else:
            noising_t = int(args.step * args.strength)
            timestep = torch.full((args.batch_size, 1), noising_t - 1, device=dev)
            x_noised = diffusion.q_sample(x_start.unsqueeze(-1), timestep, mask=input_ids_mask).squeeze(-1)

        samples = sample_fn(
            model=model,
            shape=(x_start.shape[0], training_args.seq_len, training_args.hidden_dim),
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap,
            t_enc=noising_t,
            only_last=True  # Use only last step (returns length-1 list)
        )

        sample = samples[-1]
        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        sample_tokens = torch.argmax(logits, dim=-1)

        # Decode sequentially
        for i in range(world_size):
            if i == rank and not generation_done:
                out = StringIO()
                try:
                    with redirect_stdout(out):
                        valid_count = midi_decode_fn(
                            sequences=sample_tokens.cpu().numpy(),
                            input_ids_mask_ori=input_ids_mask_ori.cpu().numpy(),
                            output_dir=out_path,
                            batch_index=batch_index,
                            previous_count=(
                                total_valid_count.item() if args.__GENERATE__ else batch_index * args.batch_size
                            )
                        )
                        total_valid_count += valid_count
                        print(total_valid_count.item())
                finally:
                    logs_per_batch = out.getvalue()
                    # Print logs sequentially
                    with open(log_path, "a") as fp:
                        fp.write(logs_per_batch)
                    print(logs_per_batch)
            dist_util.sync_params([total_valid_count], src=i)
            dist_util.barrier()
            if args.__GENERATE__ and total_valid_count.item() >= args.num_samples:
                generation_done = True

    # Sync each distributed node with dummy barrier() call
    if not args.__GENERATE__:
        rem = len(data_loader) % world_size
        if rem and rank >= rem:
            for i in range(world_size):
                dist_util.sync_params([total_valid_count], src=i)
                dist_util.barrier()

    # Log final result
    logger.log(f'### Total takes {time.time() - start_t:.2f}s .....')
    logger.log(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    from MuseDiffusion.utils.dist_run import parse_and_autorun
    main(parse_and_autorun(create_parser()))
