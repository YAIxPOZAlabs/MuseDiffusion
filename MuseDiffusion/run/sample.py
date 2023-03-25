# python3 MuseDiffusion/run/sample.py
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from MuseDiffusion.config import GenerationSettings, ModificationSettings


# Switch to calculate metric in modification
GET_METRIC = False


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


@record
@torch.no_grad()
def main(namespace):

    # Create config from parsed argument namespace
    mode = namespace.__dict__.pop("mode")
    klass = {"generation": GenerationSettings, "modification": ModificationSettings}[mode]
    args: "ModificationSettings|GenerationSettings" = klass.from_argparse(namespace)

    # Import dependencies
    import os
    import time
    import numpy as np
    from io import StringIO
    from contextlib import redirect_stdout
    from collections import OrderedDict
    from functools import partial
    from tqdm.auto import tqdm

    # Import everything
    from MuseDiffusion.config import TrainSettings
    from MuseDiffusion.data import load_data_music, infinite_loader_from_single
    from MuseDiffusion.models.rounding import denoised_fn_round
    from MuseDiffusion.utils import dist_util, logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all
    from MuseDiffusion.utils.decode_util import meta_to_batch, decode_batch, split_meta_midi
    from MuseDiffusion.metric import ONNC, Controllability_Pitch, Controllability_Velocity

    # Credit
    try: from MuseDiffusion.utils.credit_printer import credit; credit()  # NOQA
    except Exception: pass  # NOQA

    # Setup distributed
    dist_util.setup_dist()
    world_size = dist_util.get_world_size()
    rank = dist_util.get_rank()
    dev = dist_util.dev()

    # Prepare output directory
    base_path = os.path.join(args.out_dir, os.path.basename(os.path.split(args.model_path)[0]))
    file_or_dir_name = os.path.split(args.model_path)[1].split('.pt')[0] + "." + mode
    out_path = os.path.join(base_path, file_or_dir_name + ".samples")
    log_path = os.path.join(base_path, file_or_dir_name + ".log")

    # In sampling, we will log decoding results MANUALLY, so we will not configure logger properly.
    # logger.log() - equal to print(), but only in master process
    # print() - all process
    if rank == 0:
        logger.configure(out_path, format_strs=["stdout"])
        os.makedirs(out_path, exist_ok=True)
    else:
        logger.configure(out_path, format_strs=[])
        logger.set_level(logger.DISABLED)

    # Reload train configurations from model folder
    logger.log(f"### Loading training config from {args.model_config_json} ... ")
    training_args = TrainSettings.parse_file(args.model_config_json)
    dist_util.barrier()  # Sync

    # Initialize model and diffusion, and reload model weight from model folder
    logger.log("### Creating model and diffusion... ")
    model, diffusion = create_model_and_diffusion(training_args)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))

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

    # Freeze weight and set eval mode
    model.eval().requires_grad_(False).to(dev)
    model_emb.eval().requires_grad_(False).to(dev)
    torch.cuda.empty_cache()
    dist_util.barrier()  # Sync

    # Make cudnn deterministic
    seed_all(args.sample_seed, deterministic=True)

    # Set up sample function from step gap
    if args.step == training_args.diffusion_steps:
        step_gap = 1
        sample_fn = diffusion.p_sample_loop
    else:
        step_gap = training_args.diffusion_steps // args.step
        sample_fn = diffusion.ddim_sample_loop

    # Prepare dataloader and decoding function
    if args.__GENERATE__:  # this indicates generation mode
        data_loader = infinite_loader_from_single(meta_to_batch(
            midi_meta_dict=args.midi_meta_dict,
            batch_size=args.batch_size,
            seq_len=training_args.seq_len
        ))
        tqdm_total = float('inf')
    else:  # this indicates modification mode
        args.overload_corruption_settings_from(training_args)
        data_loader = load_data_music(
            split=args.split,
            batch_size=args.batch_size,
            data_dir=training_args.data_dir,
            use_corruption=args.use_corruption,
            corr_available=args.corr_available,
            corr_max=args.corr_max,
            corr_p=args.corr_p,
            corr_kwargs=args.corr_kwargs,
            use_bucketing=False,  # for stable sampling
            seq_len=training_args.seq_len,
            deterministic=True,
            loop=False,
            num_preprocess_proc=1,
        )
        tqdm_total = len(data_loader)

    # Convert dataloader to enumerated-progress-bar form
    iterator = enumerate(data_loader)
    if rank == 0:
        iterator = tqdm(iterator, total=tqdm_total)
    dist_util.barrier()  # Sync

    # Initialize sampling state variables
    logger.log(f"### Start {mode} ...")
    total_valid_count = torch.tensor(0, device=dev)  # define it as tensor for synchronization
    generation_done = False  # for generation - indicates if generation is done
    start_t = time.time()

    if GET_METRIC and not args.__GENERATE__ and args.use_corruption:
        logger.log(f"### with calculating metrics ...")
        metric_total = OrderedDict()
        metric_total["onnc_sum"] = torch.tensor(0., device=dev)
        metric_total["onnc_count"] = torch.tensor(0, device=dev)
        metric_total["total_total_p"] = torch.tensor(0, device=dev)
        metric_total["total_total_v"] = torch.tensor(0, device=dev)
        metric_total["total_wrong_p"] = torch.tensor(0, device=dev)
        metric_total["total_wrong_v"] = torch.tensor(0, device=dev)
    else:
        metric_total = None

    # Run sample loop
    for batch_index, cond in iterator:
        if batch_index % world_size != rank:
            # This makes sampling with multi node available.
            # Each node decodes batch number of which modular is same with node rank.
            continue
        if args.__GENERATE__ and generation_done:
            # In generation mode, this flag means generation is done, so we can stop infinite loop.
            break

        input_ids_x = cond['input_ids']
        input_ids_mask_ori = cond['input_mask']

        # for metric
        if not args.__GENERATE__ and args.use_corruption:
            correct_ids = cond['correct_ids']

        # Prepare variables for noising and sampling
        x_start = model.get_embeds(input_ids_x.to(dev))
        input_ids_mask = torch.broadcast_to(input_ids_mask_ori.to(dev).unsqueeze(dim=-1), x_start.shape)
        model_kwargs = cond

        # Noising - Generation: Random Noise, Modification: Q-Sample (forward step)
        if args.__GENERATE__:
            noising_t = None
            noise = torch.randn_like(x_start)  # device will be same as x_start
            x_noised = torch.where(torch.eq(input_ids_mask, 0), x_start, noise)
        else:
            noising_t = int(args.step * args.strength)
            timestep = torch.full((len(cond['input_ids']), 1), noising_t - 1, device=dev)
            x_noised = diffusion.q_sample(x_start.unsqueeze(-1), timestep, mask=input_ids_mask).squeeze(-1)

        # Run diffusion reverse step
        samples = sample_fn(
            model=model,
            shape=(x_start.shape[0], training_args.seq_len, training_args.hidden_dim),
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, model_emb, dist=None),  # sig: (model_emb, x, t, dist) -> (x, t)
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

        # Convert continuous sample to discrete token
        sample = samples[-1]
        logits = model.get_logits(sample)  # bsz, seq_len, vocab
        sample_tokens = torch.argmax(logits, dim=-1)

        for i in range(world_size):  # Decode sequentially
            if i == rank and not generation_done:
                # We use redirecting context manager again: to log into both stdout and log file.
                out = StringIO()
                try:
                    with redirect_stdout(out):
                        sample_tokens = sample_tokens.cpu().numpy()
                        input_ids_mask_ori = input_ids_mask_ori.cpu().numpy()

                        valid_count, invalid_idxes = decode_batch(
                            mode=mode,
                            sequences=sample_tokens,
                            input_ids_mask_ori=input_ids_mask_ori,
                            output_dir=out_path,
                            batch_index=batch_index,
                            previous_count=(
                                total_valid_count.item() if args.__GENERATE__ else batch_index * args.batch_size
                            ),
                            return_indices=True,
                            strict_validation=metric_total is not None
                        )
                        total_valid_count += valid_count

                        if metric_total is not None and valid_count:
                            correct_ids = correct_ids.cpu().numpy()
                            valid_mask = np.ones((len(correct_ids),), dtype=bool)
                            valid_mask[invalid_idxes] = False

                            correct_ids = correct_ids[valid_mask]
                            sample_tokens = sample_tokens[valid_mask]
                            input_ids_mask_ori = input_ids_mask_ori[valid_mask]

                            # for ONNC (Only when dataloader is there)
                            ground_truth_midis = tuple(
                                split_meta_midi(c, i)[0]
                                for c, i in zip(correct_ids, input_ids_mask_ori)
                            )
                            generated_midis, metas = zip(*(
                                split_meta_midi(s, i)
                                for s, i in zip(sample_tokens, input_ids_mask_ori)
                            ))
                            onnc = ONNC(ground_truth_midis + generated_midis, device=dev)
                            metric_total["onnc_sum"] += valid_count * onnc
                            metric_total["onnc_count"] += valid_count

                            # for CP, CV
                            total_p, wrong_p = Controllability_Pitch(metas, generated_midis)
                            total_v, wrong_v = Controllability_Velocity(metas, generated_midis)
                            metric_total["total_total_p"] += total_p
                            metric_total["total_wrong_p"] += wrong_p
                            metric_total["total_total_v"] += total_v
                            metric_total["total_wrong_v"] += wrong_v

                            print(f"{f' Metric of Batch {batch_index} ':=^60}")
                            print(f"{f' ONNC: {float(onnc):.6f} ': ^60}")
                            print(f"{f' CP: {wrong_p / total_p:.6f} ': ^60}")
                            print(f"{f' CV: {wrong_v / total_v:.6f} ': ^60}")
                            print(("=" * 60) + "\n")

                finally:
                    logs_per_batch = out.getvalue()
                    with open(log_path, "a") as fp:
                        fp.write(logs_per_batch)
                    print(logs_per_batch)

            # We can get 'exact' total_valid_count, due to sequential decoding.
            dist_util.broadcast(total_valid_count, src=i)
            if metric_total is not None:
                dist_util.sync_params(metric_total.values(), src=i)
            dist_util.barrier()
            if args.__GENERATE__ and total_valid_count.item() >= args.num_samples:
                # We have made enough midis, so we can stop generation loop.
                generation_done = True

    # In modification mode, you must sync remaining distributed nodes with dummy barrier() call
    if not args.__GENERATE__:
        rem = len(data_loader) % world_size
        if rem and rank >= rem:
            for i in range(world_size):
                dist_util.broadcast(total_valid_count, src=i)
                if metric_total is not None:
                    dist_util.sync_params(metric_total.values(), src=i)
                dist_util.barrier()

    # Log the whole metric
    if metric_total is not None:
        if rank == 0:
            total_onnc = float(metric_total["onnc_sum"] / metric_total["onnc_count"])
            total_cp = float(metric_total["total_wrong_p"] / metric_total["total_total_p"])
            total_cv = float(metric_total["total_wrong_v"] / metric_total["total_total_v"])
            print()
            print(f"{f' Summary: Metric ':=^60}")
            print(f"{f' ONNC: {total_onnc:.6f} ': ^60}")
            print(f"{f' CP: {total_cp:.6f} ': ^60}")
            print(f"{f' CV: {total_cv:.6f} ': ^60}")
            print(("=" * 60) + "\n")
        dist_util.barrier()

    # Log final result
    logger.log(f'### Total takes {time.time() - start_t:.2f}s .....')
    logger.log(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    from MuseDiffusion.utils.dist_run import parse_and_autorun
    main(parse_and_autorun(create_parser()))
