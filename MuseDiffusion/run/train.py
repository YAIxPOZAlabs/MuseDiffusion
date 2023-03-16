# python3 MuseDiffusion/run/train.py
from MuseDiffusion.config import TrainSettings


def create_parser():
    return TrainSettings.to_argparse(add_json=True)


def main(namespace):

    # Create config from parsed argument namespace
    args: TrainSettings = TrainSettings.from_argparse(namespace)

    # Import dependencies
    import os
    import time

    # Import everything
    from MuseDiffusion.data import load_data_music
    from MuseDiffusion.models.step_sample import create_named_schedule_sampler
    from MuseDiffusion.utils import dist_util, logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all, \
        fetch_pretrained_embedding, overload_embedding, \
        fetch_pretrained_denoiser, overload_denoiser
    from MuseDiffusion.utils.train_util import TrainLoop
    from MuseDiffusion.utils.plotting import embedding_tsne_trainer_wandb_callback

    # Credit
    try: from MuseDiffusion.utils.credit_printer import credit; credit()  # NOQA
    except Exception: pass  # NOQA

    # Setup distributed
    dist_util.setup_dist()
    rank = dist_util.get_rank()

    # Set checkpoint path
    folder_name = "diffusion_models/"
    if not os.path.isdir(folder_name) and rank == 0:
        os.mkdir(folder_name)
    if not args.checkpoint_path:
        model_file = f"MuseDiffusion_{args.dataset}_h{args.hidden_dim}_lr{args.lr}" \
                     f"_t{args.diffusion_steps}_{args.noise_schedule}_{args.schedule_sampler}" \
                     f"_seed{args.seed}_{time.strftime('%Y%m%d-%H:%M:%S')}"
        args.checkpoint_path = os.path.join(folder_name, model_file)
    if not os.path.isdir(args.checkpoint_path) and rank == 0:
        os.mkdir(args.checkpoint_path)

    # Configure log and seed
    logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"] + (["stdout"] if rank == 0 else []))
    seed_all(args.seed)

    # Prepare dataloader
    logger.log("### Creating data loader...")
    dist_util.barrier()  # Sync
    data = load_data_music(
        split='train',
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_corruption=args.use_corruption,
        corr_available=args.corr_available,
        corr_max=args.corr_max,
        corr_p=args.corr_p,
        corr_kwargs=args.corr_kwargs,
        use_bucketing=args.use_bucketing,
        seq_len=args.seq_len,
        deterministic=False,
        loop=True,
        num_loader_proc=args.data_loader_workers,
    )
    data_valid = load_data_music(
        split='valid',
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        corr_available=args.corr_available,
        corr_max=args.corr_max,
        corr_p=args.corr_p,
        corr_kwargs=args.corr_kwargs,
        use_bucketing=args.use_bucketing,
        seq_len=args.seq_len,
        deterministic=True,
        loop=True,
        num_loader_proc=args.data_loader_workers,
    )
    dist_util.barrier()  # Sync

    # Initialize model and diffusion
    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    # Load Pretrained Embedding Layer
    pretrained_emb_weight = fetch_pretrained_embedding(args)
    if pretrained_emb_weight is not None:
        overload_embedding(model, pretrained_emb_weight, args.freeze_embedding)

    # Loaded Pretrained De-noising Layer
    pretrained_denoiser_dict = fetch_pretrained_denoiser(args)
    if pretrained_denoiser_dict is not None:
        overload_denoiser(model, pretrained_denoiser_dict)

    # Load model to each node's device
    model.to(dist_util.dev())
    dist_util.barrier()  # Sync

    # Count and log total params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Save training args
    training_args_path = f'{args.checkpoint_path}/training_args.json'
    if not os.path.exists(training_args_path):
        logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
        if rank == 0:
            with open(training_args_path, 'w') as fp:
                print(args.json(indent=2), file=fp)

    # Init wandb
    if rank == 0:
        # Uncomment and customize your wandb setting on your own, or just use environ.
        import wandb
        wandb.init(
            mode=os.getenv("WANDB_MODE", "online"),
            # entity=os.getenv("WANDB_ENTITY", "<your-value>"),
            # project=os.getenv("WANDB_PROJECT", "<your-value>"),
        )
        wandb.config.update(args.dict(), allow_val_change=True)
    dist_util.barrier()  # Sync last

    # Run train loop
    logger.log("### Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        eval_callbacks=[embedding_tsne_trainer_wandb_callback]
    ).run_loop()


if __name__ == "__main__":
    from MuseDiffusion.utils.dist_run import parse_and_autorun
    main(parse_and_autorun(create_parser()))
