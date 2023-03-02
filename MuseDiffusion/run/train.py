"""
Train a diffusion model on images.
"""
import os
import json

import wandb
from MuseDiffusion.config import TrainSettings
from MuseDiffusion.utils import dist_util


def configure_wandb(args):
    wandb.init(
        mode=os.getenv("WANDB_MODE", "online"),  # you can change it to offline
        entity=os.getenv("WANDB_ENTITY", "yai-diffusion"),
        project=os.getenv("WANDB_PROJECT", "YAIxPOZAlabs"),
        name=args.checkpoint_path
    )
    wandb.config.update(args.__dict__, allow_val_change=True)


def print_credit():  # Optional
    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        try:
            from MuseDiffusion.utils.etc import credit
            credit()
        except ImportError:
            pass


def patch_proc_name_by_rank():
    if dist_util.is_available():
        title = f"[DISTRIBUTED NODE {dist_util.get_rank()}]"
        try:
            import setproctitle  # NOQA
            setproctitle.setproctitle(title)
        except ImportError:
            pass


def parse_args() -> TrainSettings:

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    setting_group = parser.add_argument_group(title="settings")
    setting_group.add_mutually_exclusive_group().add_argument(
        "--config_json", type=str, required=False,
        help="You can alter arguments all below by config_json file.")
    TrainSettings.to_argparse(setting_group.add_mutually_exclusive_group())

    namespace = dist_util.parse_and_autorun(parser)

    if namespace.config_json:
        return TrainSettings.parse_file(namespace.config_json)
    else:
        if hasattr(namespace, "config_json"):
            delattr(namespace, "config_json")
        return TrainSettings.from_argparse(namespace)


def main(args: TrainSettings):

    # Import everything
    import time
    from MuseDiffusion.data import load_data_music
    from MuseDiffusion.models.diffusion.step_sample import create_named_schedule_sampler
    from MuseDiffusion.utils import logger
    from MuseDiffusion.utils.initialization import create_model_and_diffusion, seed_all, \
        fetch_pretrained_embedding, overload_embedding, \
        fetch_pretrained_denoiser, overload_denoiser
    from MuseDiffusion.utils.train_util import TrainLoop
    from MuseDiffusion.utils.plotting import embedding_tsne_trainer_wandb_callback

    # Setup distributed
    dist_util.setup_dist()
    rank = dist_util.get_rank()
    dist_util.barrier()  # Sync

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
        use_bucketing=args.use_bucketing,
        seq_len=args.seq_len,
        deterministic=True,
        loop=True,
        num_loader_proc=args.data_loader_workers,
    )
    dist_util.barrier()  # Sync

    # Initialize model and diffusion
    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args.dict())

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
        if dist_util.get_rank() == 0:
            with open(training_args_path, 'w') as fp:
                json.dump(args.dict(), fp, indent=2)

    # Init wandb
    if dist_util.get_rank() == 0:
        configure_wandb(args)
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
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
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
    arg = parse_args()
    print_credit()
    patch_proc_name_by_rank()
    main(arg)
