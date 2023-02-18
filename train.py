"""
Train a diffusion model on images.
"""

import os
import json

import wandb

from config import CHOICES, DEFAULT_CONFIG
from utils.argument_parsing import add_dict_to_argparser, args_to_dict


def configure_wandb(args):
    wandb.init(
        mode=os.getenv("WANDB_MODE", "online"),  # you can change it to offline
        entity=os.getenv("WANDB_ENTITY", "yai-diffusion"),
        project=os.getenv("WANDB_PROJECT", "YAIxPOZAlabs"),
        name=args.checkpoint_path
    )
    wandb.config.update(args.__dict__, allow_val_change=True)


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, DEFAULT_CONFIG, CHOICES)  # update latest args according to argparse
    return parser.parse_args(argv)


def print_credit():  # Optional
    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        try:
            from utils.etc import credit
            credit()
        except ImportError:
            pass


def main(args):

    # Import everything
    from data import load_data_music
    from models.diffuseq.step_sample import create_named_schedule_sampler
    from utils import dist_util, logger
    from utils.initialization import create_model_and_diffusion, random_seed_all
    from utils.train_util import TrainLoop

    # Setup everything
    dist_util.setup_dist()
    dist_util.barrier()  # Sync
    logger.configure()
    random_seed_all(args.seed)

    # Prepare dataloader
    logger.log("### Creating data loader...")
    dist_util.barrier()  # Sync
    data = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        split='train',
        deterministic=False,
        num_loader_proc=args.data_loader_workers,
    )
    data_valid = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        split='valid',
        deterministic=True,
        num_loader_proc=args.data_loader_workers,
    )
    dist_util.barrier()  # Sync

    # Initialize model and diffusion
    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, DEFAULT_CONFIG.keys()))
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
            with open(training_args_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)

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
        eval_interval=args.eval_interval
    ).run_loop()


if __name__ == "__main__":
    arg = parse_args()
    print_credit()
    main(arg)
