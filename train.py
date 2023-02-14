"""
Train a diffusion model on images.
"""

import argparse
import os
import json

import wandb

from config import CHOICES, DEFAULT_CONFIG

from data import load_data_music

from models.diffuseq.step_sample import create_named_schedule_sampler

from utils import dist_util, logger

from utils.initialization import create_model_and_diffusion, load_model_emb, random_seed_all
from utils.argument_parsing import add_dict_to_argparser, args_to_dict

from utils.train_util import TrainLoop


### custom your wandb setting here ### TODO
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


def parse_args(argv=None):
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
    random_seed_all(args.seed)
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    model_emb = load_model_emb(args)

    data = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        split='train',
        deterministic=False,
        model_emb=model_emb  # use model's weights as init
    )
    next(data)  # try iter

    data_valid = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        split='valid',
        deterministic=True,
        model_emb=model_emb  # using the same embedding wight with tranining data
    )

    # print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, DEFAULT_CONFIG.keys())
    )
    model.to(dist_util.dev())
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "YAIxPOZAlabs"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

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
