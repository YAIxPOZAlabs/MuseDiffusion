"""
Train a diffusion model on images.
"""

import argparse
import json, os

from transformers import set_seed
import wandb

from config import load_defaults_config

from data import load_data_music

from models.diffuseq.utils import dist_util, logger
from models.diffuseq.step_sample import create_named_schedule_sampler

from utils.initialization import create_model_and_diffusion, load_model_emb
from utils.argument_parsing import add_dict_to_argparser, args_to_dict

from utils.train_util import TrainLoop


### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    model_emb = load_model_emb(args)

    data = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        model_emb=model_emb # use model's weights as init
    )
    next(data)

    data_valid = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        split='valid',
        deterministic=True,
        model_emb=model_emb # using the same embedding wight with tranining data
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
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
    main()
