"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import json
import time

import numpy as np
import torch as th
import torch.distributed as dist

from tqdm.auto import tqdm

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from functools import partial

from models.diffuseq.rounding import denoised_fn_round, get_weights

from config import CHOICES, DEFAULT_CONFIG
from data import load_data_music

from utils import dist_util, logger

from utils.argument_parsing import add_dict_to_argparser, args_to_dict
from utils.initialization import create_model_and_diffusion, load_model_emb, random_seed_all

from utils.decode_util import SequenceToMidi


def parse_args(argv=None):
    # defaults = dict(model_path='./', step=100, out_dir='', top_p=0)
    # decode_defaults = dict(split='valid', clamp_step=0, sample_seed=105, clip_denoised=False)
    # defaults.update(decode_defaults)
    # TODO: 이거 다 바꾸기
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='folder where model checkpoint exist')
    parser.add_argument('--step', type=int, default=100, help='ddim step, if not using ddim, should be same as diffusion step')
    parser.add_argument('--out_dir', type=str, default='./output/', help='output directory to store generated midi')
    parser.add_argument('--midi_out_dir', type=str, default='./output/midi/', help='output directory to store generated midi')
    parser.add_argument('--token_out_dir', type=str, default='./output/token/', help='output directory to store genearted token')
    parser.add_argument('--use_ddim_reverse', type=bool, default=False, help='choose forward process as ddim or not')
    parser.add_argument('--top_p', type=int, default=0, help='이거는 어떤 역할을 하는지 확인 필요')
    parser.add_argument('--split', type=str, default='valid', help='dataset type used in sampling')
    parser.add_argument('--clamp_step', type=int, default=0, help='in clamp_first mode, choose end clamp step, otherwise starting clamp step')
    parser.add_argument('--sample_seed', type=int, default=105, help='random seed for sampling')
    parser.add_argument('--clip_denoised', type=bool, default=False, help='아마도 denoising 시 clipping 진행여부')
    add_dict_to_argparser(parser, DEFAULT_CONFIG, CHOICES)
    args = parser.parse_args(argv)

    if not args.model_path:  # GET DEFAULT MODEL PATH

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

    args.batch_size = 32  # batch size 살짝 조정.  # TODO

    return args


def print_credit():  # Optional
    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        try:
            from utils.etc import credit
            credit()
        except ImportError:
            pass


def main(args):

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb') as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.checkpoint_path = config_path

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, DEFAULT_CONFIG.keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    model_emb = load_model_emb(args, sync_weight=False)
    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    random_seed_all(args.sample_seed)

    print("### Sampling...on", args.split)

    # load data
    data_loader = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        split=args.split,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False,
        num_preprocess_proc=1
    )

    # 동하 TODO: 윗부분 Arg & 여기서부터

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_path = os.path.join(
        args.out_dir,
        f"{model_base_name.split('.ema')[0]}",
        f"ema{model_base_name.split('.ema')[1]}.samples"
    )
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"seed{args.sample_seed}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    # TODO #######################################

    if args.step == args.diffusion_steps:
        args.use_ddim = False
        step_gap = 1
    else:
        args.use_ddim = True
        step_gap = args.diffusion_steps // args.step

    # forward_fn = diffusion.q_sample if not args.use_ddim_reverse else diffusion.ddim_reverse_sample # config에 use_ddim_reverse boolean 타입으로 추가해야됨
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    for batch_index, (_, cond) in tqdm(enumerate(data_loader), total=len(data_loader)):

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask_ori = cond.pop('input_mask')

        # noise = th.randn_like(x_start)

        model_kwargs = {}
        input_ids_mask = th.broadcast_to(input_ids_mask_ori.to(dist_util.dev()).unsqueeze(dim=-1), x_start.shape)
        if args.use_ddim_reverse:  # TODO ################################################################################
            noise = x_start
            for i in range(args.diffusion_steps):
                noise = diffusion.ddim_reverse_sample(model, noise, t=i, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, )
        else:
            timestep = th.full((args.batch_size, 1), args.diffusion_steps - 1, device=dist_util.dev())
            noise = diffusion.q_sample(x_start.unsqueeze(-1), timestep, mask=input_ids_mask)
        
        x_noised = th.where(input_ids_mask == 0, x_start, noise.squeeze(-1))  # TODO: SQUEEZED ###########################

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

        model_emb_copy.cpu()

        # #################################################################################### #
        #                                        Decode                                        #
        #                          Convert Note Sequence To Midi file                          #
        # #################################################################################### #

        decoder = SequenceToMidi()

        sample = samples[-1]  # 하나 sample한다고 가정했을 때
        print(sample.shape)
        # reshaped_x_t = sample

        gathered_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample_tokens = cands.indices

        SequenceToMidi.save_tokens(
            input_ids_x.cpu().numpy(),
            sample_tokens.cpu().squeeze(-1).numpy(),
            output_dir=args.token_out_dir,
            batch_index=batch_index
        )

        decoder(
            sequences=sample_tokens.unsqueeze(-1).cpu().numpy(),
            input_ids_mask_ori=input_ids_mask_ori,
            seq_len=args.seq_len,
            output_dir=".",
            batch_index=batch_index,
            batch_size=args.batch_size
        )

        dist.barrier()

        # decoder(
        #     sequences=input_ids_x.cpu().numpy(),
        #     input_ids_mask_ori=input_ids_mask_ori,
        #     seq_len=args.seq_len,
        #     output_dir=args.midi_out_dir,
        #     batch_index=batch_index,
        #     batch_size=args.batch_size
        # )  # 원래 token으로 decoding 진행
 
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {args.midi_out_dir}')


if __name__ == "__main__":
    arg = parse_args()
    print_credit()
    main(arg)
