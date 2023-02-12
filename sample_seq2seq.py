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
from transformers import set_seed

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from functools import partial

from models.diffuseq.rounding import denoised_fn_round, get_weights

from config import load_defaults_config, CHOICES
from data import load_data_music

from utils import dist_util, logger

from utils.argument_parsing import add_dict_to_argparser, args_to_dict
from utils.initialization import create_model_and_diffusion, load_model_emb

from utils.decode_util import SequenceToMidi

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults, CHOICES)
    return parser


def print_credit():
    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        try:
            from utils.etc import credit
            credit()
        except ImportError:
            pass


def main():
    args = create_argparser().parse_args()
    print_credit()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    model_emb = load_model_emb(args)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_music(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        split=args.split,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')
    
    from tqdm import tqdm

    #forward_fn = diffusion.q_sample if not args.use_ddim_reverse else diffusion.ddim_reverse_sample # config에 use_ddim_reverse boolean 타입으로 추가해야됨

    for cond in tqdm(all_test_data):

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        #noise = th.randn_like(x_start)
        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        model_kwargs = {}
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        if args.use_ddim_reverse:
            noise = x_start
            for i in range(args.diffusion_steps):
                noise = diffusion.ddim_reverse_sample(model, noise, t = i, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, )
        else:
            noise = diffusion.q_sample(x_start, args.diffusion_steps, mask=input_ids_mask)
        
        x_noised = th.where(input_ids_mask==0, x_start, noise)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

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
        # print(samples[0].shape) # samples for each step
        
        ########################################################################################       
        # Decode
        # Convert Note Sequence To Midi file
        ########################################################################################
        decoder = SequenceToMidi()

        sample = samples[-1]
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]
        
        # # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample_tokens = cands.indices
        decoder(sequences=sample_tokens,input_ids_mask_ori=input_ids_mask_ori,seq_len=args.seq_len,output_dir=".")

        # for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
        #     len_x = args.seq_len - sum(input_mask).tolist()
        #     tokens = tokenizer.decode_token(seq[len_x:])
        #     word_lst_recover.append(tokens)
        #
        # for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
        #     # tokens = tokenizer.decode_token(seq)
        #     len_x = args.seq_len - sum(input_mask).tolist()
        #     word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
        #     word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))
        #
 
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    print("Not implemented")
    raise SystemExit(1)
    # main()
else:
    raise ImportError("Not Implemented")
