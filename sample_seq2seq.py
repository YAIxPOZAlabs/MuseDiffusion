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

from config import CHOICES, DEFAULT_CONFIG
from data import load_data_music

from utils import dist_util, logger

from utils.argument_parsing import add_dict_to_argparser, args_to_dict
from utils.initialization import create_model_and_diffusion, load_model_emb

from utils.decode_util import SequenceToMidi

def create_argparser():
    #defaults = dict(model_path='./', step=100, out_dir='', top_p=0)
    #defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults = dict()
    defaults.update(DEFAULT_CONFIG)
    #defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='folder where model checkpoint exist')
    parser.add_argument('--step', type=int, default=100, help='ddim step, if not using ddim, should be same as diffusion step')
    parser.add_argument('--midi_out_dir', type=str, default='./output/midi', help='output directory to store generated midi')
    parser.add_argument('--token_out_dir', type=str, default='./output/token', help='output directory to store genearted token')
    parser.add_argument('--use_ddim_reverse', type=bool, default=False, help='choose forward process as ddim or not')
    parser.add_argument('--top_p', type=int, default=0, help='이거는 어떤 역할을 하는지 확인 필요')
    parser.add_argument('--split', type=str, default='valid', help='dataset type')
    parser.add_argument('--clamp_step', type=int, default=0, help='이것도 확인 필요')
    parser.add_argument('--seed2', type=int, default=105, help='random seed?')
    parser.add_argument('--clip_denoised', type=bool, default=False, help='아마도 denoising 시 clipping 진행여부')
    add_dict_to_argparser(parser, defaults, CHOICES)
    parser.batch_size = 32 # batch size 살짝 조정.
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

    #dist_util.setup_dist()
    DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "train_cfg.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.__dict__.update(checkpoint_path=config_path)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, DEFAULT_CONFIG.keys())
    )
    checkpoint_name = sorted(os.listdir(args.model_path))[-1]
    checkpoint_path = os.path.join(args.model_path, checkpoint_name)

    model.load_state_dict(
        #dist_util.load_state_dict(args.model_path, map_location="cpu")
        th.load(checkpoint_path)
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    #model.to(dist_util.dev())
    model.to(DEVICE)
    model.eval()

    #model_emb = load_model_emb(args)
    model_emb = th.nn.Embedding(args.vocab_size, args.hidden_dim, padding_idx=0)
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
        loop=False,
        num_proc=1,
    )

    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    # model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    # out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    # if not os.path.isdir(out_dir):
    #     os.mkdir(out_dir)

    # out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    # if not os.path.isdir(out_path):
    #     os.mkdir(out_path)
    # out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        #while True:
        i = 0
        while i != 1: # for fast inference
            batch, cond = next(data_valid)
            # print(batch.shape)
            all_test_data.append(cond)
            i = 1

    except StopIteration:
        print('### End of reading iteration...')
    
    from tqdm import tqdm

    #forward_fn = diffusion.q_sample if not args.use_ddim_reverse else diffusion.ddim_reverse_sample # config에 use_ddim_reverse boolean 타입으로 추가해야됨
    index=-1
    for cond in tqdm(all_test_data):
        index+=1
        input_ids_x = cond.pop('input_ids').to(DEVICE)#.to(dist_util.dev())
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
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(DEVICE)#.to(dist_util.dev())
        if args.use_ddim_reverse:
            noise = x_start
            for i in range(args.diffusion_steps):
                noise = diffusion.ddim_reverse_sample(model, noise, t = i, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, )
        else:
            timestep = th.full((args.batch_size, 1), args.diffusion_steps - 1).to(DEVICE)
            noise = diffusion.q_sample(x_start.unsqueeze(-1), timestep, mask=input_ids_mask)
        
        x_noised = th.where(input_ids_mask==0, x_start, noise.squeeze(-1))

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
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)
        # all_sentence = [sample.cpu().numpy() for sample in gathered_samples]
        
        # # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        #arr = np.concatenate(sample, axis=0)
        #x_t = th.tensor(arr).cuda()

        reshaped_x_t = sample
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample_tokens = cands.indices

        decoder.save_tokens(input_ids_x.cpu().numpy(), sample_tokens.cpu().squeeze(-1).numpy(), output_dir=args.token_out_dir, index=index)

        #decoder(sequences=sample_tokens.unsqueeze(-1).cpu(),input_ids_mask_ori=input_ids_mask_ori,seq_len=args.seq_len,output_dir=".")
        decoder(sequences=input_ids_x.cpu().numpy(),input_ids_mask_ori=input_ids_mask_ori,seq_len=args.seq_len,output_dir=args.midi_out_dir) # 원래 token으로 decoding 진행
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
    print(f'### Written the decoded output to {args.midi_out_dir}')

if __name__ == "__main__":
    #print("Not implemented")
    #raise SystemExit(1)
    main()
else:
    raise ImportError("Not Implemented")
