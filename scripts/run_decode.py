import importlib
import sys
import os
import glob
import psutil
import argparse


def main():

    parser = argparse.ArgumentParser(description='decoding args.')

    parser.add_argument('--nproc_per_node', type=int, default=0,
                        help='number of gpu used in distributed. (0=auto)')
    parser.add_argument('--master_port', type=int, default=12233,
                        help='master port used in distributed')

    parser.add_argument('--model_dir', type=str, default='',
                        help='path (or GLOB PATTERN) to the folder of diffusion model')
    parser.add_argument('--sample_seed', type=int, default=123,
                        help='random seed for sampling')
    parser.add_argument('--step', type=int, default=100,
                        help='if less than diffusion training steps, like 1000, use ddim sampling')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='dataset split used to decode')
    parser.add_argument('--top_p', type=int, default=-1,
                        help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema',
                        help='training pattern (e.g. "ema" - "ema*.pt" will be selected')

    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(__file__)
    dname, script_name = os.path.split(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    if importlib.util.find_spec('torch.distributed.run') is not None:  # NOQA
        distributed_run = 'torch.distributed.run'
        use_env = ''
    else:
        distributed_run = 'torch.distributed.launch'
        use_env = '--use_env'

    if args.nproc_per_node == 0:
        import torch  # lazy import
        args.nproc_per_node = torch.cuda.device_count() or 1
    os.environ.setdefault("OMP_NUM_THREADS", str(psutil.cpu_count(logical=False) // int(args.nproc_per_node)))

    out_dir = 'generation_outputs'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Pre-defined environs and Run name
    commandline_format = f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} " \
                         f"python -m {distributed_run} " \
                         f"--nproc_per_node={args.nproc_per_node} " \
                         f"--master_port={args.master_port + int(args.sample_seed)} " \
                         f"{use_env} " \
                         f"sample_seq2seq.py " \
                         f"--model_path {{model_path}} " \
                         f"--step {args.step} " \
                         f"--batch_size {args.batch_size} " \
                         f"--sample_seed {args.sample_seed} " \
                         f"--split {args.split} " \
                         f"--top_p {args.top_p} " \
                         f"--out_dir {out_dir} "

    run_result = 0

    for lst in glob.glob(args.model_dir):
        print(f"[{script_name.upper()}]", "Selecting model_dir:", lst)
        checkpoints = sorted(glob.glob(f"{lst}/{args.pattern}*.pt"))[::-1]

        for checkpoint_one in checkpoints:
            commandline = commandline_format.format(model_path=checkpoint_one)
            print(f"[{script_name.upper()}]", "Selecting checkpoint:", checkpoint_one)

            print(commandline)
            run_result = os.system(commandline) or run_result

            if run_result:
                print(f"[{script_name.upper()}]", "Decoding failed at:", checkpoint_one)
                sys.exit(run_result)

    print(f"[{script_name.upper()}]", 'decoding finished...')


if __name__ == '__main__':
    main()
