import sys
import os
import psutil
import argparse
import time
import json


# def original_main():
#
#     parser = argparse.ArgumentParser(description='training args.')
#
#     parser.add_argument('--config_file', type=str, default='', help='path to training config')
#
#     parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args')
#     parser.add_argument('--app', type=str, default='', help='other input args')
#
#     # parser.add_argument('--dataset', type=str, default='', help='name of training dataset')
#     # parser.add_argument('--data_dir', type=str, default='', help='path to training dataset')
#     #
#     # parser.add_argument('--diff_steps', type=int, default=4000, help='diffusion steps')
#     # parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin'], help='the distribution of noises')
#     # parser.add_argument('--schedule_sampler', type=str, default='uniform', choices=['uniform', 'lossaware', 'fixstep'], help='schedule sampler of timesteps')
#     #
#     # parser.add_argument('--seq_len', type=int, default=128, help='max len of input sequence')
#     # parser.add_argument('--hidden_t_dim', type=int, default=128, help='hidden size of time embedding')
#     # parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size of word embedding')
#     # parser.add_argument('--learning_steps', type=int, default=40000, help='total steps of learning')
#     # parser.add_argument('--save_interval', type=int, default=10000, help='save step')
#     # parser.add_argument('--resume_checkpoint', type=str, default='', help='path to resume checkpoint, like xxx/xxx.pt')
#     # parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
#     # parser.add_argument('--bsz', type=int, default=64, help='batch size')
#     # parser.add_argument('--microbatch', type=int, default=64, help='microbatch size')
#     # parser.add_argument('--seed', type=int, default=101, help='random seed')
#
#     args = parser.parse_args()
#
#     # set working dir to the upper folder
#     abspath = os.path.abspath(__file__)
#     dname = os.path.dirname(abspath)
#     dname = os.path.dirname(dname)
#     sys.path.append(dname)  # Assure upper folder import
#     os.chdir(dname)
#
#     import config
#
#     config_file = args.config_file
#     if not os.path.isfile(config_file):
#         raise argparse.ArgumentTypeError("config_file does not exist: {}".format(config_file))
#     with open(args.config_file) as fp:
#         train_py_configs = json.load(fp)
#     args.__dict__.update(config.load_dict_config(train_py_configs))
#
#     folder_name = "diffusion_models/"
#
#     if int(os.environ['LOCAL_RANK']) == 0:
#         if not os.path.isdir(folder_name):
#             os.mkdir(folder_name)
#
#     Model_FILE = f"diffuseq_{args.dataset}_h{args.hidden_dim}_lr{args.lr}" \
#                  f"_t{args.diffusion_steps}_{args.noise_schedule}_{args.schedule_sampler}" \
#                  f"_seed{args.seed}"
#     if args.notes:
#         args.notes += time.strftime("%Y%m%d-%H:%M:%S")
#         Model_FILE = Model_FILE + f'_{args.notes}'
#     Model_FILE = os.path.join(folder_name, Model_FILE)
#
#     if int(os.environ['LOCAL_RANK']) == 0:
#         if not os.path.isdir(Model_FILE):
#             os.mkdir(Model_FILE)
#
#     COMMANDLINE = f"OPENAI_LOGDIR={Model_FILE} " \
#                   f"TOKENIZERS_PARALLELISM=false " \
#                   f"python train.py   " \
#                   f"--checkpoint_path {Model_FILE} "
#     for k, v in train_py_configs.items():
#         if isinstance(v, bool):
#             v = 'y' if v else 'n'
#         COMMANDLINE += f"--{k} {v} "
#
#     COMMANDLINE += " " + args.app
#
#     if int(os.environ['LOCAL_RANK']) == 0:
#         with open(os.path.join(Model_FILE, 'saved_bash.sh'), 'w') as f:
#             print(COMMANDLINE, file=f)
#
#     print(COMMANDLINE)
#     os.system(COMMANDLINE)


def main():

    parser = argparse.ArgumentParser(description='training args.')

    parser.add_argument('--nproc_per_node', type=int, default=4, help='number of gpu used in distributed')
    parser.add_argument('--master_port', type=int, default=12233, help='master port used in distributed')
    parser.add_argument('--config_file', type=str, default='', help='path to training config')

    parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args')
    parser.add_argument('--app', type=str, default='', help='other input args')

    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    dirname = os.path.dirname(dirname)
    sys.path.append(dirname)  # Assure upper folder import
    os.chdir(dirname)

    import config

    config_file = args.config_file
    if not os.path.isfile(config_file):
        raise argparse.ArgumentTypeError("config_file does not exist: {}".format(config_file))
    with open(args.config_file) as fp:
        train_py_configs = json.load(fp)
    args.__dict__.update(config.load_dict_config(train_py_configs))

    from importlib.util import find_spec
    if find_spec('torch.distributed.run') is not None:
        distributed_run = 'torch.distributed.run'
        use_env = ''
    else:
        distributed_run = 'torch.distributed.launch'
        use_env = '--use_env'

    folder_name = "diffusion_models/"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    model_file = f"diffuseq_{args.dataset}_h{args.hidden_dim}_lr{args.lr}" \
                 f"_t{args.diffusion_steps}_{args.noise_schedule}_{args.schedule_sampler}" \
                 f"_seed{args.seed}"
    if args.notes:
        args.notes += time.strftime("%Y%m%d-%H:%M:%S")
        model_file = model_file + f'_{args.notes}'
    model_file = os.path.join(folder_name, model_file)

    if not os.path.isdir(model_file):
        os.mkdir(model_file)

    os.environ.setdefault("OMP_NUM_THREADS", str(psutil.cpu_count(logical=False) // int(args.nproc_per_node)))

    commandline = f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} OPENAI_LOGDIR={model_file} " \
                  f"python -m {distributed_run} " \
                  f"--nproc_per_node={args.nproc_per_node} --master_port={args.master_port} {use_env} " \
                  f"train.py " \
                  f"--checkpoint_path {model_file} "
    for k, v in train_py_configs.items():
        if isinstance(v, str) and not v:
            continue
        if isinstance(v, bool):
            v = 'y' if v else 'n'
        commandline += f"--{k} {v} "
    commandline += args.app

    with open(os.path.join(model_file, 'saved_bash.sh'), 'w') as f:
        print(commandline, file=f)

    print(commandline)
    os.system(commandline)


if __name__ == '__main__':
    main()
