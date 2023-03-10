def run_argv_as_distributed(program_or_module, argv, dist_namespace, *, run_as_module=False):

    import os
    import psutil

    import torch
    assert torch.__version__ >= (1, 9), "Requires torch version greater than or equal to 1.9!"

    from torch.cuda import device_count
    from torch.distributed.run import run
    from torch.distributed.elastic.multiprocessing.errors import record

    from .dist_util import is_available
    if not is_available():
        raise RuntimeError("torch.distributed runtime not available")

    os.environ.setdefault("OMP_NUM_THREADS", str(psutil.cpu_count(logical=False) // (device_count() or 1)))

    if hasattr(dist_namespace, "distributed"):
        delattr(dist_namespace, "distributed")
    default = create_distributed_parser().parse_args([])
    cmdline = "python3 -m torch.distributed.run "
    cmdline += " ".join(
        "--{0} {1}".format(k, v) for k, v in dist_namespace.__dict__.items()
        if (getattr(default, k) != v and v not in ('', None))
        or k == "nproc_per_node" and v != 1
    )
    cmdline += " " + "-m " * run_as_module + program_or_module + " "
    cmdline += " ".join(argv)
    print("[COMMANDLINE]\n" + cmdline + "\n")

    dist_namespace.module = run_as_module
    dist_namespace.training_script = program_or_module
    dist_namespace.training_script_args = argv

    @record
    def main(args):
        return run(args)

    main(dist_namespace)


def create_distributed_parser(parser=None):
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
    from torch.distributed.argparse_util import env, check_env
    from torch.cuda import is_available
    if parser is None:
        parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    # copied from torch.distributed.run
    parser.add_argument(  # Added
        "--distributed",
        action="store_true",
        help="to use torch.distributed.run"
    )
    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc_per_node",
        action=env,
        type=str,
        default="gpu" if is_available() else 1,  # Changed
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )
    parser.add_argument(
        "--rdzv_backend",
        action=env,
        type=str,
        default="static",
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on port 29400. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv_backend, --rdzv_endpoint, --rdzv_id are auto-assigned; any explicitly set values "
        "are ignored.",
    )
    parser.add_argument(
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor_interval",
        action=env,
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "--no_python",
        action=check_env,
        help=SUPPRESS,  # Not Required
    )

    parser.add_argument(
        "--run_path",
        action=check_env,
        help=SUPPRESS,  # Not Required
    )
    parser.add_argument(
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0). It should be either the IP address or the "
        "hostname of rank 0. For single node multi-proc training the --master_addr can simply be "
        "127.0.0.1; IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training.",
    )
    return parser


def parse_distributed_args(parser, args=None, parse_all=True):

    dist_parser = create_distributed_parser()
    dist_parser.prog = len(parser.prog)
    dist_namespace, args = dist_parser.parse_known_args(args)

    # Attach helps to original parser
    subparsers = [parser]
    try:
        if parser._subparsers is not None:  # NOQA
            from argparse import _SubParsersAction  # NOQA
            subparsers = next(s for s in parser._subparsers._actions if isinstance(s, _SubParsersAction))  # NOQA
            subparsers = list(subparsers._name_parser_map.values())  # NOQA
            subparsers.append(parser)  # NOQA
    except (ImportError, StopIteration):
        pass
    for _parser in subparsers:
        dist_parser.prog = " " * len(_parser.prog)
        _parser.usage = ("\n" if _parser is parser else "\n       ").join(_parser._get_formatter()._format_usage(  # NOQA
            _parser.usage, _parser._actions, _parser._mutually_exclusive_groups, "" # NOQA
        ).splitlines())
        _parser.usage += dist_parser.format_usage().replace("usage: ", " " * 7 if _parser is parser else "")
        _parser.epilog = (
            "NOTE - You can run this script with [torch.distributed]. "
            "Add `--distributed` argument, and other options from `python3 -m torch.distributed.run --help`. "
            "signature: " + dist_parser.format_usage().replace("usage: ", "")
        )

    dist_parser.prog = " " * len(parser.prog)

    if parse_all:
        namespace = parser.parse_args(args)
        return dist_namespace, namespace
    else:
        return dist_namespace, args


def get_main_modname():
    import os
    import sys
    depth = 1
    try:
        while True:
            f_globals = sys._getframe(depth).f_globals  # NOQA
            if f_globals['__name__'] == '__main__':
                break
            depth += 1
    except (AttributeError, ValueError):
        return
    if f_globals['__spec__'] is not None:
        module_name = f_globals['__spec__'].name
    elif f_globals['__package__'] is not None and '__file__' in f_globals:
        mod = os.path.splitext(os.path.split(f_globals['__file__'])[1])[0]
        module_name = f_globals['__package__'] + '.' + mod
    else:
        return
    if module_name.endswith(".__main__"):
        module_name = module_name[:-9]
    return module_name


def parse_and_autorun(parser, args=None, namespace=None, *, module_name=None, parse_all=True):
    """
    usage
    original: parser.parse_args()
    new:      parse_and_autorun(parser)
    """

    import os
    import sys

    if args is None:
        args = sys.argv[1:]
    if module_name is None:  # try to run as module
        module_name = get_main_modname()
    if module_name is None:
        run_as_module = False
        program_or_module = sys.argv[0]
    else:
        run_as_module = True
        program_or_module = module_name

    dist_namespace, args = parse_distributed_args(parser, args=args, parse_all=False)
    if parse_all:
        result = parser.parse_args(args, namespace)
    else:
        result = args

    if dist_namespace.__dict__.pop('distributed'):
        os.environ["DIST_UTIL_AUTORUN_FLAG"] = "1"
        sys.exit(run_argv_as_distributed(program_or_module, args, dist_namespace, run_as_module=run_as_module))

    if int(os.getenv("DIST_UTIL_AUTORUN_FLAG", "0")) == 1:  # subprocess called by this function
        from .dist_util import is_available
        is_available.cache = True
        try:  # If available, set process title with node name.
            import setproctitle  # NOQA
            setproctitle.setproctitle(f"[DISTRIBUTED NODE {os.getenv('LOCAL_RANK', '0')}]")
        except ImportError:
            pass
    return result
