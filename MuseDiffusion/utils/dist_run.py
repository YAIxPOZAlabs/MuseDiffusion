def run_argv_as_distributed(program, args, nproc_per_node=None, master_port=None):

    import os
    import sys
    import importlib
    import psutil
    import shlex
    import runpy
    import torch

    from .dist_util import is_available
    if not is_available():
        raise RuntimeError("torch.distributed runtime not available")

    if importlib.util.find_spec('torch.distributed.run') is not None:  # NOQA
        distributed_run = 'torch.distributed.run'
        use_env = ''
    else:
        distributed_run = 'torch.distributed.launch'
        use_env = '--use_env'

    if master_port is None:
        from .dist_util import _find_free_port
        master_port = _find_free_port()

    if not nproc_per_node:
        nproc_per_node = torch.cuda.device_count() or 1

    os.environ.setdefault("OMP_NUM_THREADS", str(psutil.cpu_count(logical=False) // (torch.cuda.device_count() or 1)))

    distributed_argv = f"--nproc_per_node={nproc_per_node} --master_port={master_port} {use_env} "

    print("[COMMANDLINE]")
    print(f"python -m {distributed_run} {distributed_argv} {program} " + " ".join(args))
    print()

    sys.argv[:] = distributed_run, *shlex.split(distributed_argv), program, *args
    return runpy.run_module(distributed_run, run_name='__main__', alter_sys=True)


def parse_distributed_args(parser, argv=None, parse_all=True):
    from argparse import ArgumentParser
    dist_parser = ArgumentParser(add_help=False, prog=" " * len(parser.prog))
    dist_parser.add_argument('--distributed', action='store_true',
                             help='to use torch.distributed')
    dist_parser.add_argument('--nproc_per_node', type=int, default=0, metavar='{optional int, 0: auto}')
    dist_parser.add_argument('--master_port', type=int, default=12233, metavar='{optional int}')
    dist_namespace, args = dist_parser.parse_known_args(argv)
    # Attach helps to original parser
    parser.usage = parser.format_usage().strip("usage: ") + dist_parser.format_usage().replace("usage: ", " " * 7)
    parser.epilog = (
        "NOTE - You can run this script with [torch.distributed]. "
        "Add some args at first by " + dist_parser.format_usage()
    )
    if parse_all:
        namespace = parser.parse_args(args)
        return dist_namespace, namespace
    else:
        return dist_namespace, args


def parse_and_autorun(parser, parse_all=True):
    import os
    import sys
    program = sys.argv[0]
    dist_namespace, args = parse_distributed_args(parser, parse_all=False)
    if dist_namespace.distributed:
        os.environ["DIST_RUNNING_FLAG"] = "1"
        sys.exit(run_argv_as_distributed(
            os.path.relpath(program, os.getcwd()), args,
            nproc_per_node=dist_namespace.nproc_per_node,
            master_port=dist_namespace.master_port
        ))
    if parse_all:
        namespace = parser.parse_args(args)
        result = namespace
    else:
        result = args
    if int(os.getenv("DIST_RUNNING_FLAG", "0")) == 1:
        from .dist_util import is_available
        is_available.cache = True
    return result
