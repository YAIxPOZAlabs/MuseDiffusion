"""
Helpers for distributed training.
"""

import io
import os
import socket
import functools

import blobfile as bf

import torch as th
import torch.distributed as dist
from torch.cuda import is_available as _cuda_available

# Change this to reflect your cluster layout.

USE_DIST_IN_WINDOWS = False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                       Setup Tools                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def is_available():
    if hasattr(is_available, 'cache'):
        return is_available.cache
    if os.name == 'nt' and not USE_DIST_IN_WINDOWS:
        if os.environ.get("LOCAL_RANK", str(0)) == str(0):
            import warnings
            warnings.warn(
                "In Windows, Distributed is unavailable by default settings.\n"
                "To enable Distributed, edit utils.dist_util.USE_DIST_IN_WINDOWS to True."
            )
    elif dist.is_available():  # All condition passed
        is_available.cache = True
        return True
    os.environ.setdefault("LOCAL_RANK", str(0))  # make legacy-rank-getter compatible
    is_available.cache = False
    return False


def is_initialized():
    return is_available() and getattr(dist, "is_initialized", lambda: False)()


@functools.lru_cache(maxsize=None)
def setup_dist(backend=None, hostname=None):
    """
    Setup a distributed process group.
    """

    if is_available():
        if os.name == 'nt':
            hostname = "localhost"

        try:
            _setup_dist(backend=backend, hostname=hostname)
            if os.environ["LOCAL_RANK"] == str(0):
                print("<INFO> torch.distributed setup success, using distributed setting..")
            return True
        except Exception as exc:
            print(f"<INFO> {exc.__class__.__qualname__}: {exc}")
            is_available.cache = False

    os.environ.setdefault("LOCAL_RANK", str(0))  # make legacy-rank-getter compatible
    if int(os.getenv("LOCAL_RANK")) == 0:
        print("<INFO> torch.distributed is not available, skipping distributed setting..")
    return False


def _setup_dist(backend=None, hostname=None):

    if is_initialized():
        return

    if backend is None:
        backend = "gloo" if not _cuda_available() else "nccl"

    if hostname is None:
        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)

    dist.init_process_group(backend=backend, init_method="env://")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                      General Tools                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_rank(group=None):
    if is_initialized():
        return dist.get_rank(group=group)
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size(group=None):
    if is_initialized():
        return dist.get_world_size(group=group)
    return 1


def barrier(*args, **kwargs):
    if is_initialized():
        return dist.barrier(*args, **kwargs)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if _cuda_available():
        return th.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                         Runners                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def run_argv_as_distributed(program, args, nproc_per_node=None, master_port=None):

    import sys
    import importlib
    import psutil
    import shlex
    import runpy
    import torch

    if importlib.util.find_spec('torch.distributed.run') is not None:  # NOQA
        distributed_run = 'torch.distributed.run'
        use_env = ''
    else:
        distributed_run = 'torch.distributed.launch'
        use_env = '--use_env'

    if master_port is None:
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
        if not is_available():
            raise RuntimeError("torch.distributed runtime not available")
        os.environ["DIST_RUNNING_FLAG"] = "1"
        sys.exit(run_argv_as_distributed(
            os.path.relpath(program, os.getcwd()), args,
            nproc_per_node=dist_namespace.nproc_per_node,
            master_port=dist_namespace.master_port
        ))
    is_available.cache = bool(int(os.getenv("DIST_RUNNING_FLAG", "0")))
    if parse_all:
        namespace = parser.parse_args(args)
        return namespace
    else:
        return args


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                    Internal Function                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


try:
    import nt  # NOQA
    os.sync = nt.sync = lambda: None  # signature: () -> None
except ImportError:
    pass


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
