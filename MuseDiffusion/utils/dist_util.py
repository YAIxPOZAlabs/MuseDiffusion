"""
Helpers for distributed training.
this utility FORCES distributed learning at first.
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


def is_available():
    if getattr(is_available, 'cache', None) is None:
        is_available.cache = setup_dist()
    return is_available.cache


@functools.lru_cache(maxsize=None)
def setup_dist():
    """
    Setup a distributed process group.
    """
    if os.name == 'nt':
        _fetch_windows()
        kwd = dict(hostname="localhost")
    else:
        kwd = dict()
    if os.name == 'nt' and not USE_DIST_IN_WINDOWS:
        if os.environ.get("LOCAL_RANK", str(0)) == str(0):
            import warnings
            warnings.warn(
                "In Windows, Distributed is unavailable by default settings. "
                "To enable Distributed, edit utils.dist_util.USE_DIST_IN_WINDOWS to True."
            )
    elif dist.is_available():  # All condition passed
        try:
            _setup_dist(**kwd)
            if os.environ["LOCAL_RANK"] == str(0):
                print("<INFO> torch.distributed setup success, using distributed setting..")
            return True
        except Exception as exc:
            print(f"<INFO> {exc.__class__.__qualname__}: {exc}")
    os.environ.setdefault("LOCAL_RANK", str(0))  # make legacy-rank-getter compatible
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("<INFO> torch.distributed setup failed, skipping distributed setting..")
    return False


def _setup_dist(backend=None, hostname=None):
    if dist.is_initialized():
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


def _fetch_windows():
    os.sync = lambda: None  # signature: () -> None


def get_rank(group=None):
    if is_available():
        return dist.get_rank(group=group)
    return 0


def get_world_size(group=None):
    if is_available():
        return dist.get_world_size(group=group)
    return 1


def barrier(*args, **kwargs):
    if is_available():
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
    if not is_available():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
