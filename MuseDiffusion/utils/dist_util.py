"""
Helpers for distributed training.
"""

import io
import os
import socket
import functools

import blobfile as bf

import torch
import torch.distributed as dist
from torch.cuda import is_available as _cuda_available


USE_DIST_IN_WINDOWS = False  # Change this to enable torch.distributed in windows.


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
                "To enable Distributed, edit MuseDiffusion.utils.dist_util.USE_DIST_IN_WINDOWS to True."
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
def setup_dist(backend=None, silent=False):
    """
    Setup a distributed process group.
    """

    if is_initialized():
        return True

    if is_available() and os.environ.get("LOCAL_RANK") is not None:
        try:
            if backend is None:
                backend = "gloo" if not _cuda_available() else "nccl"
            dist.init_process_group(backend=backend, init_method="env://")
            if _cuda_available():
                torch.cuda.set_device(dev())
                torch.cuda.empty_cache()
            if os.environ["LOCAL_RANK"] == str(0) and not silent:
                print("<INFO> torch.distributed setup success, using distributed setting..")
            return True
        except Exception as exc:
            if not silent:
                print(f"<INFO> {exc.__class__.__qualname__}: {exc}")
            is_available.cache = False

    os.environ.setdefault("LOCAL_RANK", str(0))
    if int(os.getenv("LOCAL_RANK")) == 0 and not silent:
        print("<INFO> torch.distributed is not available, skipping distributed setting..")
    return False


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
        return torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params, src=0):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not is_initialized():
        return
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, src)


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
