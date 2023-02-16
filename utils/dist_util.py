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


@functools.lru_cache(maxsize=None)
def is_available():
    if os.name == 'nt' and not USE_DIST_IN_WINDOWS:
        if os.environ.get("LOCAL_RANK", str(0)) == str(0):
            import warnings
            warnings.warn(
                "In Windows, Distributed is unavailable by default settings. "
                "To enable Distributed, edit utils.dist_util.USE_DIST_IN_WINDOWS to True."
            )
        os.sync = lambda: None
    elif dist.is_available():  # All condition passed
        return True
    os.environ.setdefault("LOCAL_RANK", str(0))
    return False


def setup_dist():
    """
    Setup a distributed process group.
    """
    if not is_available() or dist.is_initialized():
        return

    backend = "gloo" if not _cuda_available() else "nccl"

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
