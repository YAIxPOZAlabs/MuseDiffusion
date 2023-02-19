from . import download  # import order: Top
from . import preprocess
from . import wrapper

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    import os


# usage: from data import load_data_text
def load_data_music(  # # # DiffuSeq에서 사용하는 유일한 함수 # # #
        batch_size: int,
        seq_len: "Optional[int]" = None,
        data_dir: "Union[str, os.PathLike]" = None,
        deterministic: bool = False,
        split: str = 'train',
        num_preprocess_proc: int = 4,
        num_loader_proc: int = 0,
        loop: bool = True,
        corruption: "Optional[dict]" = None,
        log_function: "Callable" = print
):
    """
For a dataset, create a generator over (seqs, kwargs) pairs.

Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
more keys, each of which map to a batched Tensor of their own.
The kwargs dict can be used for some meta information.

:param batch_size: the batch size of each returned pair.
:param seq_len: the max sequence length (one-side).
:param deterministic: if True, yield results in a deterministic order.
:param data_dir: data directory.
:param split: how to split data - train, or valid.
:param num_preprocess_proc: num of worker while tokenizing.
:param num_loader_proc: num of worker for data loader.
:param loop: loop to get batch data or not.
    if loop is True - infinite iterator will be returned
    if loop is False - default iterator will be returned
    if loop is None - raw dataloader will be returned
:param corruption: option for corruption (key: cor_func, max_cor, cor_func's value: mt, mn, rn, rr)
:param log_function: custom function for log. default is print.
"""
    from .preprocess import tokenize_with_caching
    from .wrapper import wrap_dataset

    if corruption is not None:
        from .corruption import Get_corruption
        corruption = Get_corruption(corruption)

    tokenized_data = tokenize_with_caching(
        data_dir=data_dir,
        split=split,
        num_proc=num_preprocess_proc,
        log_function=log_function
    )
    data_loader = wrap_dataset(
        tokenized_data,
        batch_size=batch_size,
        seq_len=seq_len,
        deterministic=deterministic,
        corruption=corruption,
        num_loader_proc=num_loader_proc
    )
    if loop:
        return _infinite_loader(data_loader)
    else:
        return data_loader


def _infinite_loader(iterable):
    while True:
        yield from iterable
