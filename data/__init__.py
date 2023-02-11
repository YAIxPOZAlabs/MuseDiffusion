from . import download  # import order: Top
from . import dataset_wrapper
from . import io
from . import tokenize

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from torch.nn import Embedding
    import os
    PathLike = Union[str, os.PathLike]
else:
    PathLike = Embedding = None


# usage: from data import load_data_text
def load_data_music(  # # # DiffuSeq에서 사용하는 유일한 함수 # # #
        batch_size: int,
        seq_len: int,
        data_dir: PathLike = None,
        deterministic: bool = False,
        model_emb: Embedding = None,  # TODO: Model_emb 구현
        split: str = 'train',
        num_proc: int = 4,
        loop: bool = True
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    param batch_size: the batch size of each returned pair.
    param seq_len: the max sequence length (one-side).
    param deterministic: if True, yield results in a deterministic order.
    param data_dir: data directory.
    param model_emb: loaded word embeddings.
    param split: how to split data - train, or valid.
    param num_proc: num of worker while tokenizing.
    param loop: loop to get batch data or not.
        if loop is True - infinite iterator will be returned
        if loop is False - default iterator will be returned
        if loop is None - raw dataloader will be returned
    """
    tokenized_data = _tokenize_data(seq_len=seq_len, data_dir=data_dir, split=split, num_proc=num_proc)
    data_loader = _wrap_data_loader(tokenized_data,
                                    batch_size=batch_size, deterministic=deterministic, model_emb=model_emb)
    if loop is None:  # Option for development
        return data_loader
    elif loop is True:
        return _infinite_loader(data_loader)
    elif loop is False:
        return iter(data_loader)
    else:
        raise TypeError("loop argument expected to be bool or NoneType, got {!r}".format(loop))


def _wrap_data_loader(
        tokenized_data,
        *,
        batch_size,
        deterministic,
        model_emb,
):

    from torch.utils.data import DataLoader
    from .dataset_wrapper import EmbeddingWrappedDataset

    data_loader = DataLoader(
        EmbeddingWrappedDataset(tokenized_data, model_emb=model_emb),
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        # drop_last=True,
    )
    return data_loader


def _tokenize_data(
        *,
        seq_len,
        data_dir,
        split,
        num_proc,
):

    import os
    from datasets import Dataset as ArrowDataset

    from .download import guarantee_data, get_data_dir
    from .io import load_raw_data
    from .dataset_wrapper import EmbeddingWrappedDataset
    from .tokenize import helper_tokenize

    data_dir = get_data_dir(data_dir)
    guarantee_data(data_dir)  # Download data

    print('#' * 30, '\nLoading {split} text data...'.format(split=split.upper()))

    tokenized_data_path = 'tokenized-{split}-{seq_len}'.format(split=split.lower(), seq_len=seq_len)
    tokenized_data_path = os.path.join(data_dir, tokenized_data_path)

    tokenized_data = None
    try:
        if os.path.exists(tokenized_data_path):
            tokenized_data = ArrowDataset.load_from_disk(tokenized_data_path)
            print("Loaded tokenized {split} data from disk successfully.".format(split=split.upper()))
    except Exception as exc:
        print(repr(exc))
        print("Loading tokenized {split} data from disk failed, try tokenizing...".format(split=split.upper()))
    finally:
        if tokenized_data is None:
            sentence_lst = load_raw_data(data_dir, split=split)
            tokenized_data = helper_tokenize(sentence_lst, seq_len, num_proc=num_proc)
            try:
                tokenized_data.save_to_disk(tokenized_data_path)
            except Exception as exc:
                print(repr(exc))
                print("Saving tokenized {split} data to disk failed, try tokenizing...".format(split=split.upper()))
            else:
                print("Saved tokenized {split} data to: {path}".format(split=split.upper(), path=tokenized_data_path))

    return tokenized_data


def _infinite_loader(iterable):
    while True:
        yield from iterable
