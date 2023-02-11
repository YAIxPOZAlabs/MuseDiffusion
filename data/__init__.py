from . import download  # import order: Top
from . import dataset_wrapper
from . import io
from . import tokenize


# usage: from dataset import load_data_text
def load_data_music(  # # # DiffuSeq에서 사용하는 유일한 함수 # # #
        batch_size,
        seq_len,
        data_dir=None,
        deterministic=False,
        model_emb=None,  # TODO: Model_emb 구현
        split='train',
        num_proc=4,
        loop=True,
        return_raw_loader=False,  # for internal experiment
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
    param return_raw_loader: for experiment.
                             if True, dataloader will be returned despite loop option.
    """

    import os
    import torch
    from datasets import Dataset as ArrowDataset

    from .download import guarantee_data, get_data_dir
    from .io import load_raw_data
    from .dataset_wrapper import EmbeddingWrappedDataset
    from .tokenize import helper_tokenize

    data_dir = get_data_dir(data_dir)
    guarantee_data(data_dir)  # Download data

    print('#' * 30, '\nLoading text data...')

    tokenized_data_path = 'tokenized-{split}-{seq_len}'.format(split=split, seq_len=seq_len)
    tokenized_data_path = os.path.join(data_dir, tokenized_data_path)

    tokenized_data = None
    try:
        if os.path.exists(tokenized_data_path):
            tokenized_data = ArrowDataset.load_from_disk(tokenized_data_path)
            print("Loaded tokenized data from disk successfully.")
    except Exception as exc:
        print(repr(exc))
        print("Loading tokenized data from disk failed, try tokenizing...")
    finally:
        if tokenized_data is None:
            sentence_lst = load_raw_data(data_dir, split=split)
            tokenized_data = helper_tokenize(sentence_lst, seq_len, num_proc=num_proc)
            try:
                tokenized_data.save_to_disk(tokenized_data_path)
            except Exception as exc:
                print(repr(exc))
                print("Saving tokenized data to disk failed, try tokenizing...")
            else:
                print("Saved tokenized data to: {}".format(tokenized_data_path))

    dataset = EmbeddingWrappedDataset(
        tokenized_data,
        model_emb=model_emb
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        # drop_last=True,
    )

    if return_raw_loader:
        import warnings
        warnings.warn("return_raw_loader option will ignore loop option.")
        return data_loader
    elif loop:
        def infinite_loader(iterable):
            while True:
                yield from iterable
        return iter(infinite_loader(data_loader))
    else:
        # print(data_loader)
        return iter(data_loader)
