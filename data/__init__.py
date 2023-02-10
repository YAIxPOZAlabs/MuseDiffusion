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
        loop=True,
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
    :param model_emb: loaded word embeddings.
    :param loop: loop to get batch data or not.
    """

    import torch

    from .download import guarantee_data
    from .io import load_raw_data
    from .dataset_wrapper import EmbeddingWrappedDataset
    from .tokenize import helper_tokenize

    guarantee_data(data_dir)  # Download data

    print('#' * 30, '\nLoading text data...')

    sentence_lst = load_raw_data(data_dir, split=split)
    training_data = helper_tokenize(sentence_lst, seq_len)

    dataset = EmbeddingWrappedDataset(
        training_data,
        model_emb=model_emb
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        # drop_last=True,
    )

    if loop:
        def infinite_loader(iterable):
            while True:
                yield from iterable
        return iter(infinite_loader(data_loader))
    else:
        # print(data_loader)
        return iter(data_loader)
