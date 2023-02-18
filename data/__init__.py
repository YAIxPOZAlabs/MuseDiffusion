from . import download  # import order: Top
from . import tokenize
from torch.utils.data import Dataset
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    import os
    PathLike = Union[str, os.PathLike]


class MusicDataset(Dataset):
    def __init__(self, music_datasets):
        super().__init__()
        self.music_datasets = music_datasets

    def __len__(self):
        return self.music_datasets.__len__()

    def __getitem__(self, idx):
        out_kwargs = {}
        out_kwargs['input_ids'] = torch.IntTensor(self.music_datasets[idx]['input_ids'])
        out_kwargs['input_mask'] = torch.IntTensor(self.music_datasets[idx]['input_mask'])
        out_kwargs['attention_mask'] = torch.IntTensor(self.music_datasets[idx]['attention_mask'])
        out_kwargs['length'] = self.music_datasets[idx]['length']

        return out_kwargs


# usage: from data import load_data_text
def load_data_music(  # # # DiffuSeq에서 사용하는 유일한 함수 # # #
        batch_size: int,
        seq_len = None,
        data_dir: "PathLike" = None,
        deterministic: bool = False,
        split: str = 'train',
        num_preprocess_proc: int = 4,
        num_loader_proc: int = 0,
        loop: bool = True,
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
:param log_function: custom function for log. default is print.
"""
    from torch.utils.data import DataLoader
    from .tokenize import tokenize_with_caching
    tokenized_data = tokenize_with_caching(data_dir=data_dir, split=split,
                                           num_proc=num_preprocess_proc, log_function=log_function)

    # tokenized_data = tokenized_data.sort("length")
    # batch_sampler = []
    # for i in range(0, len(tokenized_data), batch_size):
    #     batch_sampler.append(list(range(i, i + batch_size)))
    # random.shuffle(batch_sampler)

    data_loader = DataLoader(
        MusicDataset(tokenized_data),
        collate_fn=collate_fn,
        num_workers=num_loader_proc,
        batch_size=batch_size,
        shuffle=not deterministic,
        # batch_sampler=batch_sampler,
    )
    if loop:
        return _infinite_loader(data_loader)
    else:
        return data_loader


def collate_fn(batch_samples, seq_len=None, dtype=None):
    import torch

    seq_len = seq_len or max(sample['length'] for sample in batch_samples)
    batch_len = len(batch_samples)
    shape = (batch_len, seq_len)
    dtype = dtype or torch.int

    input_ids = torch.zeros(shape, dtype=dtype)
    input_mask = torch.ones(shape, dtype=dtype)
    attention_mask = torch.zeros(shape, dtype=dtype)
    length = torch.zeros((batch_len, ), dtype=dtype)

    for idx, batch in enumerate(batch_samples):
        lth = batch['length']
        print(input_ids.shape, batch['input_ids'].shape)
        input_ids[idx][:lth] = batch['input_ids']
        input_mask[idx][:lth] = batch['input_mask']
        attention_mask[idx][:lth] = batch['attention_mask']
        length[idx] = lth

    return {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'attention_mask': attention_mask,
        'length': length
    }


def _infinite_loader(iterable):
    while True:
        yield from iterable
