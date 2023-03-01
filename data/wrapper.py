import torch
import datasets

from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing import *


def wrap_dataset(
        processed_data: datasets.Dataset,
        *,
        use_bucketing: bool,
        seq_len: int,
        batch_size: int,
        deterministic: bool,
        num_loader_proc: int,
        corruption: "Optional[Callable]",
):

    assert isinstance(processed_data, datasets.Dataset)

    dataset = MidiSequenceDataset(processed_data, corruption=corruption)

    # processed_data = processed_data.sort("length")
    # batch_sampler = []
    # for i in range(0, len(processed_data), batch_size):
    #     batch_sampler.append(list(range(i, min(i + batch_size, len(processed_data))))
    # random.shuffle(batch_sampler)

    if use_bucketing:
        collate_fn = collate_batches
    else:
        from functools import partial
        collate_fn = partial(collate_batches, seq_len=seq_len)

    data_loader = torch.utils.data.DataLoader(  # NOQA
        dataset,
        collate_fn=collate_fn,
        num_workers=num_loader_proc,
        batch_size=batch_size,
        shuffle=not deterministic,
        persistent_workers=num_loader_proc > 0,
        # batch_sampler=batch_sampler,
    )
    return data_loader


class MidiSequenceDataset(torch.utils.data.Dataset[datasets.Dataset], datasets.Dataset):  # NOQA

    def __init__(self, dataset: "datasets.Dataset", corruption: "Callable" = None):  # NOQA
        expected_column_names = {'input_ids', 'input_mask', 'label', 'length'}
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("argument dataset must be instance of datasets.Dataset!")
        elif set(dataset.column_names) != expected_column_names:
            raise TypeError("argument dataset's columns must be {},\ngot {}"
                            .format(", ".join(expected_column_names), ", ".join(dataset.column_names)))
        elif corruption is not None and not callable(corruption):
            raise TypeError("corruption must be callable!")
        self.__dict__.update(dataset.__dict__)  # make data pointer same
        datasets.Dataset.set_format(
            self,
            type='torch',
            columns=['input_ids', 'input_mask', 'label'],
            output_all_columns=True
        )
        self.corruption = corruption

    def set_format(self, *args, **kwargs):
        raise NotImplementedError

    @overload
    def __getitem__(self, key: "Union[int, slice, Iterable[int]]") -> "Dict":
        ...

    @overload
    def __getitem__(self, key: str) -> "List":
        ...

    def __getitem__(self, key):
        result = self._getitem(key)
        if self.corruption is not None:
            if isinstance(key, str):
                import warnings
                warnings.warn("getitem with string key will not apply corruptions.")
            elif isinstance(key, int):
                correct_ids = result['input_ids']
                result['correct_ids'] = correct_ids
                result['input_ids'] = self.corruption(correct_ids)
            else:
                correct_ids = result['input_ids']
                result['correct_ids'] = correct_ids
                result['input_ids'] = [self.corruption(cid) for cid in correct_ids]
        return result


def collate_batches(batch_samples, seq_len=None, dtype=None):
    # type: (List[Union[int, torch.Tensor]], int, torch.dtype) -> ...

    seq_len = seq_len or max(sample['length'] for sample in batch_samples)
    batch_len = len(batch_samples)
    shape = (batch_len, seq_len)
    dtype = dtype or torch.long
    has_corruption = 'correct_ids' in batch_samples[0]

    correct_ids = torch.zeros(shape, dtype=dtype) if has_corruption else None
    input_ids = torch.zeros(shape, dtype=dtype)
    input_mask = torch.ones(shape, dtype=dtype)
    label = torch.zeros(shape, dtype=dtype)
    length = torch.zeros((batch_len, ), dtype=dtype)

    for idx, batch in enumerate(batch_samples):
        lth = batch['length']
        if has_corruption:
            correct_ids[idx][:lth] = batch['correct_ids']
        input_ids[idx][:lth] = batch['input_ids']
        input_mask[idx][:lth] = batch['input_mask']
        label[idx][:lth] = batch['label']
        length[idx] = lth

    result = {
        'correct_ids': correct_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'label': label,
        'length': length
    }
    if not has_corruption:
        result.pop('correct_ids')
    return result


__all__ = ('wrap_dataset', )
