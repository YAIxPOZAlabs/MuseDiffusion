def wrap_dataset(
        processed_data,
        *,
        seq_len: int,
        batch_size: int,
        deterministic: bool,
        num_loader_proc: int,
):

    from functools import partial
    from torch.utils.data import DataLoader
    from datasets import Dataset as ArrowDataset

    assert isinstance(processed_data, ArrowDataset)
    processed_data.set_format(
        type='torch',
        columns=['input_ids', 'input_mask', 'attention_mask'],
        output_all_columns=True
    )

    # processed_data = processed_data.sort("length")
    # batch_sampler = []
    # for i in range(0, len(processed_data), batch_size):
    #     batch_sampler.append(list(range(i, min(i + batch_size, len(processed_data))))
    # random.shuffle(batch_sampler)

    if seq_len is not None:
        collate_fn = partial(_collate_fn, seq_len=seq_len)
    else:
        collate_fn = _collate_fn

    data_loader = DataLoader(
        processed_data,
        collate_fn=collate_fn,
        num_workers=num_loader_proc,
        batch_size=batch_size,
        shuffle=not deterministic,
        # batch_sampler=batch_sampler,
    )
    return data_loader


def _collate_fn(batch_samples, seq_len=None, dtype=None):

    import torch

    seq_len = seq_len or max(sample['length'] for sample in batch_samples)
    batch_len = len(batch_samples)
    shape = (batch_len, seq_len)
    dtype = dtype or torch.long

    input_ids = torch.zeros(shape, dtype=dtype)
    input_mask = torch.ones(shape, dtype=dtype)
    attention_mask = torch.zeros(shape, dtype=dtype)
    length = torch.zeros((batch_len, ), dtype=dtype)

    for idx, batch in enumerate(batch_samples):
        lth = batch['length']
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


__all__ = ('wrap_dataset', )