import os
import time
import shutil
import numpy as np
from datasets import Dataset as ArrowDataset


def load_raw_data(data_dir=None, split='train'):

    from .download import get_data_dir

    if split == 'train':
        print('### Loading from the TRAIN set...')
        src = np.load('{}/input_train.npy'.format(get_data_dir(data_dir)), allow_pickle=True)
        trg = np.load('{}/target_train.npy'.format(get_data_dir(data_dir)), allow_pickle=True)
    elif split in ('valid', 'test'):
        print('### Loading from the VALID set...')
        src = np.load('{}/input_val.npy'.format(get_data_dir(data_dir)), allow_pickle=True)
        trg = np.load('{}/target_val.npy'.format(get_data_dir(data_dir)), allow_pickle=True)
    else:
        assert False, "invalid split for dataset"

    return {'src': src, 'trg': trg}


def helper_tokenize(sentence_lst, end_token=1, num_proc=4):

    def merge_and_mask(group_lst):

        lst = []
        mask = []
        attn_mask = []
        length = []

        for i in range(len(group_lst['src'])):

            src = group_lst['src'][i]
            trg = group_lst['trg'][i]
            src_eos_len = len(src) + 1
            trg_len = len(trg)
            src_eos_trg_len = src_eos_len + trg_len

            lst.append([*src, end_token, *trg])
            mask.append([*(0 for _ in range(src_eos_len)), *(1 for _ in range(trg_len))])
            attn_mask.append([1 for _ in range(src_eos_trg_len)])
            length.append(src_eos_trg_len)

        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        group_lst['attention_mask'] = attn_mask
        group_lst['length'] = length

        return group_lst

    return ArrowDataset.from_dict(sentence_lst).map(
        merge_and_mask,
        batched=True,
        num_proc=num_proc,
        remove_columns=['src', 'trg'],
        desc="merge and mask",
    )


def helper_filter(merged_data, seq_len, num_proc=4):
    return merged_data.filter(
        lambda group_lst: [length <= seq_len for length in group_lst['length']],
        batched=True,
        num_proc=num_proc,
        desc="filter datas by [length <= {}]".format(seq_len),
    )


def tokenize_with_caching(  # Tokenized Data I/O Wrapper for Distributed Learning
        *,
        split,
        data_dir,
        seq_len,
        num_proc,
):

    from .download import guarantee_data, get_data_dir

    data_dir = get_data_dir(data_dir)

    assert split.lower() in ('train', 'valid', 'test')
    if split.lower() == 'test':
        split = 'valid'

    def _getter_merge():
        guarantee_data(data_dir)  # Download data
        print("### Merging {split} data".format(split=split.upper()))
        sentence_lst = load_raw_data(data_dir, split=split)
        return helper_tokenize(sentence_lst, num_proc=num_proc)

    merged_data_path = 'merged-{split}'.format(split=split.lower())
    merged_data_path = os.path.join(data_dir, merged_data_path)

    def _getter_filter():
        merged_data = _load_arrow(getter=_getter_merge, path=merged_data_path)
        return helper_filter(merged_data, seq_len=seq_len)

    filtered_data_path = 'filtered-{split}-{seq_len}'.format(split=split.lower(), seq_len=seq_len)
    filtered_data_path = os.path.join(data_dir, filtered_data_path)

    if seq_len < 2096:
        return _load_arrow(getter=_getter_filter, path=filtered_data_path)
    else:
        return _load_arrow(getter=_getter_merge, path=merged_data_path)


def _load_arrow(*, getter=None, path=None):
    """Data I/O Wrapper for Distributed Learning"""
    base, name = os.path.split(path)
    lock_path = os.path.join(base, name + ".lock")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if os.path.exists(path):
            data = ArrowDataset.load_from_disk(path)
        else:
            data = getter()
            with open(lock_path, "w") as _:
                pass
            print("### Saving into {}".format(path))
            try:
                data.save_to_disk(path)
                os.sync()
            except BaseException:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                raise
            finally:
                os.remove(lock_path)
    else:
        while not os.path.exists(path) or os.path.exists(lock_path):
            time.sleep(1)
        data = ArrowDataset.load_from_disk(path)
    return data


__all__ = ('load_raw_data', 'helper_tokenize', 'tokenize_with_caching')
