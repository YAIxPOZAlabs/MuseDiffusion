def load_raw_data(data_dir=None, split='train'):
    import numpy as np
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

    from datasets import Dataset as ArrowDataset

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


def tokenize_with_caching(  # Tokenized Data I/O Wrapper for Distributed Learning
        *,
        data_dir,
        split,
        num_proc,
        log_function=print
):
    import os
    import time
    import shutil
    from datasets import Dataset as ArrowDataset

    from .download import guarantee_data, get_data_dir

    data_dir = get_data_dir(data_dir)
    guarantee_data(data_dir)  # Download data

    assert split.lower() in ('train', 'valid', 'test')
    if split.lower() == 'test':
        split = 'valid'

    tokenized_data_path = 'merged-{split}'.format(split=split.lower())
    tokenized_data_lock_path = tokenized_data_path + '.lock'
    tokenized_data_path = os.path.join(data_dir, tokenized_data_path)
    tokenized_data_lock_path = os.path.join(data_dir, tokenized_data_lock_path)

    if int(os.environ.get('LOCAL_RANK', "0")) == 0:
        if os.path.exists(tokenized_data_path):
            if log_function is not None:
                log_function("Loading processed {split} data from disk".format(split=split.upper()))
            tokenized_data = ArrowDataset.load_from_disk(tokenized_data_path)
        else:
            sentence_lst = load_raw_data(data_dir, split=split)
            tokenized_data = helper_tokenize(sentence_lst, num_proc=num_proc)
            with open(tokenized_data_lock_path, "w") as _:
                pass
            if log_function is not None:
                log_function(
                    "Saving processed {split} data to {path}".format(split=split.upper(), path=tokenized_data_path)
                )
            try:
                tokenized_data.save_to_disk(tokenized_data_path)
                os.sync()
            except BaseException:
                if os.path.isdir(tokenized_data_path):
                    shutil.rmtree(tokenized_data_path)
                raise
            finally:
                os.remove(tokenized_data_lock_path)
    else:
        while not os.path.exists(tokenized_data_path) or os.path.exists(tokenized_data_lock_path):
            time.sleep(1)
        if log_function is not None:
            log_function("Loading tokenized {split} data from disk".format(split=split.upper()))
        tokenized_data = ArrowDataset.load_from_disk(tokenized_data_path)

    return tokenized_data


__all__ = ('load_raw_data', 'helper_tokenize', 'tokenize_with_caching')
