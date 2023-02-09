import numpy as np


def load_raw_data(data_args, split='train'):
    from .download import get_data_path

    if split == 'train':
        print('### Loading form the TRAIN set...')
        src = np.load('{}/input_train.npy'.format(get_data_path(data_args)), allow_pickle=True)
        trg = np.load('{}/target_train.npy'.format(get_data_path(data_args)), allow_pickle=True)
    elif split in ('valid', 'test'):
        print('### Loading form the VALID set...')
        src = np.load('{}/input_val.npy'.format(get_data_path(data_args)), allow_pickle=True)
        trg = np.load('{}/target_val.npy'.format(get_data_path(data_args)), allow_pickle=True)
    else:
        assert False, "invalid split for dataset"

    return {'src': src, 'trg': trg}


__all__ = ('load_raw_data', )
