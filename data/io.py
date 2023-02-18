import numpy as np


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
        
    length = get_length(src, trg) 
    return {'src': src, 'trg': trg, 'length': length }

def get_length(src, trg):
    length = []
    for i in range(len(trg)):
        length.append(len(trg[i])+len(src[i]))
    return length

__all__ = ('load_raw_data', )