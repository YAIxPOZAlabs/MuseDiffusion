if __name__ == '__main__':

    import sys
    import os
    import argparse

    # set working dir to the upper folder
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    sys.path.append(dname)  # Assure upper folder import
    os.chdir(dname)

    from config import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description='Data preparing args.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'], help='path to training dataset')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_CONFIG['seq_len'], help='max len of input sequence')
    parser.add_argument('--num_proc', type=int, default=4, help='max len of input sequence')

    args = parser.parse_args()

    from data import _tokenize_data  # pylint: disable=protected-access
    for split in ('train', 'valid'):
        _tokenize_data(split=split, seq_len=args.seq_len, data_dir=args.data_dir, num_proc=args.num_proc)
