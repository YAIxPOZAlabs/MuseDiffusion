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
    parser.add_argument('--num_proc', type=int, default=4, help='max len of input sequence')

    args = parser.parse_args()

    from data.preprocess import tokenize_with_caching as main
    for split in ('train', 'valid'):
        main(split=split, data_dir=args.data_dir, num_proc=args.num_proc)
