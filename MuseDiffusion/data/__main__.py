if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Data preparing args.')
    parser.add_argument('--data_dir', type=str, default='', help='path to training dataset')
    parser.add_argument('--num_proc', type=int, default=4, help='max len of input sequence')

    args = parser.parse_args()

    from .preprocess import tokenize_with_caching as main
    for split in ('train', 'valid'):
        main(split=split, data_dir=args.data_dir or None, num_proc=args.num_proc, seq_len=float('inf'))
