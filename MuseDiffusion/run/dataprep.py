# python3 MuseDiffusion/run/dataprep.py
from MuseDiffusion.config import DataPrepSettings


def create_parser():
    return DataPrepSettings.to_argparse()


def main(namespace):

    # Create config from parsed argument namespace
    args: DataPrepSettings = DataPrepSettings.from_argparse(namespace)

    # Import everything
    from MuseDiffusion.data import tokenize_with_caching
    from MuseDiffusion.utils import dist_util

    # This script does not require torch.distributed
    dist_util.setup_dist(silent=True)
    if dist_util.is_initialized():
        import warnings
        warnings.warn(
            "Data-prep process does not require torch.distributed."
            "Pipline runs in only top node."
        )

    # Prepare Dataset
    for split in ('train', 'valid'):
        tokenize_with_caching(
            split=split,
            data_dir=args.data_dir or None,
            num_proc=args.num_proc,
            seq_len=float('inf')
        )


if __name__ == '__main__':
    main(create_parser().parse_args())
