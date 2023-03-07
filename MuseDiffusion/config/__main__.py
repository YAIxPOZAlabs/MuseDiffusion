def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='dumps example config or given json config')
    parser.add_argument('--setting_name', type=str, default='train', choices=['train', 'sample'],
                        help='choose setting from train or sample.')
    parser.add_argument('--load_from', type=str, required=False,
                        help='json config path to load from. if not given, default config will be loaded.')
    parser.add_argument('--save_into', type=str, required=False,
                        help='json config path to dump into. default is stdout.')
    return parser


def main(args):
    import sys
    import json
    from . import TrainSettings, SamplingSettings
    klass = {'train': TrainSettings, 'sample': SamplingSettings}[args.setting_name]
    config = klass.parse_file(args.load_from) if args.load_from else klass()
    fp = open(args.save_into, "w") if args.save_into else sys.stdout
    with fp:
        json.dump(config.dict(), fp, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(create_parser().parse_args()))
