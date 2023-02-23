def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='dumps example config or given json config')
    parser.add_argument('--load_from', type=str, required=False,
                        help='json config path to load from. if not given, default config will be loaded.')
    parser.add_argument('--save_into', type=str, required=False,
                        help='json config path to dump into. default is stdout.')
    return parser


def main(args):
    import sys
    from config import load_json_config, dump_json_config, load_defaults_config
    config_to_dump = load_json_config(args.load_from) if args.load_from else load_defaults_config()
    dump_json_config(config_to_dump, args.save_into or sys.stdout, reduce=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(create_parser().parse_args()))
