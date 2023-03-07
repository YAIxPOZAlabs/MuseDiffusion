if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='dumps example config or given json config')
    parser.add_argument('--setting_name', type=str, default='train', choices=['train', 'seq2seq', 'generation'],
                        help='choose setting from train or sample.')
    parser.add_argument('--load_from', type=str, required=False,
                        help='json config path to load from. if not given, default config will be loaded.')
    parser.add_argument('--save_into', type=str, required=False,
                        help='json config path to dump into. default is stdout.')
    args, argv = parser.parse_known_args()
    from . import TrainSettings, Seq2seqSettings, GenerationSettings
    klass_map = {'train': TrainSettings, 'seq2seq': Seq2seqSettings, 'generation': GenerationSettings}
    klass = klass_map[args.setting_name]
    if args.load_from:
        config = klass.parse_file(args.load_from)
    elif argv:
        parser = argparse.ArgumentParser(prog=args.setting_name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        config = klass.from_argparse(klass.to_argparse(parser).parse_args(argv))
    else:
        config = klass()
    dumps = json.dumps(config.dict(), indent=2)
    if args.save_into:
        with open(args.save_into, "w") as fp:
            print(dumps, file=fp)
    else:
        print(dumps)
