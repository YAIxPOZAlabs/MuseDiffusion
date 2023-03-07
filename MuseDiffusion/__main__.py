from MuseDiffusion.config import TrainSettings, GenerationSettings, ModificationSettings, DataPrepSettings


def create_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as Df
    parser = ArgumentParser(prog="python3 -m MuseDiffusion", formatter_class=Df)
    subparsers = parser.add_subparsers(title="subcommand", dest="subcommand", required=True,
                                       help="available subcommands. "
                                            "type each subcommand followed up by --help, to show full usage.")
    TrainSettings.to_argparse(subparsers.add_parser("train", formatter_class=Df), add_json=True)
    GenerationSettings.to_argparse(subparsers.add_parser("generation", formatter_class=Df))
    ModificationSettings.to_argparse(subparsers.add_parser("modification", formatter_class=Df))
    DataPrepSettings.to_argparse(subparsers.add_parser("dataprep", formatter_class=Df))
    return parser


def main(namespace):
    subcommand = namespace.__dict__.pop("subcommand")
    if subcommand == "train":
        from MuseDiffusion.run.train import main as main_
        main_(namespace)
    if subcommand == "generation":
        namespace.mode = "generation"
        from MuseDiffusion.run.sample import main as main_
        main_(namespace)
    if subcommand == "modification":
        namespace.mode = "modification"
        from MuseDiffusion.run.sample import main as main_
        main_(namespace)
    if subcommand == "dataprep":
        from MuseDiffusion.run.dataprep import main as main_
        main_(namespace)


if __name__ == "__main__":
    from MuseDiffusion.utils.dist_run import parse_and_autorun
    main(parse_and_autorun(create_parser(), module_name="MuseDiffusion"))
