import argparse


def add_dict_to_argparser(parser, default_dict, choices):
    if choices is None:
        choices = {}
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        if k in choices:
            kwargs = dict(default=v, type=v_type, choices=choices[k])
        else:
            kwargs = dict(default=v, type=v_type)
        parser.add_argument(f"--{k}", **kwargs)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
