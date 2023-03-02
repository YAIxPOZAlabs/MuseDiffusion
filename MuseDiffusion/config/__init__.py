from .defaults import DEFAULT_CONFIG
from .choices import CHOICES


def load_defaults_config():
    return dict(DEFAULT_CONFIG)


def load_dict_config(config_dict):
    from MuseDiffusion.utils.argument_util import str2bool
    cfg = load_defaults_config()
    for key in list(config_dict):
        expected_value = DEFAULT_CONFIG[key]
        if isinstance(expected_value, bool):
            expected_type = (str, bool)
            constructor = str2bool
        else:
            expected_type = type(expected_value)
            constructor = None
        real_value = config_dict[key]
        if not isinstance(real_value, expected_type):
            raise TypeError("Invalid config type: {}".format(key))
        if constructor is not None:
            config_dict[key] = constructor(real_value)
    cfg.update(**config_dict)
    return cfg


def load_json_config(filename):
    import json
    with open(filename) as fp:
        config_dict = json.load(fp)
    return load_dict_config(config_dict)


def reduce_dict_config(cfg):
    return {k: cfg[k] for k, v in DEFAULT_CONFIG.items() if cfg[k] != v}


def dump_json_config(cfg, filename, *, indent=2, reduce=True):
    import json
    if hasattr(filename, 'write'):
        context = None
    else:
        context = filename = open(filename, "w")
    try:
        json.dump((reduce_dict_config if reduce else dict)(cfg), filename, indent=indent)
    finally:
        if context is not None:
            context.close()
