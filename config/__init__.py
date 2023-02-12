from .defaults import DEFAULT_CONFIG
from .choices import CHOICES


def load_defaults_config():
    return dict(DEFAULT_CONFIG)


def load_defaults_config_key():
    return DEFAULT_CONFIG.keys()


def load_dict_config(config_dict):
    cfg = load_defaults_config()
    for key in config_dict:
        expected_value = DEFAULT_CONFIG[key]
        if isinstance(expected_value, bool):
            expected_type = (str, bool)
        else:
            expected_type = type(expected_value)
        if not isinstance(config_dict[key], expected_type):
            raise TypeError("Invalid config type: {}".format(key))
    cfg.update(**config_dict)
    return cfg


def load_json_config(filename):
    import json
    with open(filename) as fp:
        config_dict = json.load(fp)
    return load_dict_config(config_dict)


def reduce_dict_config(cfg):
    return {k: cfg[k] for k, v in DEFAULT_CONFIG.items() if cfg[k] != v}


def dump_json_config(cfg, /, filename, *, reduce=True):
    import json
    with open(filename, "w") as fp:
        json.dump((reduce_dict_config if reduce else dict)(cfg), fp)
