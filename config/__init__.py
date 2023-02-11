from .defaults import DEFAULT_CONFIG


def load_defaults_config():
    return dict(DEFAULT_CONFIG)


def load_defaults_config_key():
    return DEFAULT_CONFIG.keys()


def load_json_config(filename):
    import json
    cfg = load_defaults_config()
    with open(filename) as fp:
        cfg.update(**json.load(fp))
    return cfg


def dump_json_config(cfg, /, filename, *, reduce=True):
    import json
    if reduce:
        cfg = {k: cfg[k] for k, v in DEFAULT_CONFIG.items() if cfg[k] != v}
    else:
        cfg = dict(cfg)
    with open(filename, "w") as fp:
        json.dump(cfg, fp)
