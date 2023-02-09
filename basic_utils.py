import argparse
import torch
import json, os
import time

from models.diffuseq import gaussian_diffusion as gd
from models.diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from models.diffuseq.transformer_model import TransformerNetModel


def load_model_emb(args):
    # random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(args.vocab_size, args.hidden_dim, padding_idx = 0)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    if int(os.environ['LOCAL_RANK']) == 0:
        if os.path.exists(path_save):
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            print('initializing the random embeddings', model)
            torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model


def load_defaults_config():
    from copy import deepcopy
    from config import default
    return deepcopy(default)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
