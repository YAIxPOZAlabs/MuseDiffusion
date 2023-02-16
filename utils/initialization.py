import os
import time

import torch

from models.diffuseq import gaussian_diffusion as gd
from models.diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from models.diffuseq.transformer_model import TransformerNetModel


def random_seed_all(seed):
    import random
    import numpy as np
    import transformers
    for seed_fn in (
            random.seed,
            np.random.seed,
            torch.manual_seed,  # contains torch.cuda.manual_seed_all
            transformers.set_seed,
    ):
        seed_fn(seed)


def load_model_emb(args, sync_weight=True, log_function=print):

    # random emb or pre-defined embedding like glove embedding. You can customize your own init here.
    model = torch.nn.Embedding(args.vocab_size, args.hidden_dim, padding_idx=0)

    # In training, you must synchronize weights of each process.
    # So save it in gpu 0 and load it in other gpus.
    if sync_weight:
        path_save_format = '{}/random_emb.torch'
        if args.resume_checkpoint:
            path_save = path_save_format.format(os.path.dirname(args.resume_checkpoint))
            assert os.path.exists(path_save)
        else:
            path_save = path_save_format.format(args.checkpoint_path)
        path_save_ind = path_save + ".done"
        if int(os.environ.get('LOCAL_RANK', "0")) == 0:
            if os.path.exists(path_save):
                if log_function is not None:
                    log_function('reload the random embeddings {}'.format(model))
                model.load_state_dict(torch.load(path_save))
            else:
                if log_function is not None:
                    log_function('initializing the random embeddings {}'.format(model))
                torch.nn.init.normal_(model.weight)
                torch.save(model.state_dict(), path_save)
                os.sync()
                with open(path_save_ind, "x") as _:
                    pass
        else:
            while not os.path.exists(path_save_ind):
                time.sleep(1)
            if log_function is not None:
                log_function('reload the random embeddings {}'.format(model))
            model.load_state_dict(torch.load(path_save))

    return model


def create_model_and_diffusion(
        *,
        hidden_t_dim,
        hidden_dim,
        vocab_size,
        dropout,
        seq_len,  # FNet Kwarg
        num_hidden_layers,  # FNet Kwarg
        fnet_hidden_dim,  # FNet Kwarg
        fnet_intermediate_dim,  # FNet Kwarg
        diffusion_steps,
        noise_schedule,
        learn_sigma,
        timestep_respacing,
        predict_xstart,
        rescale_timesteps,
        sigma_small,
        rescale_learned_sigmas,
        use_kl,
        **_,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        fnet_hidden_dim=fnet_hidden_dim,  # FNet Kwarg
        fnet_intermediate_dim=fnet_intermediate_dim,  # FNet Kwarg
        hidden_t_dim=hidden_t_dim,
        vocab_size=vocab_size,
        dropout=dropout,
        seq_len=seq_len,  # FNet Kwarg
        num_hidden_layers=num_hidden_layers,  # FNet Kwarg
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion
