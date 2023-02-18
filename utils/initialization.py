import os
import time

import torch

from models.diffuseq import gaussian_diffusion as gd
from models.diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from models.diffuseq.transformer_model import TransformerNetModel


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
