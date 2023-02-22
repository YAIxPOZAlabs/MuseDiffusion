
def seed_all(seed, deterministic=False):
    import random
    import numpy as np
    import torch
    from data.corruption import generator
    from utils.dist_util import get_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # contains torch.cuda.manual_seed_all
    if deterministic:
        generator.seed(seed)
        torch.backends.cudnn.deterministic = True  # NOQA
        torch.backends.cudnn.benchmark = False  # NOQA
    else:
        generator.seed(int(seed) + get_rank())  # Make corruption's seed differ by node rank


def fetch_pretrained_embedding(args):
    import os
    from utils import dist_util, logger
    if args.pretrained_embedding:
        emb_weight = dist_util.load_state_dict(args.pretrained_embedding)['weight']
        _, orig_hidden_dim = emb_weight.shape
        if orig_hidden_dim != args.hidden_dim:
            logger.warn(
                f"Pretrained embedding {os.path.basename(args.pretrained_embedding)}'s "
                f"hidden_dim {orig_hidden_dim} is differ from "
                f"config's hidden dim {args.hidden_dim}.\n"
                f"args.hidden_dim and args.fnet_hidden_dim will be overwritten into"
                f"pretrained embedding's hidden dim {orig_hidden_dim}"
            )
            args.hidden_dim = args.fnet_hidden_dim = orig_hidden_dim
        return emb_weight
    else:
        if args.freeze_embedding:
            import argparse
            raise argparse.ArgumentTypeError(
                "Cannot turn --freeze_embedding on without --pretrained_embedding!"
            )
        return


def overload_embedding(model, emb_weight, freeze_embedding):
    from utils import logger
    import torch
    orig_vocab_size, _ = emb_weight.shape
    assert model.word_embedding.weight.shape[0] == orig_vocab_size
    with torch.no_grad():
        model.word_embedding.weight.data[:orig_vocab_size] = emb_weight
    if freeze_embedding:
        model.word_embedding.requires_grad_(False)
    logger.log("### Successfully overloaded pretrained embedding weight.")
    return model


def create_model_and_diffusion(
        *,
        hidden_t_dim,
        hidden_dim,
        vocab_size,
        dropout,
        seq_len,  # FNet Kwarg
        num_layers,  # FNet Kwarg
        intermediate_dim,  # FNet Kwarg
        diffusion_steps,
        noise_schedule,
        learn_sigma,
        timestep_respacing,
        predict_xstart,
        rescale_timesteps,
        sigma_small,
        rescale_learned_sigmas,
        num_attention_heads,
        use_kl,
        **_,
):
    from models.diffusion.gaussian_diffusion \
        import SpacedDiffusion, space_timesteps, get_named_beta_schedule
    from models.diffusion.transformer_model import TransformerNetModel

    model = TransformerNetModel(
        input_dims=hidden_dim,
        hidden_dim = hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        hidden_t_dim=hidden_t_dim,
        vocab_size=vocab_size,
        intermediate_dim= intermediate_dim, 
        seq_len=seq_len,
        num_layers=num_layers, 
        dropout=dropout,
        num_attention_heads = num_attention_heads         
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)

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
