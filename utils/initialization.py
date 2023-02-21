
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


def load_and_fetch_pretrained_embedding(args):
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
        return


def overload_embedding(model, emb_weight):
    from utils import logger
    import torch
    orig_vocab_size, _ = emb_weight.shape
    assert model.word_embedding.weight.shape[0] >= orig_vocab_size
    with torch.no_grad():
        model.word_embedding.weight.data[:orig_vocab_size] = emb_weight
    logger.log("### Successfully overloaded pretrained embedding weight.")
    return model


def create_model_and_diffusion(
        *,
        hidden_t_dim,
        hidden_dim,
        vocab_size,
        dropout,
        seq_len,  # FNet Kwarg
        num_fnet_layers,  # FNet Kwarg
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
    from models.diffusion.gaussian_diffusion \
        import SpacedDiffusion, space_timesteps, get_named_beta_schedule
    from models.diffusion.transformer_model import TransformerNetModel

    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        fnet_hidden_dim=fnet_hidden_dim,  # FNet Kwarg
        fnet_intermediate_dim=fnet_intermediate_dim,  # FNet Kwarg
        hidden_t_dim=hidden_t_dim,
        vocab_size=vocab_size,
        dropout=dropout,
        seq_len=seq_len,  # FNet Kwarg
        num_fnet_layers=num_fnet_layers,  # FNet Kwarg
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
