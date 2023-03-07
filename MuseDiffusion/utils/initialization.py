def seed_all(seed, deterministic=False):
    import random
    import numpy as np
    import torch
    from ..data.corruption import generator
    from .dist_util import get_rank
    if deterministic:
        seed = int(seed)
        torch.backends.cudnn.deterministic = True  # NOQA
        torch.backends.cudnn.benchmark = False  # NOQA
    else:
        seed = int(seed) + get_rank()  # Make seed differ by node rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # contains torch.cuda.manual_seed_all
    generator.seed(seed)


def fetch_pretrained_embedding(args):  # Returns single parameter
    import os
    from . import dist_util, logger
    if args.pretrained_embedding:
        emb_weight = dist_util.load_state_dict(args.pretrained_embedding)['weight']
        _, orig_hidden_dim = emb_weight.shape
        if orig_hidden_dim != args.hidden_dim:
            logger.warn(
                f"Pretrained embedding {os.path.basename(args.pretrained_embedding)}'s "
                f"hidden_dim {orig_hidden_dim} is differ from "
                f"config's hidden dim {args.hidden_dim}.\n"
                f"args.hidden_dim will be overwritten into"
                f"pretrained embedding's hidden dim {orig_hidden_dim}"
            )
            args.hidden_dim = orig_hidden_dim
        return emb_weight
    else:
        if args.freeze_embedding:
            import argparse
            raise argparse.ArgumentTypeError(
                "Cannot turn --freeze_embedding on without --pretrained_embedding!"
            )
        return


def overload_embedding(model, emb_weight, freeze_embedding):
    from . import dist_util, logger
    import torch
    from torch.nn import Parameter
    orig_vocab_size, _ = emb_weight.shape
    assert model.word_embedding.weight.shape[0] == orig_vocab_size
    with torch.no_grad():
        model.word_embedding.weight = Parameter(emb_weight)
    if freeze_embedding:
        model.word_embedding.requires_grad_(False)
    logger.log("### Successfully overloaded pretrained embedding weight.")
    dist_util.barrier()
    return model


def fetch_pretrained_denoiser(args):  # Returns state dict
    from . import dist_util
    if args.pretrained_denoiser:
        denoiser_state_dict = dist_util.load_state_dict(args.pretrained_denoiser)
        return denoiser_state_dict
    return


def overload_denoiser(model, denoiser_state_dict):
    from . import dist_util, logger
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in denoiser_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logger.log("### Successfully overloaded pretrained denoiser dict.")
    dist_util.barrier()
    return model


def get_latest_model_path(base_path):
    try:
        import os
        candidates = filter(os.path.isdir, (os.path.join(base_path, x) for x in os.listdir(base_path)))
        candidates_sort = sorted(candidates, key=os.path.getmtime, reverse=True)
        if not candidates_sort:
            return
        ckpt_path = candidates_sort[0]
        candidates = filter(os.path.isfile, (os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path)))
        candidates = filter(lambda s: s.endswith('.pt'), candidates)
        candidates_sort = sorted(candidates, key=os.path.getmtime, reverse=True)
        if not candidates_sort:
            return
        return candidates_sort[0]
    except OSError:
        return


def create_model_and_diffusion(
        *,
        hidden_t_dim,
        hidden_dim,
        vocab_size,
        dropout,
        seq_len,  # FNet Kwarg
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
    from MuseDiffusion.models.diffusion \
        import SpacedDiffusion, space_timesteps, get_named_beta_schedule
    from MuseDiffusion.models.network import TransformerNetModel

    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        hidden_t_dim=hidden_t_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
        dropout=dropout,
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
