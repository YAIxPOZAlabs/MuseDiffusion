from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Optional, OrderedDict, Union, Tuple
    from os import PathLike
    from torch.nn import Parameter
    from MuseDiffusion.config import TrainSettings
    from MuseDiffusion.models.network import TransformerNetModel
    from MuseDiffusion.models.diffusion import GaussianDiffusion


def seed_all(seed: "Any", deterministic: "bool" = False) -> "None":
    import random
    import numpy as np
    import torch
    from ..data.corruption import generator
    from .dist_util import get_rank
    if deterministic:
        seed = hash(seed)
        torch.backends.cudnn.deterministic = True  # NOQA
        torch.backends.cudnn.benchmark = False  # NOQA
    else:
        seed = hash(seed) + get_rank()  # Make seed differ by node rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # contains torch.cuda.manual_seed_all
    generator.seed(seed)


def fetch_pretrained_embedding(args: "TrainSettings") -> "Optional[Parameter]":  # Returns single parameter
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


def overload_embedding(
        model: "TransformerNetModel", emb_weight: "Parameter", freeze_embedding: "bool"
) -> "TransformerNetModel":
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


def fetch_pretrained_denoiser(args: "TrainSettings") -> "Optional[OrderedDict]":  # Returns state dict
    from . import dist_util
    if args.pretrained_denoiser:
        denoiser_state_dict = dist_util.load_state_dict(args.pretrained_denoiser)
        return denoiser_state_dict
    return


def overload_denoiser(model: "TransformerNetModel", denoiser_state_dict: "OrderedDict") -> "TransformerNetModel":
    from . import dist_util, logger
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in denoiser_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logger.log("### Successfully overloaded pretrained denoiser dict.")
    dist_util.barrier()
    return model


def get_latest_model_path(base_path: "Union[str, PathLike]") -> "Optional[str]":
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


def create_model_and_diffusion(args: "TrainSettings") -> "Tuple[TransformerNetModel, GaussianDiffusion]":

    from MuseDiffusion.models.diffusion \
        import SpacedDiffusion, space_timesteps, get_named_beta_schedule
    from MuseDiffusion.models.network import TransformerNetModel

    model = TransformerNetModel(
        input_dims=args.hidden_dim,
        output_dims=args.hidden_dim,
        hidden_t_dim=args.hidden_t_dim,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )

    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    timestep_respacing = args.timestep_respacing
    if not timestep_respacing:
        timestep_respacing = [args.diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=args.rescale_timesteps,
        predict_xstart=args.predict_xstart,
    )

    return model, diffusion
