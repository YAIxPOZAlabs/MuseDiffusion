from .base import S, Choice, Item as _, Validator


class SamplingSettings(S):
    model_path: str \
        = _('', 'folder where model checkpoint exist')
    step: int \
        = _(100, 'ddim step, if not using ddim, should be same as diffusion step')
    out_dir: str \
        = _('./generation_outputs/', 'output directory to store generated midi')
    batch_size: int \
        = _(50, 'batch size to run decode')
    top_p: int \
        = _(1, 'range of the noise added to input, should be set between 0 and 1 (0=no restriction)')
    split: Choice('train', 'valid', 'test') \
        = _('valid', 'dataset type used in sampling')
    clamp_step: int \
        = _(0, 'in clamp_first mode, choose end clamp step, otherwise starting clamp step')
    sample_seed: int \
        = _(105, 'random seed for sampling')
    clip_denoised: bool \
        = _(True, 'to do clipping while denoising')

    @Validator('model_path')
    def validate(cls, model_path):  # NOQA
        if not model_path:  # Try to get latest model_path
            from MuseDiffusion.utils.initialization import get_latest_model_path
            model_path = get_latest_model_path("diffusion_models")
            if model_path is None:
                raise ValueError("You should specify --model_path: no trained model in ./diffusion_models")
        return model_path
