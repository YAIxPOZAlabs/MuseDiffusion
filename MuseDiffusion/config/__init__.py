from .base import S, Choice, Item as _, Validator


class GeneralSettings(S):
    lr: float \
        = _(1e-4, "Learning Rate")
    batch_size: int \
        = _(2048, "Batch size of running step and optimizing")
    microbatch: int \
        = _(64, "Batch size for forward and backward")
    learning_steps: int \
        = _(320000, "Steps for whole iteration")
    log_interval: int \
        = _(20, "Steps per log")
    save_interval: int \
        = _(2000, "Steps per save")
    eval_interval: int \
        = _(1000, "Steps per eval")
    ema_rate: str \
        = "0.5,0.9,0.99"
    seed: int \
        = _(102, "Seed for train or test.")
    resume_checkpoint: str \
        = _("", "Checkpoint path(.pt) to resume training")
    checkpoint_path: str \
        = _("", "! This will be automatically updated while training !")


class DiffusionSettings(S):
    diffusion_steps: int \
        = _(2000, "The number of diffusion steps")
    schedule_sampler: Choice('uniform', 'lossaware', 'fixstep') \
        = _("lossaware", "Type of Schedule Sampler for Diffusion")
    noise_schedule: Choice('linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin') \
        = _("sqrt", "Type of Beta Schedule for Diffusion")
    timestep_respacing: str \
        = _("", "A string containing comma-separated numbers, "
                "indicating the step count per section. "
                "As a special case, use \"ddimN\" where N is a number of steps "
                "to use the striding from the DDIM paper.")


class DataModelCommonSettings(S):
    seq_len: int \
        = _(256, "Sequence length to be used in model and data filtering. max is 2096.\n")
    vocab_size: int \
        = _(729, "Vocab size for embeddings. Fixed to 729")
    pretrained_denoiser: str \
        = _("", "To use pretrained denoiser, provide full file path of .pt file.")
    pretrained_embedding: str \
        = _("", "To use pretrained embedding, provide full file path of .pt file.")
    freeze_embedding: bool \
        = _(False, "Whether to disable embedding weight's gradient. you MUST use this with pretrained_embedding.")
    use_bucketing: bool \
        = _(True, "Whether to enable bucketing in data loader.")


class DataSettings(S):
    dataset: str \
        = _("ComMU", "Name of dataset.")
    data_dir: str \
        = _("datasets/ComMU-processed", "Path for dataset to be saved.")
    data_loader_workers: int \
        = _(2, "num_workers for DataLoader.")


class CorruptionSettings(S):
    use_corruption: bool \
        = _(False, "Switch to use corruption.")
    corr_available: str \
        = _("mt,mn,rn,rr", "Available corruptions: see data.corruptions module.")  # TODO: add 'at'
    corr_max: int \
        = _(0, "Max number of corruptions.")
    corr_p: float \
        = _(0.5, "Probability to choice each corruption.")


class ModelSettings(S):
    hidden_t_dim: int \
        = _(128, "hidden_t_dim for Transformer backbone.")
    hidden_dim: int \
        = _(128, "hidden_dim for Embedding and Transformer backbone.")
    dropout: float \
        = _(0.1, "Dropout rate.")
    # fnet_hidden_dim: int \
    #     = _(128)            # FNet
    # fnet_intermediate_dim: int \
    #     = _(512)            # FNet
    # num_fnet_layers: int \
    #     = _(6)              # Added for FNet
    # use_attention: bool \
    #     = _(False)


class OtherSettings(S):
    use_fp16 = False
    fp16_scale_growth = 0.001
    gradient_clipping = -1.0
    weight_decay = 0.0
    learn_sigma = False
    use_kl = False
    predict_xstart = True
    rescale_timesteps = True
    rescale_learned_sigmas = False
    sigma_small = False
    emb_scale_factor = 1.0


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
    use_ddim: bool \
        = _(True, 'DDIM Sampling')

    @Validator('model_path')
    def validate(cls, model_path):  # NOQA
        if not model_path:  # Try to get latest model_path
            from MuseDiffusion.utils.initialization import get_latest_model_path
            model_path = get_latest_model_path("diffusion_models")
            if model_path is None:
                raise ValueError("You should specify --model_path: no trained model in ./diffusion_models")
        return model_path


class TrainSettings(
        OtherSettings,
        ModelSettings,
        CorruptionSettings,
        DataSettings,
        DataModelCommonSettings,
        DiffusionSettings,
        GeneralSettings
):
    pass


__all__ = ('TrainSettings', 'SamplingSettings')
