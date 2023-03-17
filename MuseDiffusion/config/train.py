from typing import final
from argparse import ArgumentParser as Ap, ArgumentDefaultsHelpFormatter as Df
from .base import S, Choice, Item as _


class GeneralSettings(S):
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
        = _("0.5,0.9,0.99", "Rate(s) of Exponential Moving Average Method")
    lr: float \
        = _(1e-4, "Learning Rate for Optimizer AdamW")
    weight_decay: float \
        = _(0., "Weight Decay for Optimizer AdamW")
    gradient_clipping: float \
        = _(-1., "How to do clip_grad_norm_")
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
    predict_xstart: bool \
        = _(True, "the model outputs to predict x_0, else to predict eps.")
    rescale_timesteps: bool \
        = _(True, "if True, pass floating point timesteps into the model "
                  "so that they are always scaled like in theoriginal paper (0 to 1000)")
    timestep_respacing: str \
        = _("", "A string containing comma-separated numbers, indicating the step count per section. As a special case,"
                " use \"ddimN\" where N is a number of steps to use the striding from the DDIM paper.")


class ModelSettings(S):
    seq_len: int \
        = _(2096, "Sequence length to be used in model and data filtering. max is 2096.\n")
    vocab_size: int \
        = _(729, "Vocab size for embeddings. Fixed to 729")
    hidden_t_dim: int \
        = _(128, "hidden_t_dim for Transformer backbone.")
    hidden_dim: int \
        = _(128, "hidden_dim for Embedding and Transformer backbone.")
    dropout: float \
        = _(0.1, "Dropout rate.")
    pretrained_denoiser: str \
        = _("", "To use pretrained denoiser, provide full file path of .pt file.")
    pretrained_embedding: str \
        = _("", "To use pretrained embedding, provide full file path of .pt file.")
    freeze_embedding: bool \
        = _(False, "Whether to disable embedding weight's gradient. you MUST use this with pretrained_embedding.")


class DataSettings(S):
    dataset: str \
        = _("ComMU", "Name of dataset.")
    data_dir: str \
        = _("datasets/ComMU-processed", "Path for dataset to be saved.")
    data_loader_workers: int \
        = _(2, "num_workers for DataLoader.")
    use_bucketing: bool \
        = _(True, "Whether to enable bucketing in data loader.")


class CorruptionSettings(S):
    use_corruption: bool \
        = _(True, "Switch to use corruption.")
    corr_available: str \
        = _("mt,mn,rn,rr", "Available corruptions: see data.corruptions module.")
    corr_max: int \
        = _(4, "Max number of corruptions.")
    corr_p: float \
        = _(0.5, "Probability to choice each corruption.")
    corr_kwargs: str \
        = _("dict(p=0.4)", "Default arguments for each corruption input.")


@final
class TrainSettings(
        CorruptionSettings,
        DataSettings,
        ModelSettings,
        DiffusionSettings,
        GeneralSettings
):

    @classmethod
    def to_argparse(cls, parser=None, add_json=False):
        if not add_json:
            return super(TrainSettings, cls).to_argparse(parser)
        if parser is None:
            parser = Ap(formatter_class=Df)
        setting_group = parser.add_argument_group(title="settings")
        setting_group.add_argument(
            "--config_json", type=str, required=False,
            help="You can alter arguments all below by config_json file.")
        super(TrainSettings, cls).to_argparse(setting_group)
        return parser

    @classmethod
    def from_argparse(cls, namespace, __top=True):
        if getattr(namespace, "config_json", None):
            return cls.parse_file(namespace.config_json)
        else:
            if hasattr(namespace, "config_json"):
                delattr(namespace, "config_json")
            return super(TrainSettings, cls).from_argparse(namespace)


__all__ = ('TrainSettings', )
