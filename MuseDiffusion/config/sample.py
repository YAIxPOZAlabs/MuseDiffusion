from typing import Literal, final
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args
import os
from argparse import ArgumentParser as Ap, ArgumentDefaultsHelpFormatter as Df
from .base import S, Choice, Item as _, validator


KEY_MAP = tuple(
    i + j
    for j in ("major", "minor")
    for i in ("c", "c#", "db", "d", "d#", "eb", "e", "f", "f#", "gb", "g", "g#", "ab", "a", "a#", "bb", "b")
)

TIME_SIG_MAP = ("4/4", "3/4", "6/8", "12/8")

PITCH_RANGE_MAP = ("very_low", "low", "mid_low", "mid", "mid_high", "high", "very_high")

INST_MAP = (
    "accordion",
    "acoustic_bass",
    "acoustic_guitar",
    "acoustic_piano",
    "banjo",
    "bassoon",
    "bell",
    "brass_ensemble",
    "celesta",
    "choir",
    "clarinet",
    "drums_full",
    "drums_tops",
    "electric_bass",
    "electric_guitar_clean",
    "electric_guitar_distortion",
    "electric_piano",
    "fiddle",
    "flute",
    "glockenspiel",
    "harp",
    "harpsichord",
    "horn",
    "keyboard",
    "mandolin",
    "marimba",
    "nylon_guitar",
    "oboe",
    "organ",
    "oud",
    "pad_synth",
    "percussion",
    "recorder",
    "sitar",
    "string_cello",
    "string_double_bass",
    "string_ensemble",
    "string_viola",
    "string_violin",
    "synth_bass",
    "synth_bass_808",
    "synth_bass_wobble",
    "synth_bell",
    "synth_lead",
    "synth_pad",
    "synth_pluck",
    "synth_voice",
    "timpani",
    "trombone",
    "trumpet",
    "tuba",
    "ukulele",
    "vibraphone",
    "whistle",
    "xylophone",
    "zither",
    "orgel",
    "synth_brass",
    "sax",
    "bamboo_flute",
    "yanggeum",
    "vocal",
)

GENRE_MAP = ("newage", "cinematic")

TRACK_ROLE_MAP = ("main_melody", "sub_melody", "accompaniment", "bass", "pad", "riff")

RHYTHM_MAP = ("standard", "triplet")


class SamplingCommonSettings(S):
    __GENERATE__: ...
    model_path: str \
        = _('', 'path where model checkpoint exists')
    step: int \
        = _(100, 'ddim step, if not using ddim, should be same as diffusion step')
    out_dir: str \
        = _('./generation_outputs/', 'output directory to store generated midi')
    batch_size: int \
        = _(50, 'batch size to run decode')
    top_p: int \
        = _(1, 'range of the noise added to input, should be set between 0 and 1 (0=no restriction)')
    clamp_step: int \
        = _(0, 'in clamp_first mode, choose end clamp step, otherwise starting clamp step')
    sample_seed: int \
        = _(105, 'random seed for sampling')
    clip_denoised: bool \
        = _(True, 'to do clipping while denoising')
    model_config_json: str \
        = _('', 'path where training_args.json exists (default: automatically parsed from model_path)')

    @validator('model_path')  # NOQA
    @classmethod
    def validate_model_path(cls, value):
        if not value:  # Try to get latest model_path
            from MuseDiffusion.utils.initialization import get_latest_model_path
            value = get_latest_model_path("diffusion_models")
            if value is None:
                raise ValueError("You should specify --model_path: no trained model in ./diffusion_models")
        return value

    @validator('model_config_json')  # NOQA
    @classmethod
    def validate_model_config_json(cls, value, values):
        if not value:
            model_path = values.get('model_path')
            if not model_path:
                model_path = cls.validate_model_path(model_path)
            value = os.path.join(os.path.split(model_path)[0], "training_args.json")
        if not os.path.isfile(value):
            raise ValueError("--model_config_json={} not exists!".format(value))
        return value


class ModificationExtraSettingsMixin(S):
    split: Choice('train', 'valid', 'test') \
        = _('test', 'dataset type used (default: test)')
    use_corruption: bool \
        = _(None, "switch to use corruption (default: same as train config)")
    corr_available: str \
        = _(None, "available corruptions (default: same as train config)")
    corr_max: int \
        = _(None, "max number of corruptions (default: same as train config)")
    corr_p: float \
        = _(None, "probability to choice each corruption (default: same as train config)")
    corr_kwargs: str \
        = _(None, "default arguments for each corruption input (default: same as train config)")


class MidiMeta(S):

    bpm: int
    audio_key: Choice(*KEY_MAP)
    time_signature: Choice(*TIME_SIG_MAP)
    pitch_range: Choice(*PITCH_RANGE_MAP)
    num_measures: int
    inst: Choice(*INST_MAP)
    genre: Choice(*GENRE_MAP)
    min_velocity: int
    max_velocity: int
    track_role: Choice(*TRACK_ROLE_MAP)
    rhythm: Choice(*RHYTHM_MAP)
    chord_progression: str

    @validator('chord_progression')  # NOQA
    @classmethod
    def validate(cls, value):
        mapping = {',': '-', '[': '', ']': '', "'": '', ' ': ''}
        return ''.join(mapping.get(c, c) for c in value)

    @classmethod
    def to_argparse(cls, parser_or_group=None):
        if cls is not MidiMeta:
            return super(MidiMeta, cls).to_argparse(parser_or_group)
        parser_or_group = parser_or_group or Ap(formatter_class=Df)
        for name, field in cls.__fields__.items():
            kw = dict(dest=name, type=field.type_, default=field.default,
                      help=field.field_info.description, required=False)
            if getattr(field.type_, '__origin__', None) is Literal:
                choices = tuple(get_args(field.outer_type_))
                kw.update(type=str, choices=choices, metavar="{"+", ".join(map(str, choices))+"}")
            parser_or_group.add_argument("--" + name, **kw)
        return parser_or_group


@final
class ModificationSettings(SamplingCommonSettings, ModificationExtraSettingsMixin):
    __GENERATE__ = False


@final
class GenerationSettings(SamplingCommonSettings, MidiMeta):
    __GENERATE__ = True

    num_samples: int = _(1000, "number of midi samples to generate from metadata")

    @property
    def midi_meta(self) -> MidiMeta:
        return MidiMeta(**{k: getattr(self, k) for k in MidiMeta.__fields__})

    @classmethod
    def to_argparse(cls, parser_or_group=None):
        if parser_or_group is None:
            parser_or_group = Ap(formatter_class=Df)
        meta_group = parser_or_group.add_argument_group(title="meta")
        meta_group.add_mutually_exclusive_group().add_argument(
            "--meta_json", type=str, required=False,
            help="you can alter meta arguments all below by meta_json file.")
        MidiMeta.to_argparse(meta_group.add_mutually_exclusive_group().add_argument_group())
        num_samples = parser_or_group.add_argument_group(title="num_samples")
        num_samples.add_argument("--num_samples", type=int, default=cls.__fields__["num_samples"].default,
                                 help="number of midi samples to generate from metadata")
        setting_group = parser_or_group.add_argument_group(title="settings")
        SamplingCommonSettings.to_argparse(setting_group)
        return parser_or_group

    @classmethod
    def from_argparse(cls, namespace, __top=True):
        if not isinstance(namespace, dict):
            namespace = vars(namespace)
        num_samples = namespace.pop('num_samples')
        sample_commons = {field: namespace.pop(field) for field in SamplingCommonSettings.__fields__}
        if namespace.get('meta_json', None):
            midi_meta = MidiMeta.parse_file(namespace['meta_json'])
        else:
            if 'meta_json' in namespace:
                namespace.pop('meta_json')
            for k, v in namespace.items():
                if v is None:
                    namespace.pop(k)
            midi_meta = MidiMeta(**namespace)
        return cls(num_samples=num_samples, **sample_commons, **midi_meta.dict())


__all__ = ('ModificationSettings', 'GenerationSettings')
