from random import Random as _Random
import reprlib as _reprlib
import torch as _torch

generator = _Random()  # To available seeding in seed_all function
del _Random


class Corruptions:  # config key: corr_available, corr_max, corr_p

    __slots__ = ('corr_available', 'corr_max', 'corr_p', 'corr_kwargs')

    @classmethod
    def from_config(
            cls,
            corr_available: "str",
            corr_max: "int|str",
            corr_p: "float|str",
            corr_kwargs: "str|None" = None
    ):
        return cls(
            corr_available=tuple(corr_available.split(',')),
            corr_max=int(corr_max),
            corr_p=float(corr_p),
            corr_kwargs=eval(corr_kwargs) if corr_kwargs else None
        )

    def __new__(
            cls,
            corr_available: "tuple|list",
            corr_max: "int",
            corr_p: "float",
            corr_kwargs: "str|dict[str, ...]" = None
    ):
        assert all(key in cls.MAP or callable(key) for key in corr_available)
        assert 0 <= corr_max <= len(corr_available) and 0 <= corr_p <= 1
        assert corr_kwargs is None or isinstance(corr_kwargs, dict)
        corr_available = tuple(cls.get(key, corr_kwargs) for key in corr_available)
        self = super(Corruptions, cls).__new__(cls)
        self.corr_available = corr_available
        self.corr_max = corr_max
        self.corr_p = corr_p
        self.corr_kwargs = corr_kwargs
        return self

    def __call__(self, seq: _torch.Tensor, inplace: bool = False):
        if _torch.is_grad_enabled():
            with _torch.no_grad():
                return self(seq, inplace=inplace)
        assert seq.ndim == 1
        corrupted = seq if inplace else seq.clone()  # To available in-place calculation
        corr_available = list(self.corr_available)
        generator.shuffle(corr_available)
        for corruption_fn in corr_available[:self.corr_max]:
            if generator.random() > self.corr_p:
                corrupted = corruption_fn(corrupted, inplace=True)
        return corrupted

    @_reprlib.recursive_repr()
    def __repr__(self):
        corr_names = ','.join(
            repr(fn) if isinstance(fn, Corruptions) else
            fn.func.__name__ if hasattr(fn, 'func') else
            fn.__name__
            for fn in self.corr_available
        )
        return f"{self.__class__.__qualname__}(" \
               f"corr_available=[{corr_names}], " \
               f"corr_max={self.corr_max!r}, " \
               f"corr_p={self.corr_p!r}, " \
               f"corr_kwargs={self.corr_kwargs!r})"

    @classmethod
    def get(cls, key, update_kwargs=None, inplace=None):
        if callable(key):
            return key
        func, required_kwargs, default_kwargs = cls.MAP[key]
        default_kwargs = default_kwargs.copy()
        if update_kwargs is not None:
            default_kwargs.update(update_kwargs)
        kwargs = {k: update_kwargs[k] for k in required_kwargs}
        if inplace is not None:
            kwargs.update(inplace=inplace)
        if kwargs:
            from functools import partial
            func = partial(func, **kwargs)
        return func

    @classmethod
    def register(cls, key, required_kwargs=None, **default_kwargs):
        def decorator(func):
            cls.MAP[key] = (func, required_kwargs, default_kwargs)
            return func
        assert key not in cls.MAP
        return decorator

    @classmethod
    def finalize(cls):
        from types import MappingProxyType
        cls.MAP = MappingProxyType(cls.MAP)  # Immutable type

    MAP = {}


@Corruptions.register('mt', required_kwargs=['p'], p=0.5)
def masking_token(seq: _torch.Tensor, p: float, inplace: bool = False):
    """
    masking_token으로 변경하는 함수
    token 단위로 진행
    이거 쓰려면 masking token값으로 730 지정해주고 embedding layer의 dim 1 추가해야됨
    """
    corrupted = seq if inplace else seq.clone()

    for i in range(len(seq[12:])):
        if seq[i+12] == 1:
            break
        rnd = generator.random()
        if rnd < p:
            corrupted[i+12] = 729
    return corrupted


@Corruptions.register('mn', required_kwargs=['p'], p=0.5)
def masking_note(seq: _torch.Tensor, p: float, inplace: bool = False):
    """
    masking_token으로 변경하는 함수
    note (position - velocity - pitch - duration) 단위로 진행
    이거 쓰려면 masking token값으로 730 지정해주고 embedding layer의 dim 1 추가해야됨
    """
    corrupted = seq if inplace else seq.clone()

    vel_idx, = _torch.nonzero(_torch.logical_and(131 <= seq, seq <= 194), as_tuple=True)

    for idx in vel_idx:
        if idx + 3 > len(seq):
            continue
        if generator.random() < p:
            corrupted[idx-1:idx+3] = 729
    return corrupted


@Corruptions.register('rn', required_kwargs=['p'], p=0.5)
def randomize_note(seq: _torch.Tensor, p: float, inplace: bool = False):
    """
    주어진 확률(혹은 비율) p 만큼의 note의 pitch, duration, velocity, position? 을 랜덤하게 변경하는 함수
    seq: meta+0+note_seq+zero_padding (torch type)
    p: 0.0~1.0 사이값
    note token 순서: position - velocity - pitch - duration 순으로 존재
    pitch token range: 3 ~ 130
    velocity: 131 ~ 194
    duration: 304~431
    position: 432~559
    """
    corrupted = seq if inplace else seq.clone()

    # velocity token index 찾기 (velocity token 범위: 131~194)
    vel_idx, = _torch.nonzero(_torch.logical_and(131 <= seq, seq <= 194), as_tuple=True)

    for idx in vel_idx:
        if idx + 3 > len(seq):
            continue
        if generator.random() < p:
            corrupted[idx] = generator.randint(131, 194)  # new_velocity
            corrupted[idx+1] = generator.randint(3, 130)  # new_pitch
            corrupted[idx+2] = generator.randint(304, 431)  # new_duration
            # seq[idx-1] = new_position --> 이건 쓸까말까
    return corrupted


@Corruptions.register('at', required_kwargs=['p'], p=0.5)
def adding_token(seq: _torch.Tensor, mask: _torch.Tensor, p: float, inplace: bool = False):
    """
    단일토큰 추가하는 함수...인데 추가해버리면 input_mask에도 약간의 변화가 필요함
    """
    corrupted = seq if inplace else seq.clone()
    ...  # TODO
    raise NotImplementedError


@Corruptions.register('rr', required_kwargs=['count'], count=3)
def random_rotating(seq: _torch.Tensor, count: int, inplace: bool = False):
    """
    마디단위로 묶어서 마디의 순서를 랜덤하게 섞는다
    count: 몇번 마디 바꾸는것을 수행할것인가
    """
    rotated = seq if inplace else seq.clone()
    bar_idx, = _torch.nonzero(seq == 2, as_tuple=True)
    eos_idx, = _torch.nonzero(seq == 1, as_tuple=True)
    eos_idx = eos_idx[-1]

    for _ in range(count):
        assert len(bar_idx) > 1
        first_idx, second_idx = sorted(generator.sample(range(0, len(bar_idx)), 2))

        # find start and end of bars
        bar1_start = bar_idx[first_idx]
        bar2_start = bar_idx[second_idx]

        bar1_end = bar_idx[first_idx+1]
        bar2_end = bar_idx[second_idx+1] if second_idx < len(bar_idx)-1 else eos_idx

        # sequence에서 bar 2개를 잘라내고 순서를 바꿔서 concat
        start_to_bar1 = rotated[:bar1_start]
        bar1_array = rotated[bar1_start:bar1_end]
        bar1_to_bar2 = rotated[bar1_end:bar2_start]
        bar2_array = rotated[bar2_start:bar2_end]
        bar2_to_eos = rotated[bar2_end:]

        rotated = _torch.cat([start_to_bar1, bar2_array, bar1_to_bar2, bar1_array, bar2_to_eos])

    return rotated


Corruptions.finalize()
__all__ = tuple(v for v in vars() if not v.startswith('_'))
