from typing import Literal
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args
from argparse import ArgumentParser as Ap, ArgumentDefaultsHelpFormatter as Df
from pydantic import BaseModel, Field
from pydantic.validators import bool_validator
import yaml


class ArgparseCompatibleBaseModel(BaseModel):

    class Config:  # Override this to allow extra kwargs
        extra = "forbid"

    @classmethod
    def from_argparse(cls, namespace, __top=True):
        if not isinstance(namespace, dict):
            namespace = vars(namespace)
        kwargs = {}
        for name, field in cls.__fields__.items():
            if isinstance(field.type_, type) and issubclass(field.type_, BaseModel):
                kwargs[name] = ArgparseCompatibleBaseModel.from_argparse.__func__(field.type_, namespace, __top=False)  # NOQA
            else:
                kwargs[name] = namespace.pop(name)
        assert not (__top and namespace), str(namespace)
        return cls(**kwargs)

    @classmethod
    def to_argparse(cls, parser_or_group=None):
        if parser_or_group is None:
            parser_or_group = Ap(formatter_class=Df)
        for name, field in cls.__fields__.items():
            if isinstance(field.type_, type) and issubclass(field.type_, BaseModel):
                group = parser_or_group.add_argument_group(name)
                ArgparseCompatibleBaseModel.to_argparse.__func__(field.type_, group)  # NOQA
                continue
            kw = dict(dest=name, type=field.type_, default=field.default,
                      help=field.field_info.description, required=field.required)
            if getattr(field.type_, '__origin__', None) is Literal:
                choices = tuple(get_args(field.outer_type_))
                s = "def {name}(arg):\n    for ch in __CHOICES__:\n" \
                    "        if str(ch) == arg:\n            return ch\n    raise ValueError" \
                    .format(name=name)
                n = {"__CHOICES__": choices, "__name__": name}
                exec(s, n)
                kw.update(type=n[name], choices=choices, metavar="{"+", ".join(map(str, choices))+"}")
            elif isinstance(field.type_, type) and issubclass(field.type_, bool):
                kw.update(type=bool_validator, metavar="{true, false}")
            parser_or_group.add_argument("--" + name, **kw)
        return parser_or_group

    @classmethod
    def from_argv(cls, argv=None):
        return cls.from_argparse(cls.to_argparse().parse_args(argv))


S = Setting = ArgparseCompatibleBaseModel  # Alias


def choice(*args):
    return Literal.__getitem__(args)


C = Choice = choice  # Alias


def item(default, description=None):
    return Field(default, description=description)


_ = Item = item  # Alias


if __name__ == '__main__':

    class Config1(S):
        a: int = _(1, description='this is a')
        b: int = _(2, description='this is b')

    class Config2(S):
        c: C('choice1', 'choice2') = _('choice2', description='this is c')
        d: bool = _(True, description='this is d')

    class Config(S):
        conf1: Config1 = Config1()
        conf2: Config2 = Config2()

    Config.to_argparse().print_help()
    yaml.dump(Config.from_argv().dict(), __import__('sys').stdout)
