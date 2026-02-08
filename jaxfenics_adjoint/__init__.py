import sys
import types
import typing


def _ensure_pep604_generic_alias():
    if sys.version_info >= (3, 10):
        return
    if not hasattr(types.GenericAlias, "__or__"):
        types.GenericAlias.__or__ = lambda self, other: typing.Union[self, other]
    if not hasattr(types.GenericAlias, "__ror__"):
        types.GenericAlias.__ror__ = lambda self, other: typing.Union[other, self]


_ensure_pep604_generic_alias()

from .core import build_jax_fem_eval, build_jax_fem_eval_fwd
from .helpers import from_jax
from fecr import from_numpy, to_numpy
