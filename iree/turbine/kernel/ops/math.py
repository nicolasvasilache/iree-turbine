from typing import Tuple, Union
import typing

if typing.TYPE_CHECKING:
    from sympy import Symbol
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "exp",
    "exp2",
    "sqrt",
    "vector_constant",
]


@define_op
def exp(val): ...


@define_op
def exp2(val): ...


@define_op
def sqrt(val): ...


@define_op
def vector_constant(shape: Tuple[Union[int, "Symbol"], ...], dtype, value: int | float) -> "Vector": ...
