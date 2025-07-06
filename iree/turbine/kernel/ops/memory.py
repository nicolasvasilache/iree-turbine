from typing import (
    Any,
    List,
    Tuple,
    Optional,
    Iterator,
    overload,
    Callable,
    Tuple,
    Union,
)
import typing

if typing.TYPE_CHECKING:
    from sympy import Symbol
    from ..lang.types import Index, Vector

from .base import (
    define_op,
)

__all__ = ["kernel_buffer_load", "kernel_buffer_store", "kernel_buffer_transfer_gather"]


@define_op
def kernel_buffer_load(
    kernel_buffer,
    multi_index: Tuple["Index", ...],
    shape: Tuple[Union[int, "Symbol"], ...],
) -> "Vector": ...


@define_op
def kernel_buffer_store(
    kernel_buffer,
    multi_index: Tuple["Index", ...],
    item: "Vector",
) -> None: ...

@define_op
def kernel_buffer_transfer_gather(
    kernel_buffer,
    multi_index: Tuple[Union["Index", "Vector"], ...],
    shape: Tuple[Union[int, "Symbol"], ...],
) -> "Vector":
    ...
