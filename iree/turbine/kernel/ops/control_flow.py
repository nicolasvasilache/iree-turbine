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
    from ..lang.types import Index

from .base import (
    define_op,
)

__all__ = [
    "for_loop",
]


@define_op
def for_loop(
    start: Union[int, "Index", "Symbol"],
    stop: Optional[Union[int, "Index", "Symbol"]] = None,
    step: Optional[Union[int, "Index", "Symbol"]] = None,
    init_args: List[Any] = [],
) -> Callable[[Callable[[Union[int, "Index", "Symbol"], List[Any]], Optional[Tuple]]], List[Any]]:
    # TODO: The output type signature should also allow a single element return
    # instead of a List for better programming experience.
    ...
