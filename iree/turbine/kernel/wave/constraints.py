# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence
from sympy import ceiling, Piecewise, floor

from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from .._support.dtype import DataType
from ..lang.global_symbols import *
from ..ops.wave_ops import CustomOp

"""
Formatting for different target intrinsics:
    <kind>_<elem-type-C>_<M>x<N>x<K>_<elem-type-A>[_<elem-type-B>]

Values: 0xABCD where:
* A = vendor:
  * 1 = AMD
  * 2 = NVIDIA
* B = architecture. When an intrinsic exists in multiple architectures, this
      should be the architecture it was introduced in, as long as it still
      has the same semantics. If a new architecture breaks an existing
      intrinsic's semantics, we can use that field for versioning.
  * For AMD:
    * 0 = CDNA1
    * 1 = CDNA2
    * 2 = CDNA3
    * 8 = RDNA3
* C = element type of A-matrix:
  * 0 = 64-bit float (e.g. IEEE754 double precision)
  * 1 = 32-bit float (e.g. IEEE754 single precision, and "xf32" fast variants)
  * 2 = 16-bit float (incl. IREE754 half and bf16)
  * 3 = 8-bit float (incl. f8E5M2, f8E4M3, and "FNUZ" variants)
  * C = 8-bit integer (any signedness)
* D enumerates intrinsics that share the same 0xABC* bits.
"""


class MMAType(Enum):
    # Intrinsics introduced in CDNA1
    F32_16x16x16_F16 = 0x1020
    F32_32x32x8_F16 = 0x1021
    F32_16x16x32_K8_F16 = 0x1022
    F32_32x32x16_K8_F16 = 0x1023
    I32_16x16x16_I8 = 0x10C0
    I32_32x32x8_I8 = 0x10C1

    # Intrinsics introduced in CDNA3
    F32_16x16x32_F8 = 0x1230
    F32_32x32x16_F8 = 0x1231
    F32_16x16x32_K4_F8 = 0x1232
    F32_32x32x16_K4_F8 = 0x1233
    I32_16x16x32_I8 = 0x12C0
    I32_32x32x16_I8 = 0x12C1


class MMAOperand(Enum):
    M = 0
    N = 1
    K = 2


@dataclass
class Constraint(ABC):
    """
    Base class for constraints. Every constraint reduces to
    the following form:
        Variables: [x0, x1, ...., xN]
        Bounds: [lb0 <= x0 <= ub0, ..., lbN <= xN <= ubN]
        Equality Constraints: [f0(x0, ..., xN) = 0, f1(x0, ..., xN) = 0, ...]
        Inequality Constraints: [g0(x0, ..., xN) <= 0, g1(x0, ..., xN) <= 0, ...]
    """

    @abstractmethod
    def apply(self) -> IndexSequence:
        """Apply the constraint and get the resulting index sequence."""
        ...


@dataclass
class HardwareConstraint(Constraint):
    """
    A constraint of the form
        tkw.HardwareConstraint(threads_per_wave = N,
                               mma_type = 'MFMA_F32_16x16x16_F16')
    specifies that the hardware supports N threads per wave and that
    we want all mma operations in the microkernel to be
    mapped to a hardware mma instruction of shape (16x16x16).
    This translates to a hardware specific index constraint.

    Not all computation graphs have mma operators in them. In
    these situations, the user can specify the vector shape they
    want to tile to by specifying the vector shapes dictionary
    which maps a tensor dimension to its corresponding tile size.

    Both mma constraints and vector shapes can be specified, but
    the mapping from symbols to shapes should be injective.
    """

    threads_per_wave: int
    threads_per_block: tuple[int, int, int] = (64, 1, 1)
    mma_type: Optional[MMAType] = MMAType.F32_16x16x16_F16
    vector_shapes: Optional[dict[IndexSymbol, int]] = None
    max_bits_per_load: int = 128

    def max_elems_per_load(self, element_type: DataType) -> int:
        return self.max_bits_per_load // element_type.bitwidth()

    def get_thread_id_from_workgroup_dim(self, workgroup_dim: int) -> IndexSymbol:
        match workgroup_dim:
            case 0:
                return THREAD_0
            case 1:
                return THREAD_1
            case 2:
                return THREAD_2
            case _:
                raise ValueError("Invalid workgroup dimension. Expected 0, 1 or 2.")

    def mma_matrix_shapes(self, mma_type: Optional[MMAType]) -> tuple[int]:
        # TODO: Eventually the shapes and indices should be provided by a tool
        if mma_type == None:
            mma_type = self.mma_type
        match mma_type:
            case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
                return (16, 16, 16)
            case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
                return (32, 32, 8)
            case (
                MMAType.F32_16x16x32_F8
                | MMAType.F32_16x16x32_K8_F16
                | MMAType.F32_16x16x32_K4_F8
                | MMAType.I32_16x16x32_I8
            ):
                return (16, 16, 32)
            case (
                MMAType.F32_32x32x16_F8
                | MMAType.F32_32x32x16_K8_F16
                | MMAType.F32_32x32x16_K4_F8
                | MMAType.I32_32x32x16_I8
            ):
                return (32, 32, 16)
            case _:
                return ()

    def mma_index_offset(self, mma_type: Optional[MMAType]):
        lane = self.linearized_thread_id % self.threads_per_wave
        if mma_type == None:
            mma_type = self.mma_type
        match mma_type:
            # (M x K, N x K) -> M x N
            case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
                offset = [
                    Piecewise(
                        (lane % 16, ~MMA_ACC),
                        (4 * floor(lane / 16), MMA_ACC),
                    ),  # M
                    lane % 16,  # N
                    4 * floor(lane / 16),  # K
                ]
            case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
                offset = [
                    Piecewise(
                        (lane % 32, ~MMA_ACC),
                        (
                            (8 * floor(GPR_NUM / 4) % 32)
                            + 4 * floor(lane / 32)
                            + (GPR_NUM % 4),
                            MMA_ACC,
                        ),
                    ),  # M
                    lane % 32,  # N
                    4 * floor(lane / 32),  # K
                ]
            case (
                MMAType.F32_16x16x32_F8
                | MMAType.F32_16x16x32_K8_F16
                | MMAType.F32_16x16x32_K4_F8
                | MMAType.I32_16x16x32_I8
            ):
                offset = [
                    Piecewise(
                        (lane % 16, ~MMA_ACC), (4 * floor(lane / 16), MMA_ACC)
                    ),  # M
                    lane % 16,  # N
                    8 * floor(lane / 16),  # K
                ]
                if mma_type == MMAType.F32_16x16x32_K4_F8:
                    offset = [
                        Piecewise(
                            (lane % 16, ~MMA_ACC), (4 * floor(lane / 16), MMA_ACC)
                        ),  # M
                        lane % 16,  # N
                        (16 * floor(GPR_NUM / 4))
                        + 4 * floor(lane / 16)
                        + (GPR_NUM % 4),  # K
                    ]
            case (
                MMAType.F32_32x32x16_F8
                | MMAType.F32_32x32x16_K8_F16
                | MMAType.F32_32x32x16_K4_F8
                | MMAType.I32_32x32x16_I8
            ):
                offset = [
                    Piecewise(
                        (lane % 32, ~MMA_ACC),
                        (
                            (8 * floor(GPR_NUM / 4) % 32)
                            + 4 * floor(lane / 32)
                            + (GPR_NUM % 4),
                            MMA_ACC,
                        ),
                    ),  # M
                    lane % 32,  # N
                    8 * floor(lane / 32),  # K
                ]
                if mma_type == MMAType.F32_32x32x16_K4_F8:
                    offset = [
                        Piecewise(
                            (lane % 32, ~MMA_ACC),
                            (
                                (8 * floor(GPR_NUM / 4) % 32)
                                + 4 * floor(lane / 32)
                                + (GPR_NUM % 4),
                                MMA_ACC,
                            ),
                        ),  # M
                        lane % 32,  # N
                        (8 * floor(GPR_NUM / 4))
                        + 4 * floor(lane / 32)
                        + (GPR_NUM % 4),  # K
                    ]
            case _:
                raise ValueError("Unsupported MMA type")
        return offset

    @property
    def waves_per_block(self) -> tuple[int]:
        assert (
            self.threads_per_block[0] % self.threads_per_wave == 0
        ), "threads_per_block[0] must be a multiple of threads_per_wave"
        return (
            self.threads_per_block[0] // self.threads_per_wave,
        ) + self.threads_per_block[1:]

    @property
    def linearized_thread_id(self) -> IndexExpr:
        thread_ids = [THREAD_0, THREAD_1, THREAD_2]
        threads_per_block = [
            1,
            self.threads_per_block[0],
            self.threads_per_block[0] * self.threads_per_block[1],
        ]
        return sum([x * y for x, y in zip(thread_ids, threads_per_block)])

    # Inline substitution for vector_size given index map. In the future we can add support for other members.
    def subs_vector_shapes(self, index_map: dict[IndexSymbol, int]):
        if self.vector_shapes is None:
            return
        for vector_dim, vector_size in self.vector_shapes.items():
            if isinstance(vector_size, IndexExpr):
                self.vector_shapes[vector_dim] = vector_size.subs(index_map)

    def apply(self):
        assert False, "Call either apply_read_write_thread_mapping or apply_mma_mapping"

    def apply_read_write_thread_mapping(
        self,
        dim: IndexSymbol,
        workgroup_dim: int,
        elements_per_thread: int | IndexSymbol,
        stride: int,
    ) -> IndexSequence:
        thread_id = self.get_thread_id_from_workgroup_dim(workgroup_dim)
        return IndexSequence(
            thread_id * elements_per_thread, elements_per_thread, stride
        )

    def apply_mma_mapping(
        self,
        dim: IndexSymbol,
        constraint_index: int | MMAOperand,
        mma_type: MMAType,
    ) -> IndexSequence:
        lane = self.linearized_thread_id % self.threads_per_wave
        if mma_type == None:
            mma_type = self.mma_type
        offset = self.mma_index_offset(mma_type)
        match mma_type:
            # (M x K, N x K) -> M x N
            case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
                size = [
                    Piecewise((1, ~MMA_ACC), (4, MMA_ACC)),  # M
                    1,  # N
                    4,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
                size = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    4,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (32, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case (
                MMAType.F32_16x16x32_F8
                | MMAType.F32_16x16x32_K8_F16
                | MMAType.F32_16x16x32_K4_F8
                | MMAType.I32_16x16x32_I8
            ):
                size = [
                    Piecewise((1, ~MMA_ACC), (4, MMA_ACC)),  # M
                    1,  # N
                    8,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case (
                MMAType.F32_32x32x16_F8
                | MMAType.F32_32x32x16_K8_F16
                | MMAType.F32_32x32x16_K4_F8
                | MMAType.I32_32x32x16_I8
            ):
                size = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    8,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (32, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case _:
                raise ValueError("Unsupported MMA type")

        assert isinstance(
            constraint_index, MMAOperand
        ), f"Invalid MMA operand {constraint_index}"
        return IndexSequence(
            offset[constraint_index.value],
            size[constraint_index.value],
            stride[constraint_index.value],
        )


@dataclass
class WorkgroupConstraint(Constraint):
    """
    A constraint of the form `tkw.WorkgroupConstraint(M, BLOCK_M, 0)`
    specifies that we want to distribute dimension M along workgroup dim 0
    with a tile size of BLOCK_M resulting in M // BLOCK_M workgroups along that
    dimension. This translates to an index constraint for all tensors of the
    shape [M, ?] -> index += (workgroup_id_0 * BLOCK_M, 0)
    """

    dim: IndexExpr
    tile_size: IndexExpr
    workgroup_dim: int
    apply_fn: Optional[Callable] = None
    primary: Optional[bool] = True
    iters: Optional[IndexExpr | int] = None

    def __post_init__(self):
        self.wg_dim = None
        match self.workgroup_dim:
            case 0 | 1 | 2 | 3 | 4:
                self.wg_dim = get_workgroup_symbol(self.workgroup_dim)
            case _:
                raise ValueError(
                    "Invalid workgroup dimension. Expected 0, 1, 2, 3 or 4."
                )

    @property
    def count(self) -> IndexExpr:
        """
        Returns an expression for the total number of workgroups for the specific workgroup_dim.
        """
        if self.iters:
            return self.iters
        return ceiling(self.dim / self.tile_size)

    def apply(self) -> IndexSequence:
        if self.apply_fn:
            return IndexSequence(self.apply_fn(self.wg_dim), 1)
        return IndexSequence(self.wg_dim * self.tile_size, 1)


def get_grid_shape(wg_constraints: list[WorkgroupConstraint]) -> list[IndexExpr]:
    sorted_constraints = sorted(
        [x for x in wg_constraints if x.primary], key=lambda x: x.workgroup_dim
    )
    # Currently not more than one primary constraint in each dimension supported.
    # This is captured in verify_global_constraints.
    grid: list[IndexExpr] = [constraint.count for constraint in sorted_constraints]
    return grid


@dataclass
class TilingConstraint(Constraint):
    """
    A constraint of the form `tkw.TilingConstraint(K, BLOCK_K)` specifies
    that we want to tile the K dimension with a tile size of BLOCK_K. This
    adds an index constraint to the K-th dimension of a tensor of the form
    BLOCK_K * i, where i is the induction variable associated with the
    loop around dimension K.
    """

    dim: IndexExpr
    tile_size: IndexExpr
    induction_var: Optional[IndexExpr] = None
    iters: Optional[IndexExpr] = None

    def __eq__(self, value):
        if not isinstance(value, TilingConstraint):
            return False
        return (
            self.dim == value.dim
            and self.tile_size == value.tile_size
            and self.induction_var == value.induction_var
            and self.iters == value.iters
        )

    @property
    def count(self) -> IndexExpr:
        """
        Returns an expression for the number of iterations in the loop.
        """
        if self.iters:
            return self.iters
        return ceiling(self.dim / self.tile_size)

    def apply(self) -> IndexSequence:
        if self.induction_var is None:
            raise ValueError(
                "Index is being computed without setting induction variable"
            )
        return IndexSequence(self.induction_var * self.tile_size, 1)


@dataclass
class UnrollingConstraint(Constraint):
    """
    A constraint of the form `tkw.UnrollingConstraint(K)` specifies that we want
    to fully unroll the K dimension.
    This is only used for verification / consistency purposes and does not add
    an index constraint to the K-th dimension.
    """

    dim: IndexExpr

    def __eq__(self, value):
        if not isinstance(value, UnrollingConstraint):
            return False
        return self.dim == value.dim

    def __hash__(self):
        return hash((self.dim,))

    def apply(self) -> IndexSequence:
        return IndexSequence(0, 1)


@dataclass
class WaveConstraint(Constraint):
    """
    A constraint of the form `tkw.WaveConstraint(K, WAVE_K)` specifies
    that we want distribute the K dimension among multiple waves which
    each wave operating on a tile size of WAVE_K. The assumption is
    that the K dimension has already been distributed among workgroups.
    If the K dimension has been distributed among workgroups with a
    tile size of BLOCK_K, then the number of waves along the K dimension
    is given by BLOCK_K // WAVE_K.

    This constraint adds an index constraint to the K-th dimension of a
    a tensor of the form WAVE_K * wave_id. The index of the wave
    is determined by the following mapping:
    workgroup id 0 -> wave/thread id x
    workgroup id 1 -> wave/thread id y
    workgroup id 2 -> wave/thread id z
    (If the tensor dimension has been distributed along workgroup dimension
    {0, 1, 2}, then the corresponding thread id is {x, y, z}).

    Because we represent the number of threads per block as
    [wave_id_0 * threads_per_wave, wave_id_1, wave_id_2], special care is
    required when computing wave_id_0. Specifically,
    wave_id_0 = floor(thread_id_0 / threads_per_wave)
    wave_id_1 = thread_id_1
    wave_id_2 = thread_id_2
    """

    dim: IndexExpr
    tile_size: IndexExpr
    wave_id: Optional[IndexExpr] = None

    def apply(self) -> IndexSequence:
        if self.wave_id is None:
            raise ValueError("Index is being computed without setting wave id")
        return IndexSequence(self.tile_size * self.wave_id, 1)

    def set_wave_id_from_hardware_and_workgroup_constraint(
        self,
        hardware_constraint: HardwareConstraint,
        workgroup_constraint: WorkgroupConstraint,
    ):
        """
        The wave_id is the same as the thread_id, with the exception of
          wave_id[0] = thread_id[0] / threads_per_wave
        This is a convention that we adopt.
        """
        old_wave_id = self.wave_id
        assert self.dim == workgroup_constraint.dim, "Dimension mismatch"
        self.wave_id = hardware_constraint.get_thread_id_from_workgroup_dim(
            workgroup_constraint.workgroup_dim
        )
        if workgroup_constraint.workgroup_dim == 0:
            self.wave_id = floor(self.wave_id / hardware_constraint.threads_per_wave)
        assert (
            old_wave_id is None or self.wave_id == old_wave_id
        ), f"Conflicting preset wave_id old: {old_wave_id} new: {self.wave_id}"


@dataclass
class ThreadConstraint(Constraint):
    """
    A constraint of the form `tkw.ThreadConstraint(K, NUM_THREAD_K)` specifies that we
    want distribute the K dimension among multiple threads with each thread
    operating on a tile size of NUM_THREAD_K.
    The assumption is that the K dimension has already been distributed among
    workgroups.
    If the K dimension has been distributed among workgroups with a
    tile size of BLOCK_K, then the number of threads along the K dimension
    is given by BLOCK_K // NUM_THREAD_K.

    This constraint adds an index constraint to the K-th dimension of a
    a tensor of the form NUM_THREAD_K * thread_id. The index of the thread is
    determined by the following mapping:
    workgroup id 0 -> thread id x
    workgroup id 1 -> thread id y
    workgroup id 2 -> thread id z

    (If the tensor dimension has been distributed along workgroup dimension
    {0, 1, 2}, then the corresponding thread id is {x, y, z}).
    """

    dim: IndexExpr
    tile_size: IndexExpr
    thread_id: Optional[IndexExpr] = None

    def __eq__(self, value):
        if not isinstance(value, ThreadConstraint):
            return False
        return self.dim == value.dim and self.tile_size == value.tile_size

    def __hash__(self):
        return hash((self.dim, self.tile_size))

    def apply(self) -> IndexSequence:
        if self.thread_id is None:
            raise ValueError("Index is being computed without setting thread id")
        return IndexSequence(self.tile_size * self.thread_id, 1)

    def set_thread_id_from_hardware_and_workgroup_constraint(
        self,
        hardware_constraint: HardwareConstraint,
        workgroup_constraint: WorkgroupConstraint,
    ):
        old_thread_id = self.thread_id
        assert self.dim == workgroup_constraint.dim, "Dimension mismatch"
        self.thread_id = hardware_constraint.get_thread_id_from_workgroup_dim(
            workgroup_constraint.workgroup_dim
        )
        assert (
            old_thread_id is None or self.thread_id == old_thread_id
        ), "Conflicting thread_id"


def get_constrained_shape(
    shape: list[IndexExpr], constraints: list[WorkgroupConstraint | TilingConstraint]
) -> tuple[IndexExpr]:
    """
    Given a shape, workgroup and tiling constraints, returns the shape
    of the distributed and tiled tensor. The shape is determined using the following
    criteria:
    0. If no workgroup or tiling constraints are provided, the original shape is used.
    1. If only workgroup constraints are provided, the shape is determined by the
       tile size of the workgroup constraints.
    2. If only tiling constraints are provided, the shape is determined by the
       tile size of the tiling constraints.
    3. If both workgroup and tiling constraints are provided, the shape is determined
       from the tiling constraints*.
    * By choosing tiling constraints, the shared memory used will be less but we will
      not be able to coalesce global memory accesses (minimize_global_loads). If instead
      we choose workgroup constraints, we will be able to coalesce global memory accesses
      but will use more shared memory.
      We choose tiling constraints over workgroup constraints because workgroup constraints
      and tiling constraints will only be used when we cannot coalesce global memory
      accesses because of constraints like dynamic read indices for block tables in
      paged attention.
      To enable workgroup constraints instead, we will additionally need to remove induction
      variables from the global read and shared write indices and ensure that they get
      hoisted out of the loop.
    """
    constrained_shape = list(shape)
    all_same_type = lambda x, type: all(
        isinstance(constraint, type) for constraint in x
    )
    for i, dim in enumerate(shape):
        dim_constraints = [
            constraint
            for constraint in constraints
            if isinstance(constraint, (WorkgroupConstraint, TilingConstraint))
            and dim == constraint.dim
        ]
        if not dim_constraints:
            continue
        if all_same_type(dim_constraints, WorkgroupConstraint) or all_same_type(
            dim_constraints, TilingConstraint
        ):
            constrained_shape[i] = dim_constraints[0].tile_size
            continue
        constrained_shape[i] = [
            x.tile_size for x in dim_constraints if isinstance(x, TilingConstraint)
        ][0]
    return tuple(constrained_shape)


def type_as_int(c: Constraint):
    if isinstance(c, HardwareConstraint):
        return 0
    if isinstance(c, WorkgroupConstraint):
        return 1
    if isinstance(c, TilingConstraint):
        return 2
    if isinstance(c, WaveConstraint):
        return 3
    if isinstance(c, ThreadConstraint):
        return 4
    if isinstance(c, UnrollingConstraint):
        return 5
    return 6


__WORKGROUP_SENTINEL_DIM__ = 100
__WORKGROUP_SENTINEL_IDX__ = IndexSymbol("")


def workgroup_constraints_dim_idx_mapping(
    workgroup_constraints: Sequence[WorkgroupConstraint],
):
    dim_to_idx, idx_to_dim, dim_to_idx_seen = {}, {}, {}
    for constraint in workgroup_constraints:
        assert (
            constraint.workgroup_dim not in dim_to_idx.keys() or not constraint.primary
        ), f"""
        Multiple primary constraints in the same workgroup dimension {constraint}.
        All constraints:
        {workgroup_constraints}
        """
        dim_to_idx[constraint.workgroup_dim] = constraint.dim
        idx_to_dim[constraint.dim] = constraint.workgroup_dim
        dim_to_idx_seen[constraint.workgroup_dim] = constraint.dim
    # Add sentinel.
    dim_to_idx[__WORKGROUP_SENTINEL_DIM__] = __WORKGROUP_SENTINEL_IDX__
    dim_to_idx_seen[__WORKGROUP_SENTINEL_DIM__] = __WORKGROUP_SENTINEL_IDX__
    idx_to_dim[__WORKGROUP_SENTINEL_IDX__] = __WORKGROUP_SENTINEL_DIM__
    return dim_to_idx, idx_to_dim, dim_to_idx_seen


def constraint_list_to_str(constraints: Sequence[Constraint]):
    strs = ["\nConstraints List:"]
    workgroup_constraints = [
        c for c in constraints if isinstance(c, WorkgroupConstraint)
    ]
    dim_to_idx, idx_to_dim, _ = workgroup_constraints_dim_idx_mapping(
        workgroup_constraints
    )
    sorted_constraints = sorted(
        constraints,
        key=lambda c: (
            getattr(
                c,
                "workgroup_dim",
                idx_to_dim[getattr(c, "dim", __WORKGROUP_SENTINEL_IDX__)],
            ),
            not getattr(c, "primary", False),
            getattr(c, "dim", __WORKGROUP_SENTINEL_IDX__).name,
            type_as_int(c),
        ),
    )
    for c in sorted_constraints:
        strs += [f"{c}"]
    return "\n\t".join(strs)


def is_dim_mapped_to_waves(dim: IndexExpr, constraints: Sequence[Constraint]):
    wave_constraints = [
        c for c in constraints if isinstance(c, WaveConstraint) and c.dim == dim
    ]
    return len(wave_constraints) > 0


def is_dim_mapped_to_threads(dim: IndexExpr, constraints: Sequence[Constraint]):
    thread_constraints = [
        c for c in constraints if isinstance(c, ThreadConstraint) and c.dim == dim
    ]
    return len(thread_constraints) > 0


def verify_global_constraints(constraints: Sequence[Constraint]):
    """
    Centralize node-invariant constraint verification.

    This must be called before SymbolicConstraints that may alias are processed.
    Calling before SymbolicConstraints avoids reverse engineering primary
    constraint relationships.
    """
    non_symbolic_constraint = constraints
    # TODO: fix dependency ..
    # non_symbolic_constraint = [c for c in constraints if not isinstance(c, SymbolicConstraint)]

    hardware_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, HardwareConstraint)
    ]
    workgroup_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, WorkgroupConstraint)
    ]
    thread_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, ThreadConstraint)
    ]
    wave_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, WaveConstraint)
    ]
    tiling_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, TilingConstraint)
    ]
    unrolling_constraints = [
        c for c in non_symbolic_constraint if isinstance(c, UnrollingConstraint)
    ]

    # 1. Exactly 1 hardware constraint must be provided.
    assert (
        len(hardware_constraints) == 1
    ), "Exactly one hardware constraint must be provided"
    hw_cons = hardware_constraints[0]

    # 2. Internally, verify no conflict in the primary workgroup dimensions.
    # This must be called before SymbolicConstraints that may alias are processed.
    # Compute dim_to_idx and dim_to_idx.
    dim_to_idx, idx_to_dim, dim_to_idx_seen = workgroup_constraints_dim_idx_mapping(
        workgroup_constraints
    )

    # 3. Verify proper workgroup constraints.
    for workgroup_dim, _ in dim_to_idx_seen.items():
        assert (
            workgroup_dim in dim_to_idx.keys()
        ), f"""
        Need at least a primary constraint in each seen workgroup dimension but
        none found for dim {workgroup_dim}.
        {constraint_list_to_str(non_symbolic_constraint)}
        """

    # 4. Verify dim 0 has a WorkgroupConstraint.
    assert (
        0 in dim_to_idx.keys()
    ), f"""
        Need at least a primary constraint for workgroup dimension 0.
        {constraint_list_to_str(non_symbolic_constraint)}
        """

    # 5. For each workgroup_dim, there should be at least one of either:
    #   - WaveConstraint
    #   - ThreadConstraint
    #   - TilingConstraint
    #   - UnrollingConstraint
    # but at most one thread-bearing constraint
    # In the absence of a constraint, UnrollingConstraint is implicit but we want
    # to really make it explicit in the programming model and ensure the user
    # understands the implication.
    for workgroup_dim, idx in dim_to_idx.items():
        if workgroup_dim == __WORKGROUP_SENTINEL_DIM__:
            continue

        lcs = [
            c
            for c in non_symbolic_constraint
            if not isinstance(c, WorkgroupConstraint)
            and not isinstance(c, HardwareConstraint)
            and hasattr(c, "dim")
            and c.dim == idx
        ]

        # 6. Must always have some other constraint than WorkgroupConstraint.
        assert (
            len(lcs) > 0
        ), f"""
        No constraints found for workgroup dim {workgroup_dim} idx {idx}.
        {constraint_list_to_str(constraints)}
        """

        # 7. Dimension 0 must always carry a thread-bearing constraint to avoid
        # recomputations. Note that in certain cases, where we have a single
        # wave / thread, this is implicitly correct, but we want to make it
        # explicit and force the user to think about it.
        num_thread_bearing_constraints = sum(
            bool(isinstance(lc, WaveConstraint) or isinstance(lc, ThreadConstraint))
            for lc in lcs
        )
        if workgroup_dim == 0 or (
            worgroup_dim < len(hw_cons.waves_per_block)
            and hw_cons.waves_per_block[workgroup_dim] != 1
        ):
            assert (
                num_thread_bearing_constraints == 1
            ), f"""
            Workgroup dim {workgroup_dim} idx {idx} must have exactly a threadIdx.x constraint
            (either WaveConstraint or ThreadConstraint), otherwise it is
            guaranteed that multiple threads will race.
            {constraint_list_to_str(constraints)}
            """


def verify_node_specific_constraints(node: CustomOp, constraints: Sequence[Constraint]):
    """
    Centralize node-specific constraint verification.
    This should be called after node.vector_shapes has been derived

    Lifted from get_dim_scaling, cleanup and TODO doc this.
    """

    hw_cons = hardware_constraints[0]
    idxc = IndexingContext.current()
    for constraint in constraints:
        if not isinstance(constraint, WorkgroupConstraint) and not isinstance(
            constraint, TilingConstraint
        ):
            continue

        tile_size = idxc.get_static_value(constraint.tile_size)
        if constraint.dim not in node.vector_shapes:
            continue
        vector_size = node.vector_shapes[constraint.dim]

        # No dim scaling for dims with 0 vector size.
        if vector_size == 0:
            continue

        wave_count = 1
        if isinstance(constraint, WorkgroupConstraint):
            wave_count = hw_cons.waves_per_block[constraint.workgroup_dim]
        if tile_size is None or wave_count is None or vector_size is None:
            raise ValueError(
                "Tile size, wave count and vector size must be statically known"
            )
        if tile_size % wave_count != 0 or (tile_size / wave_count) % vector_size != 0:
            raise ValueError(
                f"Tile size must be divisible by wave count and vector size, got: "
                f"tile_size={tile_size}, wave_count={wave_count}, vector_size={vector_size}"
            )
        dim_scaling[constraint.dim] = tile_size // wave_count // vector_size
