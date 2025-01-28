# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pdb import run
from iree.turbine.kernel._support.tracing import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import get_default_run_config
import pytest
import torch
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.wave_sim import wave_sim
from torch.testing import assert_close

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
# Other hyperparameters
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


def test_with_vector_broadcast_1d():
    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                M: 1,
                N: 2
            },
        )
    ]

    @tkw.wave(constraints)
    def eltwise(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        idx1 = tkw.self_index(N, tkl.i64)
        idx2 = tkw.self_index(M, tkl.i64)
        b_reg = b_reg + tkw.cast(idx1, tkl.f32) + tkw.cast(idx2, tkl.f32)
        tkw.write(a_reg + b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    config = get_default_run_config()
    # Override manually to run.
    config = {"backend": "rocm", "device": "hip", "target": "gfx90a"}
    with TestLaunchContext(
        {
            M: 128,
            N: 256,
            BLOCK_M: 4,
            BLOCK_N: 8,
            LOAD_ELEMS_PER_THREAD: 2,
            STORE_ELEMS_PER_THREAD: 2,
        },
            canonicalize=True,
            run=True,
            run_config=config):
        a = torch.randn(128, 256, dtype=torch.float32)
        b = torch.randn(256, dtype=torch.float32)
        c = torch.zeros(128, 256, dtype=torch.float32)
        # print(eltwise(a, b, c).module_op)
        # eltwise(a, b, c).module_op.verify()
        eltwise(a, b, c)
        # assert_close(c, a + b)


test_with_vector_broadcast_1d()

def test_with_vector_broadcast_1d_as_2d():
    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                M: 1,
                N: 2
            },
        )
    ]

    @tkw.wave(constraints)
    def eltwise(c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32], ):
        i = tkw.self_index(M, tkl.i64)
        # Here we take into account static information that is passed through the
        # transformation:
        #   1. M is explicitly mapped to blockid.x and threadidx.x / waveidx.x
        #   2. N is explicitly mapped to blockid.y and threadidx.y / waveidx.y
        #   3. the multiplicity along threadidx.y and threadidx.z is always 1.
        #      This is a strong assumption of TKW.
        # As a consequence we can treat broadcast along N as a 1 -> M
        # cast + broadcast.
        # While this is very dependent on the strong assumption of TKW and
        # generally incorrect, it is a reasonable short term solution, once
        # implications are understood, to avoid broadcast to 2-D + permute.
        # Note: making such strong assumptions explicit will be necessary in the
        # future. This is the type of situation in which TD shines.
        j = tkw.broadcast(tkw.self_index(N, tkl.i64), target_shape=[M])
        # i = tkw.broadcast(tkw.self_index(N, tkl.i64), target_shape=[M, N])
        # j = tkw.broadcast(tkw.self_index(1, tkl.i64), target_shape=[N, M])
        # j = tkw.permute(j, target_shape=[M, N])
        # Somehow propagation does not happen as in `test_with_vector` and we are
        # subject to BinaryPyOp requires lhs and rhs shape to be at least
        # broadcastable. got (N,) vs (M,).
        # At least one of the 2 is wrong.
        res = tkw.cast(i, tkl.f32) + tkw.cast(j, tkl.f32)
        tkw.write(res, c, elements_per_thread=2)

    with TestLaunchContext(
        {
            M: 1024,
            N: 2048,
            BLOCK_M: 256,
            BLOCK_N: 8,
            LOAD_ELEMS_PER_THREAD: 1,
            STORE_ELEMS_PER_THREAD: 1,
        },
            canonicalize=True,
    ):
        out = torch.zeros(128, 256, dtype=torch.float32)
        print(eltwise(out).module_op)
        # eltwise(a, b, c).module_op.verify()
        # assert_close(c, a + b)


test_with_vector_broken()



def test_with_mma():
    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=MMAType.F32_32x32x8_F16,
        )
    ]

    @tkw.wave(constraints)
    def eltwise(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[K, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        acc = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        res = tkw.mma(a_reg, b_reg, acc)
        idx1 = tkw.self_index(N, tkl.i64)
        idx2 = tkw.self_index(M, tkl.i64)
        res = res + tkw.cast(idx1, tkl.f32) + tkw.cast(idx2, tkl.f32)
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with TestLaunchContext(
        {
            M: 256,
            N: 256,
            K: 8,
            BLOCK_M: 32,
            BLOCK_N: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
        },
            canonicalize=True,
    ):
        a = torch.randn(256, 256, dtype=torch.float16)
        b = torch.randn(256, dtype=torch.float16)
        c = torch.zeros(256, 256, dtype=torch.float32)
        print(eltwise(a, b, c).module_op)
        # assert_close(c, a + b)


# test_with_mma()
