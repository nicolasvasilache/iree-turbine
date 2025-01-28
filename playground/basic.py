# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel._support.tracing import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
import pytest
import torch
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.wave_sim import wave_sim
from torch.testing import assert_close


def test_stuff_with_vector():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0)
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                M: 1,
                N: 1
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
        tkw.write(a_reg + b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with TestLaunchContext(
        {
            M: 128,
            N: 256,
            BLOCK_M: 1,
            BLOCK_N: 1,
            LOAD_ELEMS_PER_THREAD: 1,
            STORE_ELEMS_PER_THREAD: 1,
        },
            canonicalize=True,
    ):
        a = torch.randn(128, 256, dtype=torch.float32)
        b = torch.randn(256, dtype=torch.float32)
        c = torch.zeros(128, 256, dtype=torch.float32)
        print(eltwise(a, b, c).module_op)
        # assert_close(c, a + b)


test_stuff_with_vector()


def test_stuff_with_mma():
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
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with TestLaunchContext(
        {
            M: 256,
            N: 256,
            K: 9,
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


test_stuff_with_mma()
