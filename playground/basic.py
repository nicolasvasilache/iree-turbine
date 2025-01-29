# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pdb import run
import pytest
import torch
from torch.testing import assert_close

from iree.turbine.kernel._support.tracing import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config
)
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw

torch.set_printoptions(linewidth=300)

def reference_row(rows: int, cols: int):
    row_indices = torch.arange(rows).unsqueeze(1).expand(-1, cols)
    return row_indices

def reference_col(rows: int, cols: int):
    col_indices = torch.arange(cols).unsqueeze(0).expand(rows, -1)
    return col_indices

def reference_row_plus_col(rows: int, cols: int):
    row_indices = torch.arange(rows).unsqueeze(1).expand(-1, cols)
    col_indices = torch.arange(cols).unsqueeze(0).expand(rows, -1)
    return row_indices + col_indices


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
# Workgroup tile sizes
# ITERATIONS_OF_M_PER_WAVE = tkl.sym.ITERATIONS_OF_M_PER_WAVE
ITERATIONS_OF_N_PER_WAVE = tkl.sym.ITERATIONS_OF_N_PER_WAVE
BLOCK_K = tkl.sym.BLOCK_K
# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
# Other hyperparameters
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


ITERATIONS_OF_M_PER_WAVE = 32
ITERATIONS_OF_N_PER_WAVE = 32


def test_with_mma():
    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, ITERATIONS_OF_M_PER_WAVE, 0)]
    constraints += [tkw.WaveConstraint(M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, ITERATIONS_OF_N_PER_WAVE, 1)]
    constraints += [tkw.WaveConstraint(N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=MMAType.F32_32x32x8_F16,
        )
    ]

    @tkw.wave(constraints)
    def row(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[K, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        acc = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        res = tkw.mma(a_reg, b_reg, acc)
        i = tkw.self_index(M, tkl.i64)
        res = res + tkw.cast(i, tkl.f32)
        tkw.write(res, c, elements_per_thread=4)

    vM, vN, vK = 64, 32, 8

    def run_harness(fun):
        config = get_default_run_config()
        # Override manually to run.
        config = {"backend": "rocm", "device": "hip", "target": "gfx90a"}
        with TestLaunchContext(
            {
                M: vM,
                N: vN,
                K: vK,
                # ITERATIONS_OF_M_PER_WAVE: 32,
                # ITERATIONS_OF_N_PER_WAVE: 32,
                # LOAD_ELEMS_PER_THREAD: 4,
                # STORE_ELEMS_PER_THREAD: 4,
            },
                canonicalize=True,
                run=True,
                run_config=config):

            fun()

    def fun_row():
        a = device_zeros(vM, vK, dtype=torch.float16)
        b = device_zeros(vK, vN, dtype=torch.float16)
        c = device_zeros(vM, vN, dtype=torch.float32)
        # print(row(a, b, c).module_op)
        row(a, b, c)
        print(c.cpu().to(dtype=torch.int64) - reference_row(vM, vN))
        assert_close(reference_row(vM, vN), c.cpu().to(dtype=torch.int64))

    run_harness(fun_row)


test_with_mma()
