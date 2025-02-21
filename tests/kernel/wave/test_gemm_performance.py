# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
import torch

from torch.nn import functional as F
from torch.testing import assert_close
from typing import Tuple

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
from iree.turbine.kernel.lang.global_symbols import *
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)

# from tests.kernel.wave.common.shapes import make_shape_param
# from tests.kernel.wave.common.utils import (
#     require_e2e,
#     require_cdna3,
#     enable_scheduling_barriers,
# )

# shapes = [
#     make_shape_param((2048, 1280, 1280), is_perf=False),
#     make_shape_param((2048, 1280, 1280), is_perf=True),
# ]


def get_gemm_kernel(shape, mfma_variant: MMAType, dynamic_dims: bool):  # Input sizes
    M, N, K = tkl.sym.M, tkl.sym.N, tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M, BLOCK_N, BLOCK_K = tkl.sym.BLOCK_M, tkl.sym.BLOCK_N, tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the reduction dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg], multi_buffering_factor=2)
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 64,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }

    return gemm, hyperparams, [], {}


def create_inputs(
    shape: Tuple[int], dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N, K = shape
    A = device_randn((M, K), dtype=dtype)
    B = device_randn((K, N), dtype=dtype)
    return A, B


def validate_accuracy(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    reference = torch.matmul(A, B)
    assert_close(reference, C, check_dtype=False, rtol=1e-5, atol=1e-5)


# TODO: Debug why failing numerics on MI250.
# @require_e2e
# @require_cdna3
# @pytest.mark.parametrize("shape", shapes)
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize(
#     "mfma_variant",
#     [MMAType.F32_16x16x16_F16],
# )
# @pytest.mark.parametrize("enable_scheduling", [True])
# @pytest.mark.parametrize("dynamic_dims", [False])
def test_matmul(
    shape: Tuple[int],
    dtype: torch.dtype,
    mfma_variant: MMAType,
    enable_scheduling: bool,
    dynamic_dims: bool,
    # request,
):
    torch.manual_seed(0)

    A, B = create_inputs(shape, dtype)
    M, N, K = shape
    output_shape = (M, N)

    gemm, hyperparams, dynamic_symbols, dynamic_symbols_map = get_gemm_kernel(
        shape, mfma_variant, dynamic_dims
    )

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # run_bench = request.config.getoption("--runperf")
    # dump_perf = request.config.getoption("--dump-perf-files-path")
    # if run_bench:
    #     config["benchmark_batch_size"] = 10
    #     config["benchmark_repetitions"] = 3
    # if dump_perf is not None:
    #     perf_filename = request.node.name + ".json"
    #     config["benchmark_results_file"] = os.path.join(
    #         dump_perf, "tk_" + perf_filename)

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        # run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        # use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        C = device_zeros(output_shape, dtype=torch.float32)
        mb = gemm(A, B, C)
        print(mb.module_op)
        # validate_accuracy(A, B, C)


test_matmul(
    shape=(2048, 1280, 1280),
    dtype=torch.float16,
    mfma_variant=MMAType.F32_16x16x16_F16,
    enable_scheduling=True,
    dynamic_dims=False,
)
