# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_zeros,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from tests.kernel.wave.common.utils import (
    require_e2e,
    require_cdna2,
    require_cdna3,
    perf_test,
    enable_scheduling_barriers,
    dump_generated_mlir,
    param_bool,
)
from iree.turbine.kernel.wave.constraints import MMAType
import os
import json
from torch.testing import assert_close
from enum import Enum

# Add test shapes for validation and performance testing.
default_test_shapes = {}
default_test_shapes["test_gemm"] = [
    (1024, 5120, 640),
    (2048, 10240, 1280),
    (4096, 20480, 2560),
]
default_test_shapes["test_gemm"] += [
    perf_test(x) for x in default_test_shapes["test_gemm"]
]
default_test_shapes["test_batched_gemm"] = [(8, 256, 128, 192), (32, 1024, 512, 768)]


user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes[test_name]


@require_e2e
@require_cdna2
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE],
)
def testBatchedGemm(shape: tuple[int], enable_scheduling: SchedulingType, request):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={B: 0}
        )
    ]

    # fmt: off
    ### Current
    x, y, z, k = [tkw.IndexMapping.iterator(i) for i in range(4)]
    d0 = tkw.IndexMapping.dynamic_val(0)
    offset_mapping_a = tkw.IndexMapping(
        num_iterators=4,
        inputs= {B: z, M: d0, N: y, K: k},
        outputs={B: z, M:  x, N: y, K: k},
        # offset_mapping_a is fed by:
        #    `idx_reg = tkw.read(ind) # : [B, M]`
        # where idx_reg is d0.
        # We need to match d0 to its indirect read `tkw.read(ind) # : [B, M]`
        dynamic_val_mappings={B: z, M: x},
    )

    # ### The following sytax would be preferred.
    # x, y, z, k = [tkw.IndexMapping.iterator(i) for i in range(4)]
    # d0 = tkw.IndexMapping.dynamic_val(0)
    # offset_mapping_w2 = tkw.IndexMapping(
    #     shape     = [ B, M,  N, K],
    #     iterators = [ z, x,  y, k],
    #     inputs    = [ z, x, d0, k],
    #     output    = [ z, x,  y, k],
    # We need to match d0 to its indirect read `tkw.read(ind) # : [B, M]` which
    # is indexed by `[z, x]`.
    #     dynamic_val_mappings = {d0 : [z, x]},
    # )
    # fmt: on

    @tkw.wave(constraints)
    def batched_gemm(
        ind: tkl.Memory[B, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
        a: tkl.Memory[B, M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            idx_reg = tkw.read(ind)
            a_reg = tkw.read(
                a,
                mapping=offset_mapping_a,
                mapping_dynamic_vals=(idx_reg,),
            )
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        res = tkw.permute(repeat, target_shape=[M, B, N])
        tkw.write(res, c)

    vB, vM, vN, vK = shape
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        B: vB,
        M: vM,
        N: vN,
        K: vK,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    torch.manual_seed(0)
    a = device_randn(vB, vM, vK, dtype=torch.float16)
    b = device_randn(vB, vN, vK, dtype=torch.float16)
    c = device_zeros(vM, vB, vN, dtype=torch.float32)
    ind = torch.ones(vB, vM, dtype=torch.int32).to(device=a.device)
    asm = batched_gemm(ind, a, b, c)

    if dump_generated_mlir:
        filename = f"wave_batched_gemm_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    torch_ref = torch.matmul(
        a[1, 1].unsqueeze(0, 1).repeat(vB, vM, 1), b.transpose(-2, -1)
    )
    assert_close(c.to(torch.float16), torch_ref.transpose(0, 1), atol=1e-3, rtol=5e-3)
