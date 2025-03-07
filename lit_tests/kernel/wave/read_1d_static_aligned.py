# RUN: python %s | FileCheck %s

import copy
import logging
from typing import Sequence

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph
from iree.turbine.kernel.wave.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.compiler.ir import Context, Location, Module
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.wave import LaunchableWave
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel.wave.utils import (
    get_default_compile_config,
    print_trace,
    run_test,
    try_apply_pass,
)


# Symbols
(
    M,
    BLOCK_M,
    ITERATIONS_PER_THREAD_M,
    ADDRESS_SPACE,
    ELEMENTS_PER_LOAD,
    ELEMENTS_PER_STORE,
) = (
    tkl.sym.M,
    tkl.sym.BLOCK_M,
    tkl.sym.ITERATIONS_PER_THREAD_M,
    tkl.sym.ADDRESS_SPACE,
    tkl.sym.ELEMENTS_PER_LOAD,
    tkl.sym.ELEMENTS_PER_STORE,
)


def harness_1d_global_mem(build_constraints_fun, kernel_fun, *args, **kwargs):
    constraints = build_constraints_fun(*args, **kwargs)
    with tk.gen.TestLaunchContext(
        kwargs["static_symbols"] if "static_symbols" in kwargs else {}
    ):
        lw = LaunchableWave(constraints, "kernel_fun", kernel_fun)

        trace: CapturedTrace = lw._trace()
        graph_passes = lw.build_initial_pass_pipeline(trace)
        for p in graph_passes:
            try_apply_pass(p, trace, ["all"])

        idxc: IndexingContext = IndexingContext.current()
        lw.infer_grid_shape(idxc)

        compile_config = get_default_compile_config()
        with Context() as context:
            mb, trace, exe, kernel_sig, entrypoint_name = lw.compile_to_mlir(
                trace, context, *args, **kwargs
            )
            print(mb.module_op)


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = [
        # Each workgroup handles BLOCK_M elements.
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        # Each thread handles ITERATIONS_PER_THREAD_M element
        tkw.ThreadConstraint(M, ITERATIONS_PER_THREAD_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            threads_per_block=kwargs["threads_per_block"]
            if "threads_per_block" in kwargs
            else (64, 1, 1),
        ),
    ]
    return constraints


def static_config_1xwave_block_1xvector_1xload_1xstore():
    return {
        "static_symbols": {
            M: 256,
            BLOCK_M: 256,
            ITERATIONS_PER_THREAD_M: 1,
            ELEMENTS_PER_LOAD: 2,
            ELEMENTS_PER_STORE: 2,
        },
        # It is the user's responsibility to ensure that:
        #   threads_per_block * ITERATIONS_PER_THREAD_M * ELEMENTS_PER_LOAD
        # spans exactly the size of the input (i.e. [0, M[).
        # Or IOW that threads_per_block, ITERATIONS_PER_THREAD_M,
        # ELEMENTS_PER_LOAD and vector_shapes agree.
        "threads_per_block": (64, 1, 1),
        # "dynamic_symbols": [M],
        "canonicalize": {True},
    }


from collections import OrderedDict


def single_read(
    a: tkl.Memory[M, ADDRESS_SPACE, tkl.f16], b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16]
):
    a_reg = tkw.read(
        a,
        elements_per_thread=ELEMENTS_PER_LOAD,
        manually_specified_indexing=OrderedDict([(M, 1)]),
    )
    tkw.write(
        a_reg,
        a,
        elements_per_thread=ELEMENTS_PER_STORE,
        manually_specified_indexing=OrderedDict([(M, 1)]),
    )


@run_test
def static_correct_1():
    ### load-1 / store-1 times 128
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    # CHECK-LABEL: static_correct_1
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c128:.*]] = arith.constant 128 : index
    #       CHECK:   stream.return %[[c128]], {{.*}}
    #         CHECK: func.func @kernel_fun
    ### wid0 + tidx
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   load
    #         CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)
