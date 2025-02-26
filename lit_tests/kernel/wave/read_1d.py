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
)


# Symbols
M, BLOCK_M, ADDRESS_SPACE, ELEMENTS_PER_LOAD, ELEMENTS_PER_STORE = (
    tkl.sym.M,
    tkl.sym.BLOCK_M,
    tkl.sym.ADDRESS_SPACE,
    tkl.sym.ELEMENTS_PER_LOAD,
    tkl.sym.ELEMENTS_PER_STORE,
)


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=kwargs["waves_per_block"]
            if "waves_per_block" in kwargs
            else (1, 1, 1),
            # One must always specify mma_type or vector_shapes.
            vector_shapes=kwargs["vector_shapes"] if "vector_shapes" in kwargs else {},
        )
    ]
    return constraints


def harness_1d_global_mem(build_constraints_fun, kernel_fun, *args, **kwargs):
    constraints = build_constraints_fun(*args, **kwargs)
    with tk.gen.TestLaunchContext(
        kwargs["static_symbols"] if "static_symbols" in kwargs else {}
    ):
        lw = LaunchableWave(constraints, "kernel_fun", kernel_fun)

        trace: CapturedTrace = lw._trace()
        idxc: IndexingContext = IndexingContext.current()
        graph_passes = lw.build_initial_pass_pipeline(trace, idxc)
        for p in graph_passes:
            lw.try_apply_pass(p, trace)

        lw.infer_grid_shape(idxc)

        compile_config = get_default_compile_config()
        with Context() as context:
            mb, trace, exe, kernel_sig, entrypoint_name = lw.compile_to_mlir(
                trace, compile_config, context, **kwargs
            )
            print(mb.module_op)


def single_read(
    a: tkl.Memory[M, ADDRESS_SPACE, tkl.f16], b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16]
):
    a_reg = tkw.read(a, elements_per_thread=ELEMENTS_PER_LOAD)
    tkw.write(a_reg, a, elements_per_thread=ELEMENTS_PER_STORE)


def static_config_1xwave_block_1xvector_1xload_1xstore():
    return {
        "static_symbols": {
            M: 128,
            BLOCK_M: 1,
            ELEMENTS_PER_LOAD: 1,
            ELEMENTS_PER_STORE: 1,
        },
        "vector_shapes": {M: 1},
        "waves_per_block": (1, 1, 1),
        # "dynamic_symbols": [M],
        "canonicalize": {True},
    }


def dynamic_config_1xwave_block_1xvector_1xload_1xstore():
    return {
        "static_symbols": {
            BLOCK_M: 1,
            ELEMENTS_PER_LOAD: 1,
            ELEMENTS_PER_STORE: 1,
        },
        "vector_shapes": {M: 1},
        "waves_per_block": (1, 1, 1),
        "dynamic_symbols": [M],
        "canonicalize": {True},
    }


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


@run_test
def static_correct_2():
    ### 2xload-1 / 2xstore-1 times 64
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    # CHECK-LABEL: static_correct_2
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #         CHECK: func.func @kernel_fun
    ### 2 * wid0 + tidx
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    ### 2 * wid0 + tidx + 1
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   load
    # CHECK-COUNT-2:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def static_correct_3():
    ### 1xload-2 / 1xstore-2 times 64
    # FIXME: in the absence of vector_shapes = {M: 2} (i.e. when {M: 1}), the
    # generated IR is incorrect!
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    cfg["static_symbols"][ELEMENTS_PER_LOAD] = 2
    cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
    cfg["vector_shapes"] = {M: 2}
    # CHECK-LABEL: static_correct_3
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #       CHECK: func.func @kernel_fun
    ### 2 * wid0 + 2 * tidx
    #       CHECK:   arith.addi {{.*}}
    #       CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #   CHECK-NOT:   load
    #       CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #   CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def static_correct_4():
    ### 2xload-1 / 2xstore-1 times 128
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    cfg["waves_per_block"] = (2, 1, 1)
    # CHECK-LABEL: static_correct_4
    #       CHECK: workgroup_size = [128, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #         CHECK: func.func @kernel_fun
    ### 2 * wid0 + tidx
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   load
    #         CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def static_compiler_error_1():
    try:
        # Compiler crashes when vector_shape does not divide WorkgroupConstraint.
        cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
        cfg["static_symbols"][BLOCK_M] = 2
        cfg["vector_shapes"] = {M: 4}
        # CHECK-LABEL: static_compiler_error_1
        harness_1d_global_mem(build_block_constraints, single_read, **cfg)
    except ValueError as e:
        #       CHECK: Tile size must be divisible by wave count and vector size, got: tile_size=2, wave_count=1, vector_size=4
        print(str(e))


@run_test
def static_compiler_error_2():
    try:
        # Compiler crashes when ELEMENTS_PER_LOAD / ELEMENTS_PER_STORE disagree.
        cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
        cfg["static_symbols"][ELEMENTS_PER_LOAD] = 1
        cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
        # CHECK-LABEL: static_compiler_error_2
        harness_1d_global_mem(build_block_constraints, single_read, **cfg)
    except AssertionError as e:
        #       CHECK: Shape doesn't match: (1,) and (2,)
        print(str(e))


@run_test
def static_compiler_error_3():
    # Compiler crashes when vector_shape is not specified and M cannot be
    # attached to either a tkw.mma op or an explicit declaration.
    try:
        cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
        cfg["vector_shapes"] = {}
        # CHECK-LABEL: static_compiler_error_3
        harness_1d_global_mem(build_block_constraints, single_read, **cfg)
    except KeyError as e:
        #       CHECK: M
        print(str(e))


@run_test
def static_compiler_error_4():
    # Compiler crashes when waves_per_block does not divide WorkgroupConstraint.
    try:
        cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
        cfg["waves_per_block"] = (2, 1, 1)
        # CHECK-LABEL: static_compiler_error_4
        harness_1d_global_mem(build_block_constraints, single_read, **cfg)
    except ValueError as e:
        #       CHECK: Tile size must be divisible by wave count and vector size, got: tile_size=1, wave_count=2, vector_size=1
        print(str(e))


@run_test
def static_incorrect_1():
    ### BUG: load-2 / store-2 times 128 when cfg["vector_shapes"] = {M: 2} is missing
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][ELEMENTS_PER_LOAD] = 2
    cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
    # CHECK-LABEL: static_incorrect_1
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c128:.*]] = arith.constant 128 : index
    #       CHECK:   stream.return %[[c128]], {{.*}}
    #          CHECK: func.func @kernel_fun
    ### BUG: wid0 + 2 * tidx
    #          CHECK:   arith.addi {{.*}}
    #          CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   load
    #          CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def static_incorrect_2():
    ### BUG: 2xload-2 / 2xstore-2 times 64 when cfg["vector_shapes"] = {M: 2} is missing
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    cfg["static_symbols"][ELEMENTS_PER_LOAD] = 2
    cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
    # CHECK-LABEL: static_incorrect_2
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #          CHECK: func.func @kernel_fun
    ### 2 * wid0 + 2 * tidx
    #          CHECK:   arith.addi {{.*}}
    #          CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    ### 2 * wid0 + 2 * tidx + 1
    #          CHECK:   arith.addi {{.*}}
    #          CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   load
    #  CHECK-COUNT-2:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def static_incorrect_3():
    ### BUG: 2xload-2 / 2xstore-2 times 64 when cfg["vector_shapes"] = {M: 2} is missing
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    cfg["static_symbols"][ELEMENTS_PER_LOAD] = 2
    cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
    cfg["waves_per_block"] = (2, 1, 1)
    # CHECK-LABEL: static_incorrect_3
    #       CHECK: workgroup_size = [128, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #          CHECK: func.func @kernel_fun
    ### 2 * wid0 + 2 * tidx
    #          CHECK:   arith.addi {{.*}}
    #          CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   load
    #          CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<2xf16>
    #      CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def dynamic_correct_1():
    ### load-1 / store-1 times 1x?
    cfg = copy.deepcopy(dynamic_config_1xwave_block_1xvector_1xload_1xstore())
    # CHECK-LABEL: dynamic_correct_1
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups(%[[NUM_WG:.*]]: index)
    #       CHECK:   stream.return %[[NUM_WG]], {{.*}}
    #         CHECK: func.func @kernel_fun
    ### wid0 + tidx
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<?xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   load
    #         CHECK:   vector.store {{.*}} : memref<?xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   store
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def dynamic_correct_2():
    ### load-1 / store-1 times 2x?
    cfg = copy.deepcopy(dynamic_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    # CHECK-LABEL: dynamic_correct_2
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups(%[[NUM_WG:.*]]: index)
    ### (NUM_WG - 1) / 2 + 1
    #       CHECK:   %[[c0:.*]] = arith.constant 0 : index
    #       CHECK:   arith.select {{.*}} %[[c0]]
    #       CHECK:   stream.return {{.*}}
    #         CHECK: func.func @kernel_fun({{.*}}, %[[M:.*]]: index)
    ###  idx: 2 * wid0 + tidx
    ### mask: 2 * wid0 + tidx < M
    #         CHECK:   %[[m0:.*]] = arith.cmpi {{.*}}
    #         CHECK:   vector.gather {{.*}}, %[[m0]], {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    ###  idx: 2 * wid0 + tidx + 1
    ### mask: 2 * wid0 + tidx + 1 < M
    #         CHECK:   %[[m1:.*]] = arith.cmpi {{.*}}
    #         CHECK:   vector.gather {{.*}}, %[[m1]], {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    #     CHECK-NOT:   load
    #     CHECK-NOT:   gather
    #         CHECK:   vector.scatter {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    #         CHECK:   vector.scatter {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    #     CHECK-NOT:   store
    #     CHECK-NOT:   scatter
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


@run_test
def dynamic_correct_3():
    ### 1xload-2 / 1xstore-2 times 64
    # FIXME: in the absence of vector_shapes = {M: 2} (i.e. when {M: 1}), the
    # generated IR is incorrect!
    cfg = copy.deepcopy(dynamic_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 2
    cfg["static_symbols"][ELEMENTS_PER_LOAD] = 2
    cfg["static_symbols"][ELEMENTS_PER_STORE] = 2
    cfg["vector_shapes"] = {M: 2}
    # CHECK-LABEL: dynamic_correct_3
    #       CHECK: workgroup_size = [64, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups(%[[NUM_WG:.*]]: index)
    ### (NUM_WG - 1) / 2 + 1
    #       CHECK:   %[[c0:.*]] = arith.constant 0 : index
    #       CHECK:   arith.select {{.*}} %[[c0]]
    #       CHECK:   stream.return {{.*}}
    #         CHECK: func.func @kernel_fun({{.*}}, %[[M:.*]]: index)
    ###  idx: 2 * wid0 + 2 * tidx
    ### mask: 2 * wid0 + 2 * tidx < M
    #         CHECK:   %[[m0:.*]] = arith.cmpi {{.*}}
    #         CHECK:   vector.gather {{.*}}, %[[m0]], {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    #     CHECK-NOT:   load
    #     CHECK-NOT:   gather
    #         CHECK:   vector.scatter {{.*}} : memref<?xf16, strided<[1], offset: ?>>
    #     CHECK-NOT:   store
    #     CHECK-NOT:   scatter
    harness_1d_global_mem(build_block_constraints, single_read, **cfg)


def build_block_wave_constraints(*args, **kwargs):
    constraints = build_block_constraints(*args, **kwargs)
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    print(constraints)
    return constraints


@run_test
def static_correct_4():
    ### 2xload-1 / 2xstore-1 times 128
    cfg = copy.deepcopy(static_config_1xwave_block_1xvector_1xload_1xstore())
    cfg["static_symbols"][BLOCK_M] = 4
    cfg["waves_per_block"] = (2, 1, 1)
    # CHECK-LABEL: static_correct_4
    #       CHECK: workgroup_size = [128, 1, 1] subgroup_size = 64
    #       CHECK: stream.executable.export public @kernel_fun workgroups()
    #       CHECK:   %[[c64:.*]] = arith.constant 64 : index
    #       CHECK:   stream.return %[[c64]], {{.*}}
    #         CHECK: func.func @kernel_fun
    ### 2 * wid0 + tidx
    #         CHECK:   arith.addi {{.*}}
    #         CHECK:   vector.load {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   load
    #         CHECK:   vector.store {{.*}} : memref<128xf16, strided<[1], offset: ?>>, vector<1xf16>
    #     CHECK-NOT:   store
    harness_1d_global_mem(build_block_wave_constraints, single_read, **cfg)
