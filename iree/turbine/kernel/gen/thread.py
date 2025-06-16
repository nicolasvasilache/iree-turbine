# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from symtable import SymbolTable
from typing import (
    Type,
    Callable,
    Optional,
)

import inspect
import math

import torch

from ..lang import (
    KernelBuffer,
    Grid,
    IndexExpr,
)

from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    EagerContext,
    Launchable,
    KernelRegionGraph,
    LaunchContext,
    AOTLaunchContext,
)

from .._support.indexing import IndexingContext

from ..compiler import (
    kernel_codegen,
    dispatch_codegen,
    builder,
    vector_codegen,
    host_codegen,
)

from ..compiler.ir import (
    Context,
    Operation,
)

from ..wave.compile_options import WaveCompileOptions
from ..wave.utils.compile_utils import compile_to_vmfb
from ..wave.utils.run_utils import invoke_vmfb

__all__ = [
    "thread",
]

# The key name for GPUPipelineOptionsAttr in the translation info config dictionary.
GPU_PIPELINE_OPTIONS_KEY = "gpu_pipeline_options"
# The key name for llvm_func_attrs attribute in the translation info config dictionary.
LLVM_FUNC_ATTRS_KEY = "llvm_func_attrs"
# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"


from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu, iree_codegen  # type: ignore

def get_translation_info_config(
    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
) -> ir.DictAttr:
    """
    Example IR
    translation_info = #iree_codegen.translation_info<
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
                    }
                >
    """
    waves_per_eu_str = str(waves_per_eu)

    # Create the waves_per_eu dictionary attribute.
    waves_per_eu_dict = ir.DictAttr.get(
        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
    )

    config_dict = ir.DictAttr.get(
        {
            GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
            LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
        }
    )

    return config_dict


def run_vmfb(vmfb, kernel_sig, options, args):
    # Partition arguments into kernel inputs and outputs.
    # ToDo: we should expose the `usage` as a property in binding desc
    #       so that we can reduce the code and use `zip``.
    usage_idx = 0
    scalar_args = []
    kernel_inputs, kernel_outputs = [], []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            scalar_args.append(arg)
            continue
        usage = kernel_sig.kernel_buffer_bindings[usage_idx].kernel_buffer_type.usage
        usage_idx += 1
        if usage == kernel_codegen.KernelBufferUsage.INPUT:
            kernel_inputs.append(arg)
        if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
            kernel_outputs.append(arg)
    kernel_inputs.extend(scalar_args)

    invoke_vmfb(vmfb, options, kernel_inputs, kernel_outputs, run_on_default_device=True)


def compile_and_run_on_cpu(asm: str, kernel_sig, args):
    options = WaveCompileOptions()
    options.device = "local-task"
    options.backend = "llvm-cpu"
    # options.target = "llvm-cpu"
    # options.print_ir_after_all = True
    options.flags = [
        "--iree-llvmcpu-target-cpu=generic",
    ]
    vmfb = compile_to_vmfb(asm, options)
    run_vmfb(vmfb, kernel_sig, options, args)

def compile_and_run_on_gpu(asm: str, kernel_sig, args):
    options = WaveCompileOptions()
    options.backend = "rocm"
    options.device = "hip"
    options.target = "gfx942"
    # options.print_ir_after_all = True
    vmfb = compile_to_vmfb(asm, options)
    run_vmfb(vmfb, kernel_sig, options, args)


def thread(*symbolic_shape: IndexExpr):
    GridType = Grid[symbolic_shape]

    def decorator(f: Callable) -> "LaunchableThread":
        return LaunchableThread(GridType, f.__name__, f)

    return decorator


class LaunchableThread(Launchable):
    def __init__(
        self,
        grid_type: Type[Grid],
        name: str,
        eager_function: Callable,
    ):
        super().__init__(eager_function)
        self.grid_type = grid_type
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)
        return trace

    def eager_execute(self, args, kwargs):
        grid = self.grid_type()
        rank = grid.rank
        with EagerContext(rank=rank) as context:
            sig = self._sig
            bound = sig.bind(*args, *kwargs)
            bound.apply_defaults()
            # Transform args to KernelBuffers.
            for arg_name in list(bound.arguments.keys()):
                arg_value = bound.arguments[arg_name]
                param = sig.parameters[arg_name]
                param_type = param.annotation
                if isinstance(param_type, type) and issubclass(
                    param_type, KernelBuffer
                ):
                    kernel_buffer = param_type(arg_value)
                    bound.arguments[arg_name] = kernel_buffer
            volume = math.prod(grid)
            current_thread = context.current_thread
            for it in range(volume):
                for i in range(rank - 1):
                    current_thread[i] = it // grid[i]
                    it = it % grid[i]
                current_thread[-1] = it
                self._eager_function(*bound.args, **bound.kwargs)

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ):
        # Trace the function.
        trace = self._trace()
        idxc = IndexingContext.current()

        sig = self._sig
        bound = sig.bind(*args, *kwargs)
        bound.apply_defaults()

        for arg_name in list(bound.arguments.keys()):
            arg_value = bound.arguments[arg_name]
            param = sig.parameters[arg_name]
            param_type = param.annotation
            if isinstance(param_type, type) and issubclass(param_type, KernelBuffer):
                assert isinstance(arg_value, torch.Tensor)
                idxc.bind_shaped(arg_name, param_type, list(arg_value.shape))

        idxc.finalize()

        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(trace.get_root_graph())
        kernel_sig.add_grid(self.grid_type)

        grid = self.grid_type()

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)
        emitter = vector_codegen.ThreadEmitter(dispatch_entrypoint, trace)
        emitter.emit()
        emitter.finish()

        # print(mb.module_op)
        mb.module_op.verify()

        return mb, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs
        )
        host_codegen.isolated_test_call(mb, exe, kernel_sig, entrypoint_name)

        gpu: bool = True
        if gpu:
            with ir.Context() as context:
                # Find the module_op and the function entry point under the stream executable `exe`.
                stream_executable_module = exe.def_module.module_op
                symbol_table = ir.SymbolTable(stream_executable_module)
                def_func_op = symbol_table[entrypoint_name]
                # Build the iree_codegen attr for vector distribution.
                pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
                    iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
                )
                translation_info = iree_codegen.TranslationInfoAttr.get(
                    pipeline_attr, None, workgroup_size=[64, 1, 1], subgroup_size=64
                )
                # Attach as a func attr.
                def_func_op.attributes["translation_info"] = translation_info
                # Compile and run on GPU.
                compile_and_run_on_gpu(mb.module_op.get_asm(), kernel_sig, args)
        else: 
            compile_and_run_on_cpu(mb.module_op.get_asm(), kernel_sig, args)

    def aot_execute(self, args, kwargs):
        launch_context = LaunchContext.current()
        assert isinstance(launch_context, AOTLaunchContext)

        module = launch_context.module

        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs, context=module.context, module_op=module.operation
        )

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[{self.grid_type}]"
