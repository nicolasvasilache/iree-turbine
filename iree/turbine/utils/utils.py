import inspect
from multiprocessing import context
from typing import cast
from typing import Any, Callable, TypeVar, Type

from numpy import isin
from sympy import Symbol, content

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl

from iree.compiler import ir
from iree.compiler.dialects.transform import interpreter as transform_interpreter
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.compiler.builder import ModuleBuilder
from iree.turbine.kernel.compiler.dispatch_codegen import StreamExecutable
from iree.turbine.kernel.compiler.kernel_codegen import KernelSignature
from iree.turbine.kernel.compiler.vector_codegen import ThreadEmitter
from iree.turbine.kernel.wave.compile_options import WaveCompileOptions
from iree.turbine.kernel.wave.wave import LaunchableWave


T = TypeVar('T')
def asserting_cast(t: Type[T], v: Any) -> T:
    assert isinstance(v, t), f"expected {v} to be of type {t}"
    return cast(T, v)


def run(func: Callable[[], None]) -> Callable[[], None]:
    func()


TRANSFORM_MODULE = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }
}
"""


def apply_transform(module: ir.Operation):
    with module.context:
        transform_module = ir.Module.parse(TRANSFORM_MODULE)
        transform_interpreter.apply_named_sequence(
            module,
            transform_module.body.operations[0],
            transform_module,
        )


def compilation_test_harness(substitutions, function):
    """
    Minimal test harness to access the low level APIs necessary to trace
    and compile to MLIR without getting lost in multi-level decorators and
    signature bindings that we don't need for lit tests.

    By default, this maps to a single workgroup and applies canonicalization + cse.

    Returns the MLIR module that can be printed or further processed.
    """
  
    # Bind static and dynamic symbols.
    dynamic_symbols = []
    idxc = IndexingContext()
    idxc.__enter__()
    for k, v in substitutions.items():
        if isinstance(v, int):
          idxc.bind_constant(k, v)
        else:
          dynamic_symbols.append(k)

    # Single workgroup Grid.
    ___ONE___ = tkl.sym.___ONE___
    grid = tkl.Grid[___ONE___]
    grid.rank = 1
    grid.dims = [1]
    
    # LaunchableWave-like codegen but simpler on top of LaunchableThread._trace
    lt = tk.gen.LaunchableThread(grid, name=function.__name__, eager_function=function)
    trace = lt._trace()
    entrypoint_name = function.__name__
    root_graph = trace.get_root_graph()
    kernel_sig = KernelSignature()
    kernel_sig.add_from_graph_placeholders(root_graph)
    kernel_sig.add_from_dynamic_symbols(dynamic_symbols)
    kernel_sig.add_grid(grid)
    kernel_sig.determine_input_output_buffers(root_graph)
    mb = ModuleBuilder(context=None, module_op=None)
    exe = StreamExecutable(mb, name=entrypoint_name)
    dispatch_entrypoint = exe.define_entrypoint(
      entrypoint_name,
      kernel_sig,
      grid,
      dynamic_symbols=dynamic_symbols,
    )

    # Necessary to "freeze" the substitutions in the IndexingContext before emitting.
    idxc.finalize()
    emitter = ThreadEmitter(dispatch_entrypoint, trace, dynamic_symbols=dynamic_symbols)
    emitter.emit()
    emitter.finish()

    # Post emission transforms
    module = mb.module_op
    apply_transform(module)

    idxc.__exit__(None, None, None)
    return module
