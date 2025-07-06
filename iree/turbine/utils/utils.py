from typing import Callable

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl

from iree.compiler import ir
from iree.compiler.dialects.transform import interpreter as transform_interpreter
from iree.turbine.kernel._support.indexing import IndexingContext


def run(func: Callable[[], None]) -> Callable[[], None]:
    func()


def compilation_test_harness(substitutions, function):
    """
    Minimal test harness to access the low level APIs necessary to trace
    and compile to MLIR without getting lost in multi-level decorators and
    signature bindings that we don't need for lit tests.

    By default, this maps to a single workgroup and applies canonicalization + cse.

    Returns the MLIR module that can be printed or further processed.
    """

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
  
    idxc = IndexingContext()
    idxc.__enter__()
    for k, v in substitutions.items():
        idxc.bind_constant(k, v)
    
    ONE = tkl.sym.ONE
    idxc.bind_constant(ONE, 1)
    grid = tkl.Grid[ONE]
    
    lt = tk.gen.LaunchableThread(grid, name=function.__name__, eager_function=function)
    mb, _, _, _ = lt._trace_and_get_kernel_signature(
        args=None,
        kwargs=None,
        skip_argument_binding=True)
    module = mb.module_op
    apply_transform(module)

    idxc.__exit__(None, None, None)
    return module
