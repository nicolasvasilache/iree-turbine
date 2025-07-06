"""Code generation for generating vector-dialect based kernels.

Such kernels operate on global memory at the boundary, scheduling
actual loads/stores/computes to local vectors using PyTorch tensor
level operations executed as threads over a grid.
"""

import re
from typing import Any, Callable, Type, Optional, Sequence, TypeVar, Union, List
from typing import cast
import types

from dataclasses import dataclass
import inspect
import operator as py_operator

from numpy import shape
import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from iree.compiler.ir import OpResult

from .._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    SymIndex,
    index_expr,
)

from ..lang.kernel_buffer import KernelBuffer

from .._support import dtype

from .._support.tracing import CapturedTrace

from .. import lang as tkl

from ..lang import (
    Index,
)

from .. import ops

from .builder import (
    IRProxyValue,
    ScalarBuilder,
)

from .base import (
    CodegenError,
    NDEBUG,
    ValidationError,
)

from .ir import (
    AffineMap,
    Attribute,
    AffineExpr,
    AffineMapAttr,
    ArrayAttr,
    FunctionType,
    VectorType,
    DenseElementsAttr,
    F32Type,
    IndexType,
    FloatAttr,
    InsertionPoint,
    IrType,
    Location,
    MemRefType,
    Operation,
    ShapedType,
    Value,
    VectorType,
    arith_d,
    func_d,
    math_d,
    vector_d,
    scf_d,
)

from .kernel_codegen import (
    BoundKernelSignature,
)

from . import op_matchers

ArgTypeUnion = Union[IndexSymbol, Type[KernelBuffer]]


@dataclass
class NodeAttrs:
    # By default, integers are assumed signed. We propagate unsigned as graph
    # node attrs.
    unsigned: bool = False

    @staticmethod
    def load(py_value) -> "NodeAttrs":
        if isinstance(py_value, fx.Node):
            return NodeAttrs(unsigned=bool(py_value.meta.get("unsigned")))
        return NodeAttrs()

    def store(self, node: fx.Node):
        node.meta["unsigned"] = self.unsigned


class ThreadEmitter:
    """Emits a 'thread function' as a `func` with a signature derived from the gm."""

    OP_HANDLERS: dict[Any, Callable[["ThreadEmitter", fx.Node], None]] = {}

    def __init__(self, root_sig: BoundKernelSignature, trace: CapturedTrace):
        self._node_values: dict[fx.Node, List[IRProxyValue]] = {}
        self._grid_axis_values: dict[int, IRProxyValue] = {}
        self._root_sig = root_sig
        self.trace = trace
        self.ip = InsertionPoint(root_sig.entry_block)

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self._root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        return values

    def lookup_grid_axis_value(self, grid_axis: int) -> IRProxyValue:
        assert NDEBUG or isinstance(grid_axis, int)
        value = self._grid_axis_values.get(grid_axis)
        if value is None:
            try:
                ir_value = self._root_sig.resolve_by_reference(("grid", grid_axis))
            except KeyError:
                raise CodegenError(f"Grid axis {grid_axis} out of bounds")
            sym_index = SymIndex(IndexingContext.current().new_unbacked_symbol())
            value = IRProxyValue(ir_value, sym_index)
            self._grid_axis_values[grid_axis] = value
        return value

    def bind_node_proxy(
        self, node: fx.Node, proxy: IRProxyValue, *, attrs: Optional[NodeAttrs] = None
    ):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = [proxy]

    def bind_node_proxies(
        self,
        node: fx.Node,
        proxies: list[IRProxyValue],
        *,
        attrs: Optional[NodeAttrs] = None,
    ):
        """Binds a node's result to a list of Python/IR proxy object."""
        assert NDEBUG or (
            all(isinstance(p, IRProxyValue) for p in proxies)
            and isinstance(node, fx.Node)
        )
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = proxies

    def emit(self):
        with self.ip, Location.unknown():
            self.emit_graph(self.trace.get_root_graph())

    def emit_function_call_node(self, node: fx.Node):
        target_op = node.target
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")
        handler(self, node)
        # dump

    def emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function":
                self.emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def emit_subgraph(self, subgraph: fx.Graph, implicit_capture: list[fx.Node]):
        # Map subgraph freevars -> implicit_capture
        freevars = self.trace.region_graph.inner_freevars[subgraph]
        assert len(freevars) == len(
            implicit_capture
        ), f"Expected {len(freevars)} implicit capture args, got {len(implicit_capture)}"
        for freevar, arg in zip(freevars, implicit_capture):
            self._node_values[freevar.node] = self.lookup_node_values(arg)

        # Emit subgraph
        return self.emit_graph(subgraph)

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])


def handle_op(op):
    def decorator(f: Callable[["ThreadEmitter", fx.Node], None]):
        ThreadEmitter.OP_HANDLERS[op] = f
        return None

    return decorator


###############################################################################
# Python/scalar ops
###############################################################################


@handle_op(py_operator.getitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        proxy, index = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguements") from e

    if not isinstance(proxy, fx.Node):
        raise CodegenError(f"Expected fx.Node")
    node_values = emitter.lookup_node_values(proxy)
    emitter.bind_node_proxy(node, node_values[index])


BINARY_ARITHMETIC_OPS = [
    (py_operator.add, "add"),
    (py_operator.mul, "mul"),
    (py_operator.sub, "sub"),
    (py_operator.mod, "mod"),
    (py_operator.floordiv, "floordiv"),
    (py_operator.truediv, "truediv"),
]

UNARY_ARITHMETIC_OPS = [
    (tkl.exp, "exp"),
    (tkl.exp2, "exp2"),
    (tkl.sqrt, "sqrt"),
]


def binary_broadcast(
    lhs: IRProxyValue, rhs: IRProxyValue
) -> tuple[bool, IRProxyValue, IRProxyValue]:
    assert NDEBUG or (isinstance(lhs, IRProxyValue) and isinstance(rhs, IRProxyValue))
    lhs_type = lhs.ir_value.type
    rhs_type = rhs.ir_value.type
    lhs_is_vector = VectorType.isinstance(lhs_type)
    rhs_is_vector = VectorType.isinstance(rhs_type)
    if not lhs_is_vector and not rhs_is_vector:
        # Not vectors: return as-is.
        return False, lhs, rhs

    # Promote to vector.
    if not lhs_is_vector:
        lhs = IRProxyValue(vector_d.splat(VectorType.get([], lhs_type), lhs.ir_value))
    if not rhs_is_vector:
        rhs = IRProxyValue(vector_d.splat(VectorType.get([], rhs_type), rhs.ir_value))
    lhs_type = VectorType(lhs.ir_value.type)
    rhs_type = VectorType(rhs.ir_value.type)

    broadcast_shape = lhs_type.shape
    rhs_shape = rhs_type.shape
    rank = max(len(broadcast_shape), len(rhs_shape))
    while len(broadcast_shape) < rank:
        broadcast_shape.insert(0, 1)
    while len(rhs_shape) < rank:
        rhs_shape.insert(0, 1)

    for i in range(rank):
        a = broadcast_shape[i]
        b = rhs_shape[i]
        if a != b:
            if a != 1 and b != 1:
                raise CodegenError(
                    f"Binary operands are not broadcast compatible: {lhs_type}, {rhs_type}"
                )
            broadcast_shape[i] = rhs_shape[i] = max(a, b)

    lhs_type = VectorType.get(broadcast_shape, lhs_type.element_type)
    rhs_type = VectorType.get(broadcast_shape, rhs_type.element_type)
    if lhs_type != lhs.ir_value.type:
        lhs = IRProxyValue(vector_d.broadcast(lhs_type, lhs.ir_value))
    if rhs_type != rhs.ir_value.type:
        rhs = IRProxyValue(vector_d.broadcast(rhs_type, rhs.ir_value))
    return True, lhs, rhs


def _define_arithmetic_handlers():
    def register_binary_op(op, mnemonic):
        @handle_op(op)
        def _(emitter: ThreadEmitter, node: fx.Node):
            try:
                lhs, rhs = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e

            lhs = cast_py_value(emitter, lhs)
            rhs = cast_py_value(emitter, rhs)
            is_vector, lhs, rhs = binary_broadcast(lhs, rhs)
            if is_vector:
                result = ScalarBuilder.binary_vector_arithmetic(mnemonic, lhs, rhs)
            else:
                result = ScalarBuilder.binary_arithmetic(mnemonic, lhs, rhs)
            emitter.bind_node_proxy(node, result)

    def register_unary_op(op, mnemonic):
        @handle_op(op)
        def _(emitter: ThreadEmitter, node: fx.Node):
            try:
                (val,) = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e

            val = cast_py_value(emitter, val)
            is_vector = VectorType.isinstance(val.ir_value.type)
            if is_vector:
                result = ScalarBuilder.unary_vector_arithmetic(mnemonic, val)
            else:
                result = ScalarBuilder.unary_arithmetic(mnemonic, val)
            emitter.bind_node_proxy(node, result)

    for op, mnemonic in BINARY_ARITHMETIC_OPS:
        # Need to capture these per iteration, not just final value,
        # so call a function.
        register_binary_op(op, mnemonic)

    for op, mnemonic in UNARY_ARITHMETIC_OPS:
        register_unary_op(op, mnemonic)


_define_arithmetic_handlers()

###############################################################################
# Core data movement and indexing ops
###############################################################################


@handle_op(ops.thread_program_id)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        (axis,) = node.args
        axis = Index(axis)
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    proxy_value = emitter.lookup_grid_axis_value(axis)
    # The value we get back is just an unbacked SymInt. Since we have the
    # type information to make a bounded instance, create that sharing the
    # symbol.
    sym_index_type = node.type
    assert issubclass(sym_index_type, SymIndex)
    emitter.bind_node_proxy(
        node,
        IRProxyValue(proxy_value.ir_value, proxy_value.py_value.cast(sym_index_type)),
    )


@handle_op(tkl.to_dtype)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        (val, dtype) = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    ir_type = cast_dtype(emitter, dtype)
    casted = cast_vector(emitter, val, element_type=ir_type)
    emitter.bind_node_proxy(node, IRProxyValue(casted))


@handle_op(ops.kernel_buffer_getitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, slice_spec = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    ref_shape = kb_py_type.symbolic_shape
    slice_spec = cast_slice_spec(emitter, ref_shape, slice_spec)
    start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)
    vector_shape = extract_static_slice_shape(emitter, ref_shape, slice_spec)
    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    pad_attr = ScalarBuilder.zero_attr(element_type)
    pad_value = arith_d.constant(element_type, pad_attr)
    result = vector_d.transfer_read(
        vector_type,
        kb_src,
        start_indices,
        AffineMap.get_identity(len(start_indices)),
        pad_value,
        in_bounds=[True for _ in range(len(vector_shape))],
    )
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(ops.kernel_buffer_setitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, slice_spec, item = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    dest_rank = kb_ir_type.rank
    ref_shape = kb_py_type.symbolic_shape
    slice_spec = cast_slice_spec(emitter, ref_shape, slice_spec)
    start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)
    if dest_rank != len(start_indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(start_indices)}"
        )
    insert_vector = cast_vector(emitter, item, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # Special case rank-0 broadcast.
    if insert_type.rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_ir_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)
        insert_rank = 1

    permutation_map = AffineMap.get_identity(dest_rank)
    vector_d.transfer_write(
        None,
        insert_vector,
        kb_dest,
        start_indices,
        AffineMapAttr.get(permutation_map),
        in_bounds=[True for _ in range(insert_rank)],
    )


###############################################################################
# Memory Ops
###############################################################################


@handle_op(tkl.load)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, multi_index, vector_shape = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, vector_shape)
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    ref_shape = kb_py_type.symbolic_shape
    slice_spec = cast_slice_spec(emitter, ref_shape, multi_index)
    start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)
    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    pad_attr = ScalarBuilder.zero_attr(element_type)
    pad_value = arith_d.constant(element_type, pad_attr)
    result = vector_d.transfer_read(
        vector_type,
        kb_src,
        start_indices,
        AffineMap.get_minor_identity(len(ref_shape), len(vector_shape)),
        pad_value,
        in_bounds=[True for _ in range(len(vector_shape))],
    )
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(tkl.store)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, multi_index, item = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    dest_rank = kb_ir_type.rank
    ref_shape = kb_py_type.symbolic_shape
    slice_spec = cast_slice_spec(emitter, ref_shape, multi_index)
    start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)
    if dest_rank != len(start_indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(start_indices)}"
        )
    insert_vector = cast_vector(emitter, item, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # Special case rank-0 broadcast.
    if insert_rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_ir_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)

    permutation_map = AffineMap.get_minor_identity(dest_rank, insert_rank)
    vector_d.transfer_write(
        None,
        insert_vector,
        kb_dest,
        start_indices,
        AffineMapAttr.get(permutation_map),
        in_bounds=[True for _ in range(insert_rank)],
    )

@handle_op(tkl.transfer_gather)
def _(emitter: ThreadEmitter, node: fx.Node):    
    T = TypeVar('T')
    def asserting_cast(t: Type[T], v: Any) -> T:
        assert isinstance(v, t), f"expected {v} to be of type {t}"
        return cast(T, v)
    
    def asserting_sequence_cast(t: Type[T], v: Any) -> Sequence[T]:
        assert isinstance(v, (list, tuple)), f"expected {v} to be of type {list} or {tuple}"
        assert all(isinstance(item, t) for item in v), f"expected all items in {v} to be of type {t}"
        return cast(Sequence[T], v)

    def get_mlir_value_of_fx_node(n : fx.Node) -> Value:
        node_values = asserting_sequence_cast(Union[IRProxyValue, Value], emitter.lookup_node_values(n))
        if len(node_values) != 1:
            raise CodegenError(
                f"Unsupported emitter.lookup_node_values of len != 1: {node_values}"
            )
        if isinstance(node_values[0], IRProxyValue):
            return asserting_cast(Value, node_values[0].ir_value)
        return asserting_cast(Value, node_values[0])

    # Consider:
    #                                                       %i %j %v  0  %c33
    #   a = tkl.transfer_gather(A, (%i, %j, %vec, 0, %c33), (2, 3, 4, 8, 16))
    # Output should be
    #   multi_index  = [[%i, %i], [%vec[0], %vec[1], %vector_a[2], %vector_a[3]], [0, %c33]]
    #   vector_shape = [  [1, 1],                                   [1, 1, 1, 1], [8,   16]]
    #   offsets      = [  [0, 1],                                   [0, 0, 0, 0], [0,    0]]
    # So then we can take the itertools.product and describe all the extractions.
    def foo(multi_index: Sequence[Union[fx.Node, int]], vector_shape: Sequence[int]):
        def is_vector_fun(v):
            return isinstance(v, fx.Node) and \
                isinstance(get_mlir_value_of_fx_node(v).type, VectorType)
            
        # 1. Which of the indices emit to MLIR vectors and which one is the last of them?
        # We need to unroll across all dimensions [0 .. last_index].
        indices = [i for i, x in enumerate(multi_index) if is_vector_fun(x)]
        last_index = None if len(indices) == 0 else indices[-1]
        if last_index is None:
            return [multi_index, ], [[0] * len(vector_shape),], [vector_shape, ]
        
        # 2. Split the multi_index and vector_shape into two parts:
        tail_multi_index, tail_vector_shape = \
            multi_index[last_index + 1:], vector_shape[last_index + 1:]
        assert len(tail_multi_index) == len(tail_vector_shape), \
            f"Expected tail_multi_index and tail_vector_shape to have the same length, "\
                "got {len(tail_multi_index)} and {len(tail_vector_shape)}"

        index, offset = [], []
        # 3. The first part is the multi_index and vector_shape up to the last index.
        for mi, si in zip(multi_index[:last_index + 1], vector_shape[:last_index + 1]):
            is_vector = is_vector_fun(mi)

            subindex, suboffset = [], []
            # 3.a. Leading positions that are not vector<index> are simply unrolled by 1.
            if not is_vector:
                for offset_value in range(si):
                    subindex.append(mi)
                    suboffset.append(offset_value)
                index.append(subindex)
                offset.append(suboffset)
                continue

            # 3.b. Vector indirections are flattened.
            ir_value = get_mlir_value_of_fx_node(asserting_cast(fx.Node, mi))
            ir_type = asserting_cast(VectorType, ir_value.type)
            import numpy as np
            rank = ir_type.rank
            num_elements = np.prod(ir_type.shape)
            for flat_index in range(num_elements):
                # Convert flat index to multi-dimensional index
                position = np.unravel_index(flat_index, ir_type.shape)
                subindex.append([vector_d.extract(ir_value, static_position=position, dynamic_position=[]), ])
                suboffset.append([0] * rank)
            index.append(subindex)
            offset.append(suboffset)
        
        if len(tail_multi_index) > 0:
            index.append([list(tail_multi_index), ])
            offset.append([[0] * len(tail_multi_index), ])

        return index, offset, list(tail_vector_shape)


    try:
        kb, multi_index, vector_shape = node.args
        kb = asserting_cast(fx.Node, kb)
        multi_index = asserting_sequence_cast(Union[fx.Node, int], multi_index)
        vector_shape = asserting_sequence_cast(int, cast_py_literal(emitter, vector_shape))
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    import itertools
    tmp_index, tmp_offset, vector_slice_shape = foo(multi_index, vector_shape)
    index, offset = list(itertools.product(*tmp_index)), list(itertools.product(*tmp_offset))
    
    resvals = []
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    ref_shape = kb_py_type.symbolic_shape
    element_type = kb_ir_type.element_type
    for idx_list, offset_list in zip(index, offset):
        multi_index = tuple(item for lst in idx_list for item in lst)
        vec_offsets = tuple(item for lst in offset_list for item in lst)

        slice_spec = cast_slice_spec(emitter, ref_shape, multi_index)
        start_indices = extract_slice_starts(emitter, ref_shape, slice_spec)
        vector_type = VectorType.get(vector_slice_shape, element_type)
        
        pad_attr = ScalarBuilder.zero_attr(element_type)
        pad_value = arith_d.constant(element_type, pad_attr)
        read = vector_d.transfer_read(
            vector_type,
            kb_src,
            start_indices,
            AffineMap.get_minor_identity(len(ref_shape), len(vector_slice_shape)),
            pad_value,
            in_bounds=[True for _ in range(len(vector_slice_shape))],
        )
        resvals.append(read)

    # For the final result, we need to stack everything back into vector_shape
    # Build the result vector<vector_shape x element_type>(zero)
    vector_type = VectorType.get(vector_shape, element_type)
    zero_attr = ScalarBuilder.zero_attr(element_type)
    splat_attr = DenseElementsAttr.get_splat(vector_type, zero_attr)
    result = arith_d.constant(vector_type, splat_attr)

    # Compute the leading shape
    delta = len(vector_shape) - len(vector_slice_shape)
    leading_shape = vector_shape[:delta]
    import numpy as np
    num_elements = np.prod(leading_shape)

    # Iteratively extract the values and insert them into the result vector.
    for flat_index in range(num_elements):
        position = np.unravel_index(flat_index, leading_shape)
        result = vector_d.insert(resvals[flat_index], result, static_position=position, dynamic_position=[])
    emitter.bind_node_proxy(node, IRProxyValue(result))


###############################################################################
# Math Ops
###############################################################################


@handle_op(tkl.constant)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    shape = cast_py_literal(emitter, shape)
    dtype = cast_dtype(emitter, dtype)
    constant = ScalarBuilder.constant_vector(value, shape, dtype)
    emitter.bind_node_proxy(node, constant)


###############################################################################
# Reduction Ops
###############################################################################


@handle_op(tkl.dot)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        lhs, rhs, acc = node.args
        lhs = cast_vector(emitter, lhs)
        rhs = cast_vector(emitter, rhs)
        acc = cast_vector(emitter, acc)
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(lhs.type)
    element_type = vector_type.element_type
    rank = vector_type.rank

    n, m, k = (
        AffineExpr.get_dim(0),
        AffineExpr.get_dim(1),
        AffineExpr.get_dim(2),
    )
    indexing_maps = [
        AffineMap.get(3, 0, [n, k]),
        AffineMap.get(3, 0, [k, m]),
        AffineMap.get(3, 0, [n, m]),
    ]
    indexing_maps_attr = [AffineMapAttr.get(map) for map in indexing_maps]
    # TODO: Bad hack, please fix.
    iterator_types = ArrayAttr.get(
        [
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<reduction>"),
        ]
    )

    result = vector_d.ContractionOp(acc.type, lhs, rhs, acc, indexing_maps_attr, iterator_types).result

    emitter.bind_node_proxy(node, IRProxyValue(result))


# TODO: this should not exist, we should only have a contract that can manipulate traced symbolic shapes
@handle_op(tkl.batched_dot)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        lhs, rhs, acc = node.args
        lhs = cast_vector(emitter, lhs)
        rhs = cast_vector(emitter, rhs)
        acc = cast_vector(emitter, acc)
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(lhs.type)
    element_type = vector_type.element_type
    rank = vector_type.rank

    b, n, m, k = (
        AffineExpr.get_dim(0),
        AffineExpr.get_dim(1),
        AffineExpr.get_dim(2),
        AffineExpr.get_dim(3),
    )
    indexing_maps = [
        AffineMap.get(4, 0, [b, n, k]),
        AffineMap.get(4, 0, [b, k, m]),
        AffineMap.get(4, 0, [b, n, m]),
    ]
    indexing_maps_attr = [AffineMapAttr.get(map) for map in indexing_maps]
    # TODO: Bad hack, please fix.
    iterator_types = ArrayAttr.get(
        [
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<reduction>"),
        ]
    )
    result = vector_d.ContractionOp(
        acc.type,
        lhs,
        rhs,
        acc,
        indexing_maps_attr,
        iterator_types,
    ).result
    emitter.bind_node_proxy(node, IRProxyValue(result))


def register_reduction(op):
    def decorator(f: Callable[[IrType, NodeAttrs], vector_d.CombiningKind]):
        @handle_op(op)
        def _(emitter: ThreadEmitter, node: fx.Node):
            try:
                vector, axis, acc = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguements") from e

            axis = cast_py_literal(emitter, axis)
            emit_reduction(emitter, node, vector, axis, acc, f)

    return decorator


def emit_reduction(
    emitter: ThreadEmitter,
    node: fx.Node,
    raw_input,
    axis: int | None,
    raw_acc,
    combiner_callback: Callable[[IrType, NodeAttrs], vector_d.CombiningKind],
):
    # Setup.
    attrs = NodeAttrs.load(raw_input)
    input = cast_vector(emitter, raw_input)
    vector_type = VectorType(input.type)
    element_type = vector_type.element_type
    rank = vector_type.rank

    if raw_acc:
        acc = cast_vector(emitter, raw_acc)
    else:
        acc = arith_d.constant(element_type, ScalarBuilder.zero_attr(element_type))

    combiner = combiner_callback(element_type, attrs)

    if axis is None:
        # Reduce to scalar.
        scalar_result = vector_d.multi_reduction(
            combiner, input, acc, list(range(rank))
        )
        result = vector_d.splat(VectorType.get([], element_type), scalar_result)
        emitter.bind_node_proxy(node, IRProxyValue(result), attrs=attrs)
    else:
        # Reduce to vector.
        vector_result = vector_d.multi_reduction(combiner, input, acc, [axis])
        emitter.bind_node_proxy(node, IRProxyValue(vector_result), attrs=attrs)


@register_reduction(tkl.max)
def _(element_type: IrType, attrs: NodeAttrs) -> vector_d.CombiningKind:
    if ScalarBuilder.is_floating_point_type(element_type):
        # Non-NaN propagating.
        # TODO: Carry a "fastmath" flag on the emitter and choose between this
        # and MAXIMUMF?
        return vector_d.CombiningKind.MAXNUMF
    elif ScalarBuilder.is_integer_type(element_type):
        return (
            vector_d.CombiningKind.MAXUI
            if attrs.unsigned
            else vector_d.CombiningKind.MAXSI
        )

    raise CodegenError(f"No max reduction for type {element_type}")


@register_reduction(tkl.sum)
def _(element_type: IrType, attrs: NodeAttrs) -> vector_d.CombiningKind:
    return vector_d.CombiningKind.ADD


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(tkl.for_loop)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        start, end, step, init_args = node.args
        subgraph = node.kwargs["subgraph"]
        implicit_capture = node.kwargs["implicit_capture"]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Check if init_args is a flattened list of values.
    for arg in init_args:
        if len(emitter.lookup_node_values(arg)) != 1:
            raise CodegenError(f"NYI: For loop init args must be flattened")

    # Get IR values mapping to the node args.
    start = cast_py_value(emitter, start)
    end = cast_py_value(emitter, end)
    step = cast_py_value(emitter, step)

    # Flatten init_args and get IR values for each of them.
    flat_init_args, init_args_spec = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    # Get the subgraph for body of the loop.
    assert isinstance(subgraph, str)
    subgraph = emitter.trace.get_subgraph(subgraph)

    # Create scf.for operation.
    forOp = scf_d.ForOp(
        start.ir_value,
        end.ir_value,
        step.ir_value,
        [a.ir_value for a in flat_init_args],
    )
    # Enter body of for loop.
    with InsertionPoint(forOp.body):
        # TODO: Flatten subgraph args here.
        subgraph_args = [
            node
            for node in subgraph.nodes
            if node.op == "placeholder" and "lifted" not in node.meta
        ]
        # Add mapping for induction variable argument.
        emitter.bind_node_proxy(
            subgraph_args[0], IRProxyValue(forOp.induction_variable)
        )
        # Add mapping for iter_args.
        for i, v in enumerate(forOp.inner_iter_args):
            emitter.bind_node_proxy(subgraph_args[i + 1], IRProxyValue(v))

        ret = emitter.emit_subgraph(subgraph, implicit_capture)
        # Use ret in terminatory of body
        # TODO: Flatten return values here.
        flat_ret_values, ret_spec = pytree.tree_flatten((ret))
        if len(flat_ret_values) == 1 and flat_ret_values[0] == None:
            flat_ret_values = []
        else:
            flat_ret_values = [
                cast_py_value(emitter, value).ir_value for value in flat_ret_values
            ]
        scf_d.YieldOp(flat_ret_values)

    results = forOp.results_
    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in results])


###############################################################################
# Shape Manipulation Ops
###############################################################################


@handle_op(tkl.broadcast)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        vector, leading_sizes = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector = cast_vector(emitter, vector)
    leading_sizes = cast_py_literal(emitter, leading_sizes)

    old_shape = vector.type.shape
    broadcasted_shape = list(leading_sizes) + old_shape
    broadcasted_type = VectorType.get(broadcasted_shape, vector.type.element_type)
    result = vector_d.broadcast(broadcasted_type, vector)
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(tkl.transpose)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        vector, permutation = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector = cast_vector(emitter, vector)
    permutation = cast_py_literal(emitter, permutation)
    new_shape = [vector.type.shape[i] for i in permutation]
    result_type = VectorType.get(new_shape, vector.type.element_type)

    result = vector_d.transpose(result_type, vector, permutation)
    emitter.bind_node_proxy(node, IRProxyValue(result))


###############################################################################
# Conversion utilities
###############################################################################


def cast_py_literal(emitter: ThreadEmitter, value) -> Any:
    """Treats the given value as a Python literal.

    An exception will be raised if it cannot be computed statically.
    """
    if isinstance(value, IndexExpr):
        simplified = IndexingContext.current().simplify_expr(value)
        try:
            return int(simplified)
        except TypeError as e:
            raise CodegenError(
                f"Literal value required but got symbolic value requiring "
                f"dynamic resolution: {simplified}"
            ) from e
    elif isinstance(value, tuple):
        return tuple(cast_py_literal(emitter, v) for v in value)
    elif isinstance(value, list):
        return [cast_py_literal(emitter, v) for v in value]
    elif isinstance(value, dict):
        return {
            cast_py_literal(emitter, k): cast_py_literal(emitter, v)
            for k, v in value.items()
        }
    elif isinstance(value, (int, float, str)):
        return value


def cast_py_value(
    emitter: ThreadEmitter, value: Value, element_type: Optional[IrType] = None
) -> IRProxyValue:
    """
    Converts the given value to an IR Value.
    If the value is a fx.Node, the result of the fx.Node should corresspond to
    exactly one IR Value.
    If the value is a constant, a constant value will be built for it.
    """
    if isinstance(value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(value)
            assert len(node_values) == 1, f"Expected exactly one value for node {value}"
            return (
                node_values[0]
                if isinstance(node_values[0], IRProxyValue)
                else IRProxyValue(node_values[0])
            )
        except KeyError:
            raise CodegenError(f"Producer node `{value}` has no IR Value")
    elif isinstance(value, IndexExpr):
        simplified = IndexingContext.current().simplify_expr(value)
        try:
            value = int(simplified)
        except TypeError as e:
            raise CodegenError(
                f"Dynamically resolved symbolic values not yet implemented. Got: "
                f"{simplified}"
            ) from e
    element_type = IndexType.get() if element_type is None else element_type
    return ScalarBuilder.constant(value, element_type)


def cast_py_lvalue(emitter: ThreadEmitter, py_value: fx.Node) -> tuple[Value, fx.Node]:
    if isinstance(py_value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(py_value)
            assert (
                len(node_values) == 1
            ), f"Expected exactly one value for node {py_value}"
            return node_values[0], py_value
        except KeyError:
            raise CodegenError(f"Producer node `{py_value}` has no IR Value")
    else:
        raise CodegenError(
            f"Required a traced node in the graph. Got: {py_value} (type {type(py_value)})"
        )


def cast_kernel_buffer(
    emitter: ThreadEmitter, kb
) -> tuple[Value, MemRefType, Type[KernelBuffer]]:
    """Casts a Python value of type KernelBuffer, which lowers to a MemRefType'd value."""
    value, node = cast_py_lvalue(emitter, kb)
    ir_type = value.type
    py_type = node.type
    if py_type is None:
        try:
            py_type = ops.wave_ops.get_custom(node).type
        except:
            raise CodegenError(f"Could not find type for node {node}")

    if not MemRefType.isinstance(ir_type):
        raise CodegenError(
            f"Expected a KernelBuffer (aka. `memref`) but got `{ir_type}`"
        )

    if not (issubclass(py_type, KernelBuffer) or issubclass(py_type, tkl.Memory)):
        raise CodegenError(
            f"Expected an lvalue of type KernelBuffer but got '{py_type}' for node {node}"
        )

    return value, MemRefType(ir_type), py_type


def cast_vector(
    emitter: ThreadEmitter, value, *, element_type: Optional[IrType] = None
):
    proxy_value = cast_py_value(emitter, value)

    # Cast scalar types correctly first.
    if element_type and not ShapedType.isinstance(proxy_value.ir_value.type):
        # Implicit scalar type promotion.
        proxy_value = ScalarBuilder.to_dtype(proxy_value, element_type)

    value = proxy_value.ir_value

    # After scalar promotion, promote to vector.
    if VectorType.isinstance(value.type):
        # Already a vector. Coerce or return.
        if element_type is not None:
            value = ScalarBuilder.to_dtype(proxy_value, element_type).ir_value
        # No target element_type.
        return value
    else:
        # Scalar -> vector.
        element_type = value.type
        vector_type = VectorType.get([], element_type)
        return vector_d.splat(vector_type, value)


def cast_scalar(emitter: ThreadEmitter, value):
    proxy_value = cast_py_value(emitter, value)
    value = proxy_value.ir_value

    # After scalar promotion, promote to vector.
    if VectorType.isinstance(value.type):
        # Vector -> scalar.
        zero = arith_d.ConstantOp(IndexType.get(), 0)
        return vector_d.extractelement(value, position=zero)
    else:
        # Already a scalar. Coerce or return.
        # No target element_type.
        return value


def cast_dtype(emitter: ThreadEmitter, dtype: dtype.DataType) -> IrType:
    try:
        ir_dtype = IrType.parse(dtype.ir_type_asm())
    except CodegenError as e:
        raise CodegenError(f"Failed to convert dtype {dtype} to IR type") from e

    return ir_dtype


###############################################################################
# Slice and indexing
###############################################################################

SliceAtom = Union[slice, None, IndexExpr, IRProxyValue]


def cast_slice_spec(
    emitter: ThreadEmitter, ref_shape: tuple[IndexExpr], py_slice_spec
) -> list[SliceAtom]:
    """Casts a node argument to a 'slice spec', normalizing it in the process.

    A 'slice spec' is what can go in the `[]` of an array access. It is either
    a tuple of slice atoms or a single slice atom. A slice atom is one of:
      * `slice` object
      * `None` indicating dimension insertion.
      * elippsis (...) to indicate a space filling `slice()`
      * `IndexExpr` for a constant index value.
      * `IRProxyValue` containing a `SymIndex` for a dynamic index value.
      * mlir `Value` containing a dynamic index

    The numpy page has a good description here:
        https://numpy.org/doc/1.26/user/basics.indexing.html

    As part of casting, this implementation will replace any ellipsis with a
    rank filling number of None values.
    """
    rank = len(ref_shape)
    if not isinstance(py_slice_spec, tuple):
        py_slice_spec = (py_slice_spec,)

    # Rank normalize.
    none_count = sum(x is None for x in py_slice_spec)
    ellipsis_count = sum(... is None for x in py_slice_spec)
    if ellipsis_count == 1:
        # Expand by the original list of slices less any unit dim insertions.
        # If negative, this does nothing and will be caught later upon
        # rank validation.
        expand_index = py_slice_spec.index(...)
        del py_slice_spec[expand_index]
        expansion_count = (rank + none_count) - len(py_slice_spec)
        for _ in range(expansion_count):
            py_slice_spec.insert(expand_index, slice(None))
    elif ellipsis_count > 1:
        raise IndexError(
            f"Cannot index into a rank expanding referrent with multiple `...` values"
        )

    return [
        cast_slice_atom(emitter, ref_shape[i], py_slice_spec[i]) for i in range(rank)
    ]


def cast_slice_atom(
    emitter: ThreadEmitter, dim_size: IndexExpr, py_slice_atom
) -> SliceAtom:
    """Casts a single 'atom' in a slice spec. See cast_slice_spec."""
    if py_slice_atom is None:
        # Pass-through.
        return py_slice_atom
    if isinstance(py_slice_atom, slice):
        # Re-compose.
        idxc = IndexingContext.current()
        start = py_slice_atom.start
        stop = py_slice_atom.stop
        step = py_slice_atom.step

        # Apply start defaults.
        if start is None:
            start = index_expr(0)
        else:
            start = cast_index_value(emitter, start)
        # Apply stop defaults.
        if stop is None:
            # Stop defaults to the dim size.
            stop = idxc.simplify_expr(dim_size)
        else:
            # Cast it.
            stop = cast_index_value(emitter, stop)
        # Apply step defaults.
        if step is None:
            step = index_expr(1)
        else:
            step = cast_index_value(emitter, step)

        return slice(start, stop, step)
    else:
        return cast_index_value(emitter, py_slice_atom)


def cast_index_value(
    emitter: ThreadEmitter, py_index
) -> Union[IRProxyValue, IndexExpr]:
    """Casts an arbitrary py_index value to either a static or dynamic index.

    Static indices are of type IndexExpr and can be completely defined in terms
    of sympy expressions on symbols. Dynamic are computed in the IR in some
    fashion and are IRProxyValue with an py_value of type SymIndex.
    """
    # Already materialized Value.
    if isinstance(py_index, Value):
        return py_index
    
    # Static IndexExpr
    if isinstance(py_index, int):
        return index_expr(py_index)
    if isinstance(py_index, IndexExpr):
        return IndexingContext.current().simplify_expr(py_index)

    # fx.Node -> IRProxyValue.
    if isinstance(py_index, fx.Node):
        # Cast index value.
        try:
            node_values = emitter.lookup_node_values(py_index)
            assert (
                len(node_values) == 1
            ), f"Expected exactly one value for node {py_index}"
            py_index = node_values[0]
        except KeyError:
            raise CodegenError(f"Producer node `{py_index}` has no IR Value")

    if not isinstance(py_index.py_value, (SymIndex, types.NoneType)):
        raise CodegenError(f"Expected dynamic index value but got {py_index}")
    return py_index


def cast_dynamic_index_value(emitter: ThreadEmitter, py_index) -> IRProxyValue:
    """Casts an arbitrary py_index value to a dynamic index.

    If it was a static index, it will be materialized.
    """
    # Already materialized Value.
    if isinstance(py_index, Value):
        return py_index

    py_index = cast_index_value(emitter, py_index)
    if isinstance(py_index, IRProxyValue):
        return py_index

    # Materialize.
    try:
        int_value = int(py_index)
    except TypeError:
        # Need to materialize the expression.
        raise CodegenError(f"NYI: Materialized index expression {py_index}")
    return ScalarBuilder.constant(int_value, IndexType.get())


def extract_slice_starts(
    emitter: ThreadEmitter,
    ref_shape: tuple[IndexExpr, ...],
    slice_spec: list[SliceAtom],
) -> list[Value]:
    def _extract(i):
        atom = slice_spec[i]
        # Already materialized Value.
        if isinstance(atom, Value):
            return atom
        if atom is None:
            return ScalarBuilder.constant(0, IndexType.get())
        elif isinstance(atom, slice):
            return cast_dynamic_index_value(emitter, atom.start).ir_value
        else:
            return cast_dynamic_index_value(emitter, atom).ir_value

    return [_extract(i) for i in range(len(ref_shape))]


def extract_static_slice_shape(
    emitter: ThreadEmitter,
    ref_shape: tuple[IndexExpr, ...],
    slice_spec: list[SliceAtom],
) -> list[int]:
    rank = len(ref_shape)
    shape = [0] * rank
    idxc = IndexingContext.current()
    for i in range(rank):
        atom = slice_spec[i]
        if atom is None:
            # Insert 1 dim.
            shape[i] = 1
        elif isinstance(atom, slice):
            # Compute from slice.
            if atom.step != 1:
                raise CodegenError(f"NYI: Step != 1")
            try:
                expr = idxc.simplify_expr(atom.stop - atom.start)
                shape[i] = int(expr)
            except TypeError:
                raise CodegenError(
                    f"A static shape was required but got: {slice_spec}[{i}] = {expr}"
                )
        else:
            # Index a single value.
            shape[i] = 1
    return shape
