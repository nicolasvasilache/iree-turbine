# Initial Pass Order (pre-optimization)

## initialize_iter_args
```
def initialize_iter_args(trace: CapturedTrace) -> None:
    """
    Initializes the IterArgs in each reduction with an index
    based on their location in the graph.
    """
```

## create_induction_vars
```
    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.
        """
```

## initialize_wave_constraints
This sets `wave_constraint.wave_id` to the `thread_id` of the matching
`workgroup_constraint` and additionally divides `thread_id_0` by `threads_per_wave`.

```
    def initialize_wave_constraints(self, trace: CapturedTrace) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.
        """
```

## initialize_symbolic_constraints
```
    def initialize_symbolic_constraints(self, trace: CapturedTrace) -> None:
        """
        For each symbolic constraint, create new constraints for the
        related symbolic values with appropriate substitutions.
        """
```

## initialize_workgroup_constraints
```
    def initialize_workgroup_constraints(self, trace: CapturedTrace) -> None:
        """
        For kernels that distribute more than three dimensions among workgroups,
        we need to update the workgroup constraints for dimensions >= 2
        with the appropriate workgroup index.
        """
```

## finalize_indices
```
    idxc: IndexingContext
    def finalize_indices():
        idxc.finalize()
```

## substitute_vector_shapes
```
    idxc: IndexingContext
    def substitute_vector_shapes():
        self.hardware_constraints[0].subs_vector_shapes(idxc.subs)
```

## infer_types
```
    for node in all_nodes:
        custom = get_custom(node)
        if isinstance(custom, NestedRegionOp):
            infer_types(trace.region_graph.subgraphs[custom.subgraph_name])
        custom.infer_type() # defined in wave_ops classes
```

## promote_placeholders
```
    # For each node touching shared memory

    def promote_node(
        node: Read | Write,
        last_write_to_shared: fx.Node,
        address_space: IndexSymbol,
        constraints: list[Constraint],
    ):
        """Promotes the given operand in the provided graph
        to the specified address space.

        The process of promotion involves allocating memory
        in the new address space and writing to the new
        memory location and subsequent uses reading from there.
        """
```

## set_node_indices
```
    def set_node_indices(trace: CapturedTrace, constraints: list[Constraint]):
        mma_index = get_mma_dimensional_mapping(trace, get_hardware_constraint(constraints))
        trace.walk(partial(set_thread_independent_index, constraints))
        set_thread_dependent_index(constraints, mma_index, trace)
        set_derived_index(trace)
        resolve_thread_shapes(trace, constraints)
        verify_nodes(trace, constraints)
```

## expand_graph
```
    def expand_graph(
        trace: CapturedTrace,
        constraints: Sequence[Constraint],
    ):
        """
        Create a graph that represents the expanded version of the wave function.
        The constraints are used to determine how the graph should be expanded.
        The expansion does a DFS starting at the leaf nodes and expanding them
        to the root of the graph.
        """
```

## set_post_expansion_indices
```
    def set_post_expansion_indices(trace: CapturedTrace, constraints: list[Constraint]):
        """
        Add offsets to the indices based on the expanded dims.
        """
```

## remove_chained_getresult
```
    def remove_chained_getresult(trace: CapturedTrace):
        while removable_nodes := trace.walk(is_chained_getresult):
            for node in removable_nodes:
                get_custom(node).replace_all_uses_with(get_custom(node).value)
                get_custom(node).graph.erase_node(node)
```


# Explanations

The vector_shapes attribute, defined HardwareConstraint is propagated to
individual operator nodes.

It specifies the shape of the data processed by a single thread/wave.
The expansion pass uses the vector_shapes to determine how many times a node
should be expanded. In essence, it tells to the system how to break down the
computation into chunks of data that has the vector_shape sizes.

While WorkgroupConstraint and TilingConstraint define how the overall problem is
broken down into workgroups and tiles. vector_shapes defines how a single
thread/wave's work is further vectorized within that tile.

It can be inferred from `tkw.mma` nodes but must be specified in the absence of
such nodes.

If both `tkw.mma` and `hw_cstr.vector_shape` are present, the `hw_cstr.vector_shape`
is used to update the inferred value.

During set_node_indices, the first step is to determine whether
`get_mma_dimensional_mapping` returns null or not:
- if there is even a single `tkw.mma` operation, the computation shifts into wave
mode (the primary mode supported), with vector_shape inferred. Indices are set
using `set_thread_dependent_index_from_mma` which starts propagating indices
from `populate_mma_source_indices`. `vector_shapes` specified in the HW constraints
are used to update the MMA op `vector_shapes` in `get_mma_dimensional_mapping`.
- otherwise `vector_shapes` is not inferred and must be specified in `hw_cstrs`.
index propagation is done via `set_thread_dependent_index_from_read_write` and
starts from `populate_read_write_source_indices`.



SIMT programming model:

1. set static symbols: `M = 128`, etc
2. set `tkw.WorkgroupConstraint(M, BLOCK_M, 0)`: each WG0 has size BLOCK_M and
the total number of blocks is `ceil(M / BLOCK_M)` (unless explictly overriden by
`WG.iters`)
3.




# Stuff derived, in what order and from what information

The entry point is in the `wave.py` file at the `LaunchableWave` class.

1. Per-kernel Workgroup and Grid size

`self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]`
which either:
  a. returns the `self.iters` IndexExpr if specified (override mode), or
  b. computes the `ceil(problem_size / WG tile_size)` IndexExpr for each primary
     constraint.
What happens when something is missing is unspecified atm.
TODO: check whether to enforce that all dims must be specified.
But this information is only used at the in `def _trace(self) -> CapturedTrace:`...

Note 1: in CompiledContext, a Grid object is created with a `Grid.symbolic_shape`
but it must fully resolve statically to static `Grid.dims` or fail. This happens
in `tracing.py`.

Note 2: Within the CompiledContext, an IndexingContext is pushed. This is where
symbolic dimensions and constraints start to be managed and, ultimately, resolved
to concrete values.

Note 3: Also in `tracing.py`, we see `CompiledContext.handle_xxx` operations that
feel like early versions of codegen/handlers.py `handle_xxx` variants.

Later, grid_types.dim is set up in `infer_grid_shape` and uses default `[1, 1, 1]`
unless overridden. This is also subject to some logic where `WG2` captures all
the `WGk, k >= 2` cases.

2. `initialize_iter_args`/`create_induction_vars`: tkw.reduction contains IterArg
nodes (apparently fx requires special handling for iter args). Then,
`create_induction_vars` adds a `tkl.IndexSymbol("$ARG" + name)` to each
`tkw.reduction` and connects it to `tiling_constraint.induction_var`.

Note 1: Given how `create_induction_vars`, a single TilingConstraint cannot be
used with multiple `tkw.reduction` on the same symbol.
```
tiling_constraint.induction_var = self.induction_vars[custom]
```

3. `initialize_wave_constraints` / `initialize_thread_constraints`: sets the
`wave_id` (resp. `thread_id`) on the Wave/ThreadConstraints

4. `initialize_reductions`: sets the `tkw.reduction` op count to the matching
`TilingConstraint.count` (which may or may not be overridden by
`TilingConstraints.iters`).

5. `verify_global_constraints` which must occur before redundant symbolic
constraint are initialized and break initial invariants.

6. `initialize_symbolic_constraints`: sets a bunch of IndexingContext substitutions
based on symbolic constraints. Seems this is just a 1-1 mapping to simplify

7. `initialize_workgroup_constraints`: update `WGk, k>= 2` using linearized /
delinearized WG indices.

8. `substitute_vector_shapes`: substitute `IndexingContext.subs` into all
vector_shapes in the HardwareConstraint.

9. `infer_types`: traverse the trace and calls `wave_ops.infer_type` for each op
type:
a. for ReadOp this is either the `self.mapping.output_shape` if this override is
specified or the `self.memory_type.symbolic_shape` otherwise.
b. for WriteOp this is either the `self.mapping.input_shape` if this override is
specified or the `self.memory_type.symbolic_shape` otherwise.
c. ExtractOp is weird and has implicit broadcast flavors
d. others behave as expected

10. **finally** `set_node_indices` which:
a. if `get_mma_dimensional_mapping` to determine whether we propagate from mma or
from read/write
b. `set_thread_independent_index`: sets wave_ops.index using
`WorkgroupConstraints.apply`, `WaveConstraints.apply` and `TilingConstraints.apply`.
Note 1: after this step, `read.index` may resemble: `$WG0*BLOCK_M + ITERATIONS_PER_WAVE_M*floor($T0/64)`
Note 2: "wave" is not considered "thread".
c. `set_thread_dependent_index_from_mma`/`set_thread_dependent_index_from_read_write`
depending on a.
  i. `populate_read_write_source_indices` uses `elements_per_thread` for the
  contiguous dimension but otherwise forces the other dimensions to be 1.
  `elements_per_thread` is indeed 1-D vector size of most minor load / store.
  ii. stride is computed from HardwareConstraint.vector_shape
d. set_derived_index
e. resolve_thread_shapes
f. verify_nodes



Expansion unrolling is determined by:
```
dim_scaling[constraint.dim] = tile_size // distribution_factor // vector_size
```


### Def-Use

`vector_shapes` as specified in HardwareConstraint is used (in chronological
occurrence location):
1. on the user side in HardwareConstraints
1. in `substitute_vector_shapes` where they are specialized in-place and must
become static
1. index_sequence_analysis:
a. `populate_read_write_source_indices`: return exactly
`hardware_constraint.vector_shapes`, so it never even tries to infer anything
```
[(node, index, hardware_constraint.vector_shapes)]
```
b. `get_mma_dimensional_mapping`: tkw.mma op shapes overrides:
```
custom.vector_shapes.update(hardware_constraint.vector_shapes)
```
Such overrides are used to specify batch dimensions that should have vector_size
== 0.
c. verify_nodes is a catchall for setting vector_shapes for nodes that haven't
been set
`# If vector_shapes is not set, see if it can be derived from the hardware constraints.`
This also relates to the derived `CustomOp.vector_shapes`
Note 1: it may be possible to incorrectly override load-bearing dimensions and
this should be hardened.

Later, per-node vector_shapes is used as follows:
1. in `wave/utils.py`: `add_reshape_if_needed / is_reshape_needed`,
`get_hardware_vector_size`,
`get_hardware_vector_map` via custom.vector_shapes.
1. in expansion unrolling, custom.vector_shapes determines unrolling:
```
dim_scaling[constraint.dim] = tile_size // distribution_factor // vector_size
```


### Indexing

1. workgroup-indexing:
