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
