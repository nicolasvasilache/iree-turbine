# \[Internal\]\[Not Shared With AMD Yet\] TK Towards SOW

In the current state of TK, people have expressed difficulties getting to a first functional kernel on their own (i.e. without copy-pasting an existing known good kernel and modifying).

This document discusses some high-level properties of TK and proposes solutions that we may or may not choose to adopt.

This document is a mix of documentation, pain points and future items to consider splitting between SoW and our IP.

We can categorize proposed improvements along different axes.

## Programming model

### Wave and thread programming models

TK started as a block-level wave DSL to generate GEMMs and has grown a lot in both expressiveness and complexity. Today it mixes an explicit workgroup-level wave programming model and an implicit thread programming model. The implicit thread programming model is currently less fleshed out and tested.

Consider adding a ThreadConstraint and update the compiler to support it correctly.
This should be done in conjunction with improved verifier and debugging as well as extensive tutorial tests and documentation.

### Improve specification and documentation of iterators, memory accesses, tensor layout and tensor shapes

#### Kernel signature: iterators and underlying iteration domain

Consider the TKW kernel signature below:

```py
@tkw.wave(constraints)
def gemm(
    a: tkl.Memory[M, K, GLOBAL, tkl.f16], # a is a 2-D f16 tensor in global memory traversed by iterators: (M, K)
    b: tkl.Memory[N, K, GLOBAL, tkl.f16], # b is a 2-D f16 tensor in global memory traversed by iterators: (N, K)
    c: tkl.Memory[M, N, GLOBAL, tkl.f32], # c is a 2-D f32 tensor in global memory traversed by iterators: (M, N)
):
```

A first counter-intuitive realization is that in TKW `M`, `N` and `K` are iterators and are potentially *completely unrelated* to either the tensor shape, tensor layout or the memory accesses.

Instead what this signature conveys is that `a` (resp. `b`, `c`) is a 2-D tensor such that, in the `gemm` kernel:

- the first dimension is indexed by iterator `M` (resp. `N`, `M`)
- the second dimension is indexed by iterator `K` (resp. `K`, `N`)

This defines a kernel computed by a 3-D iteration domain `MxNxK` (i.e. a 3-loop nest) whose size is *symbolic and potentially unrelated* to the shape of tensors.

The signature, solely serves to declare:

1. The number of dimensions in the iteration space.
2. Which dimensions of which tensor are indexed by the same iterators

Suppose `a.shape[0]` and `c.shape[0]` are of different sizes, introducing a new symbol is *not the right way* to encode this in TK; the following signature would actually specify a 4-D iteration domain:

```py
@tkw.wave(constraints)
def gemm(
    a: tkl.Memory[M, K, GLOBAL, tkl.f16], # a is a 2-D f16 tensor in global memory traversed by iterators: (M, K)
    b: tkl.Memory[N, K, GLOBAL, tkl.f16], # b is a 2-D f16 tensor in global memory traversed by iterators: (N, K)
    c: tkl.Memory[M2, N, GLOBAL, tkl.f32], # c is a 2-D f32 tensor in global memory traversed by iterators: (M2, N)
):
```

##### Potential improvements

Consider an explicit iteration domain as a constraint, or adding a `@tkw.domain` \+ verifier that only the dimensions listed are used.
Consider proper error messages when unlisted dimensions are used, tied to an explanation that iterators \== iteration domain and are potentially unrelated to shapes.
Consider a syntax using `indexed_by=(...)` in the signature rather than a form that is too close to how one specifies tensor shapes in all other languages.

### Memory accesses

#### Default access patterns
Memory access patterns in `tkw.read` and `tkw.write` operations are by default
implicit (a.k.a. "identity mapping"). They load/store scalars or 1-D vectors, but
no higher-dimensional vectors.

The final indexing is derived in `iree/turbine/kernel/wave/codegen/read_write.py`
in `_create_vec_read_write` and is determined by:
1. the `elements_per_thread` passed to the operation, determines a 1-D MLIR
   vector type
1. the `input_shape` determined from `fxNode.type.symbolic_shape`
1. the `fxNode.index` expressions, separated into thread-dependent and
   thread-independent parts
1. an optional mask built from whether

 and can be confusing to reason about. The relationship between iterators, vector shapes, and actual memory access patterns needs clearer specification and documentation.


### Build a clear tutorial and incrementally introduce language features

Given the programming model is so powerful, TK would benefit from many simple examples to gradually building kernel author's intuition. Note that today, most of these cases are hard to write correctly in TK, while more complex examples using MMA operations, broadcasting, and gather/scatter optimizations are much better supported.

Consider adding step-by-step examples with/without transposes and various unrolling/mapping:

- 1-D, 2-D, 3-D copies
- 1-D, 2-D, 3-D elementwise
- 1-D, 2-D, 3-D shuffle reductions
- 1-D, 2-D, 3-D shuffle reductions fused with recomputations

Additionally, consider composing these examples with various load/store behaviors (contiguous, masked, gather/scatter), padding, masking and vector sizes.

Then it makes sense to look at the implications of MMA operations in GEMM and later attentions \+ all the KV-cached, masking and prefill forms.

### Grid Shape Resolution

### Redundancy between WaveConstraints and waves\_per\_block

The current design allows specifying wave distribution through both `WaveConstraint` and `waves_per_block` in `HardwareConstraint`. These two mechanisms can specify conflicting information about wave distribution. `waves_per_block` in `HardwareConstraint` is redundant when `WaveConstraint` is present. Consider:

- Removing `waves_per_block` from `HardwareConstraint`
- Making wave distribution specification explicit only through `WaveConstraint`
- Adding verification to ensure wave distribution is specified exactly once Document and verify the relationship between thread count, wave size, and wave distribution. Add clear examples of correct wave distribution specification.

### Mapping of iterators to workgroups, waves and threads

TK adopts a custom workgroup mapping model where dimension 0 and 1 are mapped to x and y, whereas dimensions 2-n are linearized/delinearized into z. TK adopts the simplest thread/warp mapping model where only 3 dimensions are supported and mapped to x, y, z and mapped 1-1 as multiples of `(64, 1, 1)`. In cases where the number of resources used for a particular kernel is

In MLIR upstream, generic infrastructure is available for most cases which streamlines the mapping to arbitrary number of dimensions for blocks, warpgroups, warps, threads and lanes (including with correct prediction if necessary).

Consider using the upstream infrastructure to use a well-tested infrastructure that supports most cases and consider extending it for cases not yet supported that are of interest.

Mapping specification would be similar to scf.forall mapping. Benefits of using this abstraction include:

1. Lower cognitive overhead of the mappings, what cases are supported and compatibility with known good MLIR abstractions.
2. Consistent linearization/delinearization behavior for all quantities (blocks, warpgroups, warps, threads, lanes).
3. More general mapping behavior easily allows implementing: a) better L2 behavior via block scheduling (e.g. in [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)), b) better indexing behavior with threads\_per\_workgroup of sizes `(16, a, b)` or `(32, c, d)` and c) future-proof predication support which extends to specialization (block, warp, thread).

### Debugging and verification

In TK it is currently very easy to write kernels where different waves/threads perform the same work multiple times.

TK could consider providing various debugging modes in which the IR is produced at different levels of abstraction. In particular, it seems if could be quite beneficial from a general comprehension and debugging perspective to emit an intermediate IR with mapped scf.forall on buffers as well as higher-order non-unrolled vector operations.

## Implicit vs explicit and conflicting behavior

### Workgroup, Wave, Thread, Tiling, Unrolling and Mapping for Broadcasting

One one hand, TK is explicit in specifying workgroup, wave and tiling constraints. It also explicitly expresses tkw.reduction loops. On the other hand, TK is implicit in specifying thread, unrolling and mapping to size 1 for broadcasting. This combination of explicit and implicit behaviors can be confusing and leads to either crashes \+ debugging the compiler or producing kernels with incorrect runtime behavior.

Consider making everything explicit and implementing verifiers to ensure each iterator is properly mapped and has a set of compatible constraints. Part of the verifier will be global to the kernel, part will be local to each operation.

### Implicit Broadcasting

TK uses implicit broadcasting rules in operations like `tkw.read` and binary ops needs. It can be challenging to obtain proper programs given that all the indexings and all the shapes are implicit in a TK program.

Consider:
\- making broadcasting explicit by requiring `tkw.broadcast`
\- support multi-dimensional vectors rather than the limited `1x1x1xN` shapes currently available

### Implicit Mapping to Size 1 Required to Satisfy Implicit Broadcasting Rules

### Index Propagation Inference

- Current dual-path (MMA vs read/write) propagation logic is complex
- Consider unifying the propagation strategy
- Make the choice of propagation strategy more explicit

### Vector Shape Inference

### Add TK Assertions to

## Global vs Local Behavior

### Vector shapes

### Reduction Handling
