import torch
import torch.nn as nn

import iree.turbine.aot as aot

import numpy as np

# KVCache: [partition, pages, num_heads, head_dim]
# k_indices: bs, seq_len
# v_indices: bs, seq_len
#
# k_read = bs, seq_len, num_heads, head_dim
# v_read = bs, seq_len, num_heads, head_dim

# bs, num_heads, seq_len, head_dim

class PagedAttention(nn.Module):
    def forward(self, q, kv_cache, k_indices, v_indices):
        k_cache = kv_cache[0, ...]
        v_cache = kv_cache[1, ...]

        k_read = k_cache[k_indices]
        v_read = v_cache[v_indices]

        k = k_read.transpose(1, 2)
        v = v_read.transpose(1, 2)

        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


model = PagedAttention()

bs = 4
num_heads = 16
seq_len = 1024
head_dim = 64

# This is decode paged_attention with q seq_len == 1.
q = torch.randn((bs, num_heads, 1, head_dim))
kv_cache = torch.randn((2, 4096, num_heads, head_dim))
k_indices = torch.randint(0, 4096, (bs, seq_len))
v_indices = torch.randint(0, 4096, (bs, seq_len))

out = model(q, kv_cache, k_indices, v_indices)

np.save("paged_attention_q.npy", q.numpy())
np.save("paged_attention_kv_cache.npy", kv_cache.numpy())
np.save("paged_attention_k_indices.npy", k_indices.numpy())
np.save("paged_attention_v_indices.npy", v_indices.numpy())
np.save("paged_attention_out.npy", out.numpy())

cache_size = torch.export.Dim("cache_size")
dyn_seq_len = torch.export.Dim("seq_len")

exported = aot.export(
    model,
    args=(q, kv_cache, k_indices, v_indices),
    dynamic_shapes={
        "q": {},
        "kv_cache": {1: cache_size},
        "k_indices": {1: dyn_seq_len},
        "v_indices": {1: dyn_seq_len},
    },
    import_symbolic_shape_expressions=True,
)

# Note: in iree/turbine/aot/exporter.py :: l 317 -> print(exported_program) lets us dump the fx graph
# fx graph has dependent types with named symbols + range constraints
# We / IREE exports that info in the IR with `import_symbolic_shape_expressions=True`
#   -> used in IREE for divisibility and range bounds
# Torch doc is here: https://docs.pytorch.org/docs/stable/export.html
exported.print_readable()

"""
### Export ATen IR
python paged_attention.py > paged_attention.mlir

### Lower to Linalg IR
python paged_attention.py > paged_attention.mlir && \
iree-compile paged_attention.mlir --compile-to=input

### Compile run exectute on CPU
python paged_attention.py > paged_attention.mlir && \
iree-compile paged_attention.mlir --iree-hal-target-backends=llvm-cpu -o paged_attention.vmfb --iree-llvmcpu-target-cpu=host && \
iree-run-module --module=paged_attention.vmfb --device=local-task --input=@paged_attention_q.npy --input=@paged_attention_kv_cache.npy --input=@paged_attention_k_indices.npy --input=@paged_attention_v_indices.npy --expected_output=@paged_attention_out.npy

# Note: Yield operand #1 is not equivalent to the corresponding iter bbArg

### Print stuff
cmake_build_iree && load_iree && \
iree-compile paged_attention.mlir --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942 --mlir-print-ir-after-all
"""
