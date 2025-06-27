import torch
import torch.nn as nn

import iree.turbine.aot as aot

import numpy as np


class CausalAttention(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

model = CausalAttention()

bs = 4
num_heads = 16
seq_len = 1011
head_dim = 64

q_prefill = torch.randn((bs, num_heads, seq_len, head_dim))
q_decode = torch.randn((bs, num_heads, 1, head_dim))
q = q_decode
k = torch.randn((bs, num_heads, seq_len, head_dim))
v = torch.randn((bs, num_heads, seq_len, head_dim))
out = model(q, k, v)

np.save("causal_attention_q.npy", q.numpy())
np.save("causal_attention_k.npy", k.numpy())
np.save("causal_attention_v.npy", v.numpy())
np.save("causal_attention_out.npy", out.numpy())
dyn_seq_len = torch.export.Dim("seq_len")

exported = aot.export(
    model,
    args=(q, k, v),
    dynamic_shapes={
        # "q": {2: dyn_seq_len}, # prefill dynamic
        "q": {}, # causal or prefill static
        "k": {2: dyn_seq_len},
        "v": {2: dyn_seq_len},
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
python causal_attention.py > causal_attention.mlir

### Lower to Linalg IR
python causal_attention.py > causal_attention.mlir && \
iree-compile causal_attention.mlir --compile-to=input

# Alternatively iree-compile causal_attention.mlir --compile-to=flow
# Alternatively iree-compile causal_attention.mlir --compile-to=stream
# Alternatively iree-compile causal_attention.mlir --compile-to=executable-sources --iree-hal-target-device=hip --iree-hip-target=gfx942
# Alternatively iree-compile causal_attention.mlir --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942

### Compile run exectute on CPU
python causal_attention.py > causal_attention.mlir && \
iree-compile causal_attention.mlir --iree-hal-target-backends=llvm-cpu -o causal_attention.vmfb --iree-llvmcpu-target-cpu=host && \
iree-run-module --module=causal_attention.vmfb --device=local-task --input=@causal_attention_q.npy --input=@causal_attention_k.npy --input=@causal_attention_v.npy --expected_output=@causal_attention_out.npy

# Note: Yield operand #1 is not equivalent to the corresponding iter bbArg

### Print stuff
cmake_build_iree && load_iree && \
iree-compile paged_attention.mlir --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942 --mlir-print-ir-after-all
"""
