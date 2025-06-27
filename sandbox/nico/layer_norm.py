import torch
import torch.nn as nn

import iree.turbine.aot as aot

import numpy as np

class LayerNormForward(torch.nn.Module):
    def __init__(self, normalized_shape: tuple[int], eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    # def __init__(self, signature: LayerNormSignature):
    #     super().__init__()
    #     self.normalized_shape = signature.normalized_shape
    #     # TODO(azinenko): it is rather fishy that the signature includes values such as eps that are not actually part of the funciton signature...
    #     self.eps = signature.eps

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

dims = (16, 1536, 576)
num_normalized_dims = 1
normalized_shape = dims[-num_normalized_dims :]

model = LayerNormForward(tuple(normalized_shape))

input = torch.randn(dims)
weight = torch.ones(normalized_shape)
bias = torch.zeros(normalized_shape)
out = model(input, weight, bias)

np.save("layer_norm_input.npy", input.numpy())
np.save("layer_norm_weight.npy", weight.numpy())
np.save("layer_norm_bias.npy", bias.numpy())
np.save("layer_norm_out.npy", out.numpy())

dyn_seq_len = torch.export.Dim("seq_len")
exported = aot.export(
    model,
    args=(input, weight, bias),
    dynamic_shapes={
        # "input": {},
        # "weight": {2: dyn_seq_len},
        # "bias": {2: dyn_seq_len},
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
python layer_norm.py > layer_norm.mlir

### Lower to Linalg IR
python layer_norm.py > layer_norm.mlir && \
iree-compile layer_norm.mlir --compile-to=input

# Alternatively iree-compile layer_norm.mlir --compile-to=flow
# Alternatively iree-compile layer_norm.mlir --compile-to=stream
# Alternatively iree-compile layer_norm.mlir --compile-to=executable-sources --iree-hal-target-device=hip --iree-hip-target=gfx942
# Alternatively iree-compile layer_norm.mlir --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942

### Compile run exectute on CPU
python layer_norm.py > layer_norm.mlir && \
iree-compile layer_norm.mlir --iree-hal-target-backends=llvm-cpu -o layer_norm.vmfb --iree-llvmcpu-target-cpu=host && \
iree-run-module --module=layer_norm.vmfb --device=local-task --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy

### Print stuff
cmake_build_iree && load_iree && \
iree-compile layer_norm.mlir --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942 --mlir-print-ir-after-all
"""
