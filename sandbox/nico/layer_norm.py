import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

import iree.turbine.aot as aot

import numpy as np

class LayerNormForward(torch.nn.Module):
    def __init__(self, normalized_shape: tuple[int], eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

dims = (1, 16, 1536, 576)
num_normalized_dims = 1
normalized_shape = dims[-num_normalized_dims :]

model = LayerNormForward(tuple(normalized_shape))

input = torch.randn(dims).cuda().to(dtype=torch.float16)
weight = torch.randn(normalized_shape).cuda().to(dtype=torch.float16)
bias = torch.randn(normalized_shape).cuda().to(dtype=torch.float16)

if profile:
    # Warmup
    for i in range(100):
        out = model(input, weight, bias)
    # Profile GPU execution
    with open('results.pytorch.txt', 'w') as f:
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                for i in range(1000):
                    out = model(input, weight, bias)

        # Print a summary table sorted by self CUDA time
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3), file=f)
else:
    out = model(input, weight, bias)


np.save("layer_norm_input.npy", input.cpu().numpy())
np.save("layer_norm_weight.npy", weight.cpu().numpy())
np.save("layer_norm_bias.npy", bias.cpu().numpy())
np.save("layer_norm_out.npy", out.cpu().numpy())

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
python layer_norm.py > layer_norm.mlir

### Print stuff
iree-compile layer_norm.mlir --compile-to=input
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 --compile-to=input --mlir-print-ir-after-all
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 --compile-to=executable-configurations --mlir-print-ir-after-all
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 --compile-to=hal --mlir-print-ir-after-all

# Alternatively iree-compile layer_norm.mlir --iree-opt-level=O3 --compile-to=flow
# Alternatively iree-compile layer_norm.mlir --iree-opt-level=O3 --compile-to=stream
# Alternatively iree-compile layer_norm.mlir --iree-opt-level=O3 --compile-to=executable-sources --iree-hal-target-device=hip --iree-hip-target=gfx942
# Alternatively iree-compile layer_norm.mlir --iree-opt-level=O3 --compile-to=hal --iree-hal-target-device=hip --iree-hip-target=gfx942

### Compile run execute on CPU
python layer_norm.py > layer_norm.mlir
iree-compile layer_norm.mlir --iree-hal-target-backends=llvm-cpu --iree-opt-level=O3 -o layer_norm.vmfb --iree-llvmcpu-target-cpu=host
iree-run-module --module=layer_norm.vmfb --device=local-task --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy

### Compile run execute and test correctness on GPU
python layer_norm.py > layer_norm.mlir
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 -o layer_norm.vmfb
iree-run-module --module=layer_norm.vmfb --device=hip://GPU-36613962-3937-6562-6639-616539393537  --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy

### Compile run execute on GPU with tracy (some overhead)
python layer_norm.py > layer_norm.mlir
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 -o layer_norm.vmfb
TRACY_NO_EXIT=1 iree-run-module --module=layer_norm.vmfb --device=hip://GPU-36613962-3937-6562-6639-616539393537  --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy
TRACY_NO_EXIT=1 iree-benchmark-module --module=layer_norm.vmfb --device=hip://GPU-36613962-3937-6562-6639-616539393537  --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy --function=main

iree-tracy-capture -o results.tracy

view tracy trace with https://tracy.nereid.pl/

### Compile run execute on GPU with rocprof
python layer_norm.py > layer_norm.mlir
iree-compile layer_norm.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-opt-level=O3 -o layer_norm.vmfb
/opt/rocm/bin/rocprof --hip-trace iree-run-module --module=layer_norm.vmfb --device=hip://GPU-36613962-3937-6562-6639-616539393537  --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy
/opt/rocm/bin/rocprof --hip-trace iree-benchmark-module --module=layer_norm.vmfb --device=hip://GPU-36613962-3937-6562-6639-616539393537  --input=@layer_norm_input.npy --input=@layer_norm_weight.npy --input=@layer_norm_bias.npy --expected_output=@layer_norm_out.npy --function=main

cat results.pytorch.txt && \
cat results.stats.csv | sed 's/,/ /g' | sed 's/"//g' | grep dispatch | awk '{print $1 " " $4/1000 "us" " " $2 "calls"}'

view json with chrome://tracing

### Reference
/opt/rocm/bin/MIOpenDriver layernormfp16 --input 1x16x1536x576 -i 100 -t 1 -m 1 -w 1

/opt/rocm/bin/rocprof --hip-trace /opt/rocm/bin/MIOpenDriver layernormfp16 --input 1x16x1536x576 -i 100 -t 1 -m 1 -w 1
cat results.stats.csv | sed 's/,/ /g' | sed 's/"//g' | grep -v AverageNs | awk '{print $1 " " $4/1000 "us" " " $2 "calls"}'
"""
