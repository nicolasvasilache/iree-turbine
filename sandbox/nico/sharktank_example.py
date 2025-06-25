"""
shark-ai/docs/developer_guide.md @1:03:00

mostly used to export IR from a model

eager compilation goes through IREE, development compiler can be needed to modify that part.

pip install -e

from a model, export .irpa model file (like safetensor format but IREE friendly)
https://github.com/nod-ai/playbook/blob/main/HOWTO/Setup/chai.md

@1:16:00 command to export bith weights and model
python -m sharktank.examples.export_paged_llvm_v1 ...

Run with iree-opt-level=O3

SHARKTANK @mlir_kernel from MLIR asm string -> https://gist.github.com/Groverkss/f9e51f273abd01e72579b829079d6892

-> util.call goes to dispatch: no guarantee things stay fused. Fusion is reasonably predictable though.
"""

# RUN: TURBINE_DEBUG="log_level=DEBUG,runtime_trace_dir=./trace" python3 sharktank_gather.py

import torch

from sharktank.kernels.mlir_kernel import *

M = DynDim.M
K = StaticDim.K

S = Dtype.S
I64 = Dtype.I64


@mlir_kernel(
    inputs=(MLIRTensor[M, K, S], MLIRTensor[K, I64]),
    results=(MLIRTensor[M, S],),
)
def sharktank_gather(source, indices, result=None):
    mlir = """
module {
  util.func @{{kernel_name}}(%source: !source, %indices: !indices) -> !result {
    %c0 = arith.constant 0 : index
    %n_dim = tensor.dim %source, %c0 : !source
    %empty = tensor.empty(%n_dim) : !result
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]}
      ins(%indices : !indices)
      outs(%empty : !result) {
      ^bb0(%in: !indices_dtype, %o: !result_dtype):
        %n = arith.index_cast %in : !indices_dtype to index
        %m = linalg.index 1 : index
        %extracted = tensor.extract %source[%n, %m] : !source
        linalg.yield %extracted : !result_dtype
      } -> !result
      util.return %result : !result
  }
}
    """
    return MLIRSpec(mlir)

vM, vK = 128, 4
source = torch.randn((vM, vK), dtype=torch.float16) #, device="cuda:0")
indices = torch.randint(0, vM, (vK,)) # , device="cuda:0")
rocm_output = sharktank_gather(source, indices)

source = source.cpu()
indices = indices.cpu()
cpu_output = sharktank_gather(source, indices)

torch.testing.assert_close(rocm_output, cpu_output)