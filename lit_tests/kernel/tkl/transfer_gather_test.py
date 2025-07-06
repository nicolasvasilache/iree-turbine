# RUN: python %s | FileCheck %s

import iree.turbine.kernel.lang as tkl
from iree.turbine.utils.utils import compilation_test_harness, run

B, M, K, BLOCK_B = tkl.sym.B, tkl.sym.M, tkl.sym.K, tkl.sym.BLOCK_B

def test_read_indirect(
    A_idx: tkl.InputBuffer[B, tkl.index],
    A: tkl.InputBuffer[M, K, tkl.f32],
    O: tkl.OutputBuffer[B, tkl.f32],
) -> None:
    @tkl.for_loop(0, B, BLOCK_B, init_args=[])
    def body(b, _) -> None:
      a_idx = tkl.load(A_idx, (b, ), (BLOCK_B, ))
      a = tkl.transfer_gather(A, (a_idx, 0), (BLOCK_B, K))
      acc = tkl.constant((BLOCK_B, ), tkl.f32, 0.0)
      s = tkl.sum(a, axis=1, acc=acc)  # sum over K dimension
      tkl.store(O, (b,), s)
    return

#   CHECK-LABEL: func.func @test_read_indirect
#         CHECK: scf.for
#         CHECK:   vector.transfer_read %0[%arg3], %c0 {in_bounds = [true]} : memref<16xindex, strided<[1], offset: ?>>, vector<4xindex>
# CHECK-COUNT-4:   vector.extract {{.*}} : index from vector<4xindex>
# CHECK-COUNT-4:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<33x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
# CHECK-COUNT-4:   vector.insert {{.*}} : vector<64xf32> into vector<4x64xf32>
#         CHECK:   vector.multi_reduction <add>, {{.*}} [1] : vector<4x64xf32> to vector<4xf32>
#         CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<16xf32, strided<[1], offset: ?>>
@run
def test_transfer_gather():     
    print(compilation_test_harness(
        substitutions={B: 16, M: 33, K: 64, BLOCK_B: 4,}, function=test_read_indirect))
