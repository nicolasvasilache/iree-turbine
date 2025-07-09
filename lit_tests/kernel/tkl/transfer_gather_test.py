# RUN: python %s | FileCheck %s

import iree.turbine.kernel.lang as tkl
from iree.turbine.utils.utils import compilation_test_harness, run

B, D, M, K = tkl.sym.B, tkl.sym.D, tkl.sym.M, tkl.sym.K
BLOCK_B, BLOCK_D = tkl.sym.BLOCK_B, tkl.sym.BLOCK_D
# An encoding for dynamic dimensions (any non-int would do really)
DYN = tkl.sym.DYNDIM

def test_read_indirect_2d(
    A_idx: tkl.InputBuffer[B, tkl.index],
    A: tkl.InputBuffer[M, K, tkl.f32],
    O: tkl.OutputBuffer[B, tkl.f32],
) -> None:
    @tkl.for_loop(0, B, BLOCK_B, init_args=[])
    def body(b, _) -> None:
      # Note: this is always inbounds only when B % BLOCK_B == 0
      a_idx = tkl.load(A_idx, (b, ), (BLOCK_B, ))
      # Note: this is always inbounds only when B % BLOCK_B == 0
      a = tkl.transfer_gather(A, (a_idx, 0), (BLOCK_B, K))
      acc = tkl.constant((BLOCK_B, ), tkl.f32, 0.0)
      s = tkl.sum(a, axis=1, acc=acc)  # sum over K dimension
      # Note: this is alway inbounds only when B % BLOCK_B == 0
      tkl.store(O, (b,), s)
    return

#   CHECK-LABEL: func.func @test_read_indirect_2d
#         CHECK: scf.for
#         CHECK:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<16xindex, strided<[1], offset: ?>>, vector<4xindex>
# CHECK-COUNT-4:   vector.extract {{.*}} : index from vector<4xindex>
# CHECK-COUNT-4:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<33x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
# CHECK-COUNT-4:   vector.insert {{.*}} : vector<64xf32> into vector<4x64xf32>
#         CHECK:   vector.multi_reduction <add>, {{.*}} [1] : vector<4x64xf32> to vector<4xf32>
#         CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<16xf32, strided<[1], offset: ?>>
@run
def test_transfer_gather_sss() -> None:     
    print(
        compilation_test_harness(
            substitutions={B: 16, M: 33, K: 64, BLOCK_B: 4,}, 
            function=test_read_indirect_2d)
    )

#   CHECK-LABEL: func.func @test_read_indirect_2d({{.*}}!stream.binding, {{.*}}!stream.binding, {{.*}}!stream.binding, {{.*}}index) {
#         CHECK: scf.for
#         CHECK:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<?xindex, strided<[1], offset: ?>>, vector<4xindex>
# CHECK-COUNT-4:   vector.extract {{.*}} : index from vector<4xindex>
# CHECK-COUNT-4:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<33x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
# CHECK-COUNT-4:   vector.insert {{.*}} : vector<64xf32> into vector<4x64xf32>
#         CHECK:   vector.multi_reduction <add>, {{.*}} [1] : vector<4x64xf32> to vector<4xf32>
#         CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<?xf32, strided<[1], offset: ?>>
@run
def test_transfer_gather_dss() -> None:
    # Note: B_dim = torch.export.Dim("B", min=1, max=2 ** 9)
    print(
        compilation_test_harness(
            substitutions={B: DYN, M: 33, K: 64, BLOCK_B: 4,}, 
            function=test_read_indirect_2d
        )
    )

#   CHECK-LABEL: func.func @test_read_indirect_2d({{.*}}!stream.binding, {{.*}}!stream.binding, {{.*}}!stream.binding, {{.*}}index, {{.*}}index) {
#         CHECK: scf.for
#         CHECK:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<?xindex, strided<[1], offset: ?>>, vector<4xindex>
# CHECK-COUNT-4:   vector.extract {{.*}} : index from vector<4xindex>
# CHECK-COUNT-4:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<?x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
# CHECK-COUNT-4:   vector.insert {{.*}} : vector<64xf32> into vector<4x64xf32>
#         CHECK:   vector.multi_reduction <add>, {{.*}} [1] : vector<4x64xf32> to vector<4xf32>
#         CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<?xf32, strided<[1], offset: ?>>
@run
def test_transfer_gather_dds() -> None:
    # Note: K is used for a vector size so it must resolve statically. 
    # If we want a dynamic K, we need to introduce BLOCK_K and rewrite the transfer_gather.
    print(
        compilation_test_harness(
            substitutions={B: DYN, M: DYN, K: 64, BLOCK_B: 4,}, 
            function=test_read_indirect_2d
        )
    )


def test_read_indirect_3d(
    A_idx: tkl.InputBuffer[B, tkl.index],
    A: tkl.InputBuffer[D, M, K, tkl.f32],
    O: tkl.OutputBuffer[B, D, tkl.f32],
) -> None:
    @tkl.for_loop(0, B, BLOCK_B, init_args=[])
    def body(b, _) -> None:
        # Note: this is always inbounds only when B % BLOCK_B == 0
        a_idx = tkl.load(A_idx, (b, ), (BLOCK_B, ))
        @tkl.for_loop(0, D, BLOCK_D, init_args=[])
        def body(d, _) -> None:
            # Note: this is always inbounds only when B % BLOCK_B == 0
            a = tkl.transfer_gather(A, (d, a_idx, 0), (BLOCK_D, BLOCK_B, K))
            acc = tkl.constant((BLOCK_D, BLOCK_B, ), tkl.f32, 0.0)
            s = tkl.sum(a, axis=2, acc=acc)  # sum over K dimension
            s = tkl.transpose(s, permutation=(1, 0))
            # Note: this is alway inbounds only when B % BLOCK_B == 0
            tkl.store(O, (b, d), s)
    return

#    CHECK-LABEL: func.func @test_read_indirect
#          CHECK: scf.for
#          CHECK:   vector.transfer_read {{.*}} {in_bounds = [true]} : memref<16xindex, strided<[1], offset: ?>>, vector<4xindex>
#          CHECK:   scf.for
#  CHECK-COUNT-4:     vector.extract {{.*}} : index from vector<4xindex>
# CHECK-COUNT-32:     vector.transfer_read {{.*}} {in_bounds = [true]} : memref<32x33x64xf32, strided<[2112, 64, 1], offset: ?>>, vector<64xf32>
# CHECK-COUNT-32:     vector.insert {{.*}} : vector<64xf32> into vector<8x4x64xf32>
#          CHECK:     vector.multi_reduction <add>, {{.*}} [2] : vector<8x4x64xf32> to vector<8x4xf32>
#          CHECK:     vector.transpose {{.*}} [1, 0] : vector<8x4xf32> to vector<4x8xf32>
#          CHECK:     vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, memref<16x32xf32, strided<[32, 1], offset: ?>>
@run
def test_transfer_gather_ssss() -> None:     
    print(
        compilation_test_harness(
            substitutions={B: 16, D: 32, M: 33, K: 64, BLOCK_B: 4, BLOCK_D: 8,}, 
            function=test_read_indirect_3d)
    )