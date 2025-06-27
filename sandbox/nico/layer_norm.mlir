module @module {
  func.func @main(%arg0: !torch.vtensor<[16,1536,576],f32>, %arg1: !torch.vtensor<[576],f32>, %arg2: !torch.vtensor<[576],f32>) -> !torch.vtensor<[16,1536,576],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int576 = torch.constant.int 576
    %0 = torch.prim.ListConstruct %int576 : (!torch.int) -> !torch.list<int>
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %true = torch.constant.bool true
    %1 = torch.aten.layer_norm %arg0, %0, %arg1, %arg2, %float1.000000e-05, %true : !torch.vtensor<[16,1536,576],f32>, !torch.list<int>, !torch.vtensor<[576],f32>, !torch.vtensor<[576],f32>, !torch.float, !torch.bool -> !torch.vtensor<[16,1536,576],f32>
    return %1 : !torch.vtensor<[16,1536,576],f32>
  }
}
