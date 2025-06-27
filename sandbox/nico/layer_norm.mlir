module @module {
  func.func @main(%arg0: !torch.vtensor<[1,16,1536,576],f16>, %arg1: !torch.vtensor<[576],f16>, %arg2: !torch.vtensor<[576],f16>) -> !torch.vtensor<[1,16,1536,576],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int6 = torch.constant.int 6
    %0 = torch.prims.convert_element_type %arg0, %int6 : !torch.vtensor<[1,16,1536,576],f16>, !torch.int -> !torch.vtensor<[1,16,1536,576],f32>
    %int3 = torch.constant.int 3
    %1 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %result0, %result1 = torch.aten.var_mean.correction %0, %1, %int0, %true : !torch.vtensor<[1,16,1536,576],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[1,16,1536,1],f32>, !torch.vtensor<[1,16,1536,1],f32>
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %int1 = torch.constant.int 1
    %2 = torch.aten.add.Scalar %result0, %float1.000000e-05, %int1 : !torch.vtensor<[1,16,1536,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,16,1536,1],f32>
    %3 = torch.aten.rsqrt %2 : !torch.vtensor<[1,16,1536,1],f32> -> !torch.vtensor<[1,16,1536,1],f32>
    %int1_0 = torch.constant.int 1
    %4 = torch.aten.sub.Tensor %arg0, %result1, %int1_0 : !torch.vtensor<[1,16,1536,576],f16>, !torch.vtensor<[1,16,1536,1],f32>, !torch.int -> !torch.vtensor<[1,16,1536,576],f32>
    %5 = torch.aten.mul.Tensor %4, %3 : !torch.vtensor<[1,16,1536,576],f32>, !torch.vtensor<[1,16,1536,1],f32> -> !torch.vtensor<[1,16,1536,576],f32>
    %6 = torch.aten.mul.Tensor %5, %arg1 : !torch.vtensor<[1,16,1536,576],f32>, !torch.vtensor<[576],f16> -> !torch.vtensor<[1,16,1536,576],f32>
    %int1_1 = torch.constant.int 1
    %7 = torch.aten.add.Tensor %6, %arg2, %int1_1 : !torch.vtensor<[1,16,1536,576],f32>, !torch.vtensor<[576],f16>, !torch.int -> !torch.vtensor<[1,16,1536,576],f32>
    %int5 = torch.constant.int 5
    %8 = torch.prims.convert_element_type %7, %int5 : !torch.vtensor<[1,16,1536,576],f32>, !torch.int -> !torch.vtensor<[1,16,1536,576],f16>
    return %8 : !torch.vtensor<[1,16,1536,576],f16>
  }
}
