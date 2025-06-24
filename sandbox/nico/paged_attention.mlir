module @module {
  func.func @main(%arg0: !torch.vtensor<[4,16,1,64],f32>, %arg1: !torch.vtensor<[2,?,16,64],f32>, %arg2: !torch.vtensor<[4,?],si64>, %arg3: !torch.vtensor<[4,?],si64>) -> !torch.vtensor<[4,16,1,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (2, s0, 16, 64)> : !torch.vtensor<[2,?,16,64],f32>
    torch.bind_symbolic_shape %arg2, [%1], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %arg3, [%1], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int0 = torch.constant.int 0
    %int0_0 = torch.constant.int 0
    %2 = torch.aten.select.int %arg1, %int0, %int0_0 : !torch.vtensor<[2,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,16,64],f32>
    torch.bind_symbolic_shape %2, [%0], affine_map<()[s0] -> (s0, 16, 64)> : !torch.vtensor<[?,16,64],f32>
    %int0_1 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %3 = torch.aten.select.int %arg1, %int0_1, %int1 : !torch.vtensor<[2,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,16,64],f32>
    torch.bind_symbolic_shape %3, [%0], affine_map<()[s0] -> (s0, 16, 64)> : !torch.vtensor<[?,16,64],f32>
    %4 = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[4,?],si64>) -> !torch.list<optional<vtensor>>
    %5 = torch.aten.index.Tensor %2, %4 : !torch.vtensor<[?,16,64],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[4,?,16,64],f32>
    torch.bind_symbolic_shape %5, [%1], affine_map<()[s0] -> (4, s0, 16, 64)> : !torch.vtensor<[4,?,16,64],f32>
    %6 = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[4,?],si64>) -> !torch.list<optional<vtensor>>
    %7 = torch.aten.index.Tensor %3, %6 : !torch.vtensor<[?,16,64],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[4,?,16,64],f32>
    torch.bind_symbolic_shape %7, [%1], affine_map<()[s0] -> (4, s0, 16, 64)> : !torch.vtensor<[4,?,16,64],f32>
    %int1_2 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %8 = torch.aten.transpose.int %5, %int1_2, %int2 : !torch.vtensor<[4,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,16,?,64],f32>
    torch.bind_symbolic_shape %8, [%1], affine_map<()[s0] -> (4, 16, s0, 64)> : !torch.vtensor<[4,16,?,64],f32>
    %int1_3 = torch.constant.int 1
    %int2_4 = torch.constant.int 2
    %9 = torch.aten.transpose.int %7, %int1_3, %int2_4 : !torch.vtensor<[4,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,16,?,64],f32>
    torch.bind_symbolic_shape %9, [%1], affine_map<()[s0] -> (4, 16, s0, 64)> : !torch.vtensor<[4,16,?,64],f32>
    %none = torch.constant.none
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %none_5 = torch.constant.none
    %false_6 = torch.constant.bool false
    %10 = torch.aten.scaled_dot_product_attention %arg0, %8, %9, %none, %float0.000000e00, %false, %none_5, %false_6 : !torch.vtensor<[4,16,1,64],f32>, !torch.vtensor<[4,16,?,64],f32>, !torch.vtensor<[4,16,?,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[4,16,1,64],f32>
    return %10 : !torch.vtensor<[4,16,1,64],f32>
  }
}
