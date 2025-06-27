module @module {
  func.func @main(%arg0: !torch.vtensor<[4,16,1,64],f32>, %arg1: !torch.vtensor<[4,16,?,64],f32>, %arg2: !torch.vtensor<[4,16,?,64],f32>) -> !torch.vtensor<[4,16,1,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (4, 16, s0, 64)> : !torch.vtensor<[4,16,?,64],f32>
    torch.bind_symbolic_shape %arg2, [%0], affine_map<()[s0] -> (4, 16, s0, 64)> : !torch.vtensor<[4,16,?,64],f32>
    %none = torch.constant.none
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %true = torch.constant.bool true
    %none_0 = torch.constant.none
    %false = torch.constant.bool false
    %1 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2, %none, %float0.000000e00, %true, %none_0, %false : !torch.vtensor<[4,16,1,64],f32>, !torch.vtensor<[4,16,?,64],f32>, !torch.vtensor<[4,16,?,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[4,16,1,64],f32>
    return %1 : !torch.vtensor<[4,16,1,64],f32>
  }
}
