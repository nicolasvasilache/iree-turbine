# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl

from iree.compiler import ir
from iree.compiler.dialects.transform import interpreter as transform_interpreter

# B: batch size
# H: num heads
# M: source sequence length
# N: target sequence length
# K1: embedding dimension of key and query
# K2: embedding dimension of value
vB, vH, vM, vN, vK1, vK2 = 4, 96, 32, 64, 128, 256
B, H, M, N, K1, K2 = tkl.sym.B, tkl.sym.H, tkl.sym.M, tkl.sym.N, tkl.sym.K1, tkl.sym.K2
BLOCK_M, BLOCK_N, BLOCK_K2 = tkl.sym.BLOCK_M, tkl.sym.BLOCK_N, tkl.sym.BLOCK_K2

vLOG2E = 1.44269504089

TRANSFORM_MODULE = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["vector.contract"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    %mma_attr = transform.param.constant #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16> -> !transform.any_param
    %2 = transform.iree.infer_and_attach_vector_contract_layout mma_attr(%mma_attr) to %producer
      : (!transform.any_param, !transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.propagate_vector_distribution %func workgroup_size = [64, 1, 1] subgroup_size = 64
      : !transform.any_op
    transform.yield
  }
}
"""
class Test(unittest.TestCase):
    def testFusedAttention(self):

        def apply_transform(module: ir.Operation):
            with module.context:
                transform_module = ir.Module.parse(TRANSFORM_MODULE)
                transform_interpreter.apply_named_sequence(
                    module,
                    transform_module.body.operations[0],
                    transform_module,
                )

        @tk.gen.thread(B, H, transform=apply_transform)
        def fused_attention(
            Q: tkl.InputBuffer[B, H, M, K1, tkl.f16],
            K: tkl.InputBuffer[B, H, K2, K1, tkl.f16],
            V: tkl.InputBuffer[B, H, K2, N, tkl.f16],
            O: tkl.OutputBuffer[B, H, M, N, tkl.f16],
        ):
            # TODO: Naive mapping for now (1x1 along bxh per workgroup).
            # The goal is to use scf.forall mapping in the future and avoid having to think
            # about mod and div mapping.
            b, h = tkl.program_id(0), tkl.program_id(1)

            k1_val_f = tkl.constant((), tkl.f32, float(vK1))
            one = tkl.constant((), tkl.f32, 1.0)
            dk_sqrt = tkl.sqrt(one / k1_val_f)
            # Multiply by scaling in f32 is more accurate.
            # This could also be done ahead of time outside of the kernel.
            qkT_scaling = tkl.broadcast_in_dim(dk_sqrt, (BLOCK_M, BLOCK_K2, ), (0, 1, ))

            # This is really a forall that later should get mapped.
            @tkl.for_loop(0, M, BLOCK_M, init_args=[])
            def body(m):
                q = tkl.load(Q, (b, h, m, 0), (BLOCK_M, K1))

                # This is really a forall that later should get mapped.
                @tkl.for_loop(0, N, BLOCK_N, init_args=[])
                def body(n):
                    acc_init = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 0.0)
                    partial_max_init = tkl.constant((BLOCK_M,), tkl.f32, -1e9)
                    partial_sum_init = tkl.constant((BLOCK_M,), tkl.f32, 0.0)

                    @tkl.for_loop(
                        0, K2, BLOCK_K2, init_args=[partial_max_init, partial_sum_init, acc_init]
                    )
                    def inner_body(k2, partial_max, partial_sum, acc):
                        k = tkl.load(K, (b, h, k2, 0), (BLOCK_K2, K1))
                        # TODO: are there better ways to do this (e.g. verified tkl.transpose(k, (K1, BLOCK_K2)) ?
                        kT = tkl.transpose(k, (1, 0)) # : (K1, BLOCK_K2)

                        # TODO: (suggested by Kunwar) use the tkl.mma operation ot set the layout attribute
                        # in a controlled fashion. This will let us activate vector distribution easily on GPU.
                        #
                        # (BLOCK_M, BLOCK_K2).T <- (K1, BLOCK_K2).T * (BLOCK_M, K1).T
                        qkT = tkl.constant((BLOCK_M, BLOCK_K2), tkl.f32, 0.0)
                        q_d = tkl.transpose(q, (1, 0))
                        kT_d = tkl.transpose(kT, (1, 0))
                        qkT_d = tkl.transpose(qkT, (1, 0))
                        qkT_d = tkl.dot(kT_d, q_d, qkT_d) # : (BLOCK_M, BLOCK_K2)
                        qkT = tkl.transpose(qkT_d, (1, 0))
                        qkT = qkT * qkT_scaling

                        m_j = tkl.max(qkT, axis=1, acc=partial_max) # : (BLOCK_M)
                        e_delta_max = tkl.exp(partial_max - m_j) # : (BLOCK_M)

                        # TODO: are there better ways to do this (e.g. tkl.broadcast_to_typeof(partial_max)) ?
                        m_j_bcast = tkl.broadcast_in_dim(m_j, (BLOCK_M, BLOCK_K2), (1, )) # : (BLOCK_M, BLOCK_K2)
                        e_delta = tkl.exp(qkT - m_j_bcast) # : (BLOCK_M, BLOCK_K2)
                        e_init = e_delta_max * partial_sum # : (BLOCK_M)
                        d_j = tkl.sum(e_delta, axis=1, acc=e_init) # : (BLOCK_M)

                        # TODO: are there better ways to do this (e.g. tkl.broadcast_to_typeof(acc)) ?
                        e_delta_max = tkl.broadcast_in_dim(e_delta_max, (BLOCK_M, BLOCK_N), (1, )) # : (BLOCK_M, BLOCK_N)
                        acc = acc * e_delta_max # : (BLOCK_M, BLOCK_N)

                        # (BLOCK_M, BLOCK_N) <- (BLOCK_M, BLOCK_K2) * (BLOCK_K2, BLOCK_N)
                        imm_f16 = tkl.to_dtype(e_delta, tkl.f16) # : (BLOCK_M, BLOCK_K2)
                        v = tkl.load(V, (b, h, k2, n), (BLOCK_K2, BLOCK_N)) 
                        imm_f16_d = tkl.transpose(imm_f16, (1, 0))
                        v_d = tkl.transpose(v, (1, 0))
                        acc = tkl.transpose(acc, (1, 0))
                        acc = tkl.dot(v_d, imm_f16_d, acc) # : (BLOCK_M, BLOCK_N)
                        acc = tkl.transpose(acc, (1, 0))

                        return (m_j, d_j, acc)

                    max, sum, res = inner_body
                    one = tkl.constant((BLOCK_M,), tkl.f32, 1.0)
                    one_by_sum = one / sum
                    one_by_sum = tkl.broadcast_in_dim(one_by_sum, (BLOCK_M, BLOCK_N), (1, ))
                    result = one_by_sum * res
                    tkl.store(O, (b, h, m, n), result)

                    return
                
                return

        # B: batch size
        # H: num heads
        # M: source sequence length
        # N: target sequence length
        # K1: embedding dimension of key and query
        # K2: embedding dimension of value
        Q = torch.randn(vB, vH, vM, vK1).to(torch.float16).cuda()
        K = torch.randn(vB, vH, vK2, vK1).to(torch.float16).cuda()
        V = torch.randn(vB, vH, vK2, vN).to(torch.float16).cuda()
        O = torch.zeros(vB, vH, vM, vN).to(torch.float16).cuda()

        # TODO: (suggested by Kunwar) we could build a tk.gen.EagerLaunchContext to emit pytorch via tracing.
        # This would give us 2 intermediate testing points:
        #   1. Language traced -> PyTorch vs PyTorch reference implementation.
        #   2. Language traced -> MLIR vs Language traced -> PyTorch, potentially instruction by instruction.
        # This should be very useful for debugging and teaching purposes.
        vBLOCK_M, vBLOCK_N, vBLOCK_K2 = 32, 32, 32
        assert vM % vBLOCK_M == 0, "only divisible sizes supported for now"
        assert vN % vBLOCK_N == 0, "only divisible sizes supported for now"
        assert vK2 % vBLOCK_K2 == 0, "only divisible sizes supported for now"
        with tk.gen.TestLaunchContext(
            {
                BLOCK_M: min(vBLOCK_M, vM),
                BLOCK_N: min(vBLOCK_N, vN),
                BLOCK_K2: min(vBLOCK_K2, vK2),
            }
        ):
            # Wave equivalent code lives here:
            #   iree/turbine/kernel/wave/templates/vanilla_attention.py :: line 525
            fused_attention(Q, K, V, O)
            ref = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None
            )
            epsilon = torch.finfo(torch.float16).eps
            torch.testing.assert_close(O, ref, atol=epsilon, rtol=epsilon)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
