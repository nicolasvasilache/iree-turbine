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

class Test(unittest.TestCase):
    def testFusedAttention(self):
        @tk.gen.thread(B, H)
        def fused_attention(
            Q: tkl.InputBuffer[B, H, M, K1, tkl.f32],
            K: tkl.InputBuffer[B, H, K2, K1, tkl.f32],
            V: tkl.InputBuffer[B, H, K2, N, tkl.f32],
            O: tkl.OutputBuffer[B, H, M, N, tkl.f32],
        ):
            # TODO: Naive mapping for now (1x1 along bxh per workgroup).
            # The goal is to use scf.forall mapping in the future and avoid having to think
            # about mod and div mapping.
            b, h = tkl.program_id(0), tkl.program_id(1)

            # This is really a forall that later should get mapped.
            @tkl.for_loop(0, M, BLOCK_M, init_args=[])
            def body(m):
                q = tkl.load(Q, (b, h, m, 0), (BLOCK_M, K1))

                # TODO: how to splat a constant coming from a dim (i.e. K1; for now just use vK1)
                # k1_val = tkl.constant((BLOCK_M, K1, ), tkl.i32, K1)
                # k1_val_f = tkl.to_dtype(k1_val, tkl.f32)
                k1_val_f = tkl.constant((BLOCK_M, K1, ), tkl.f32, float(vK1))
                one = tkl.constant((BLOCK_M, K1, ), tkl.f32, 1.0)
                dk_sqrt = tkl.sqrt(one / k1_val_f)
                log2e = tkl.constant((BLOCK_M, K1, ), tkl.f32, vLOG2E)
                qkv_scaling = log2e * dk_sqrt
                q = q * qkv_scaling

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
                        # (BLOCK_M, BLOCK_K2) <- (BLOCK_M, K1) * (K1, BLOCK_K2)
                        qkT = tkl.constant((BLOCK_M, BLOCK_K2), tkl.f32, 0.0)
                        qkT = tkl.dot(q, kT, qkT) # : (BLOCK_M, BLOCK_K2)

                        m_j = tkl.max(qkT, axis=1, acc=partial_max) # : (BLOCK_M)
                        e_delta_max = tkl.exp2(partial_max - m_j) # : (BLOCK_M)

                        # TODO: are there better ways to do this (e.g. tkl.broadcast_to_typeof(partial_max)) ?
                        m_j_bcast = tkl.broadcast_in_dim(m_j, (BLOCK_M, BLOCK_K2), (1, )) # : (BLOCK_M, BLOCK_K2)
                        e_delta = tkl.exp2(qkT - m_j_bcast) # : (BLOCK_M, BLOCK_K2)
                        e_init = e_delta_max * partial_sum # : (BLOCK_M)
                        d_j = tkl.sum(e_delta, axis=1, acc=e_init) # : (BLOCK_M)

                        # TODO: are there better ways to do this (e.g. tkl.broadcast_to_typeof(acc)) ?
                        e_delta_max = tkl.broadcast_in_dim(e_delta_max, (BLOCK_M, BLOCK_N), (1, )) # : (BLOCK_M, BLOCK_N)
                        acc = acc * e_delta_max # : (BLOCK_M, BLOCK_N)

                        # (BLOCK_M, BLOCK_N) <- (BLOCK_M, BLOCK_K2) * (BLOCK_K2, BLOCK_N)
                        imm_f16 = tkl.to_dtype(e_delta, tkl.f32) # : (BLOCK_M, BLOCK_K2)
                        v = tkl.load(V, (b, h, k2, n), (BLOCK_K2, BLOCK_N)) 
                        acc = tkl.dot(imm_f16, v, acc) # : (BLOCK_M, BLOCK_N)

                        return (m_j, d_j, acc)

                    max, sum, res = inner_body
                    one = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 1.0)
                    one_by_sum = one / sum
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
        Q = torch.randn(vB, vH, vM, vK1).to(torch.float32)
        K = torch.randn(vB, vH, vK2, vK1).to(torch.float32)
        V = torch.randn(vB, vH, vK2, vN).to(torch.float32)
        O = torch.zeros(vB, vH, vM, vN).to(torch.float32)

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
            torch.testing.assert_close(O, ref)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
