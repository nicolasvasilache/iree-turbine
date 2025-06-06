# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl

vB, vM, vN, vK = 4, 32, 64, 128
B, M, N, K = tkl.sym.B, tkl.sym.M, tkl.sym.N, tkl.sym.K
BLOCK_B, BLOCK_M, BLOCK_N, BLOCK_K = tkl.sym.BLOCK_B, tkl.sym.BLOCK_M, tkl.sym.BLOCK_N, tkl.sym.BLOCK_K

class Test(unittest.TestCase):
    def testBatchMatmul(self):
        @tk.gen.thread(B)
        def batch_matmul(
            bA: tkl.InputBuffer[B, M, K, tkl.f32],
            bB: tkl.InputBuffer[B, K, N, tkl.f32],
            bC: tkl.OutputBuffer[B, M, N, tkl.f32],
        ):
            # TODO: Naive mapping for now (1 along b per workgroup).
            # The goal is to use scf.forall mapping in the future and avoid having to think
            # about mod and div mapping.
            b = tkl.program_id(0)

            # This is really a forall that later should get mapped.
            @tkl.for_loop(0, M, BLOCK_M, init_args=[])
            def body(m):
                # This is really a forall that later should get mapped.
                @tkl.for_loop(0, N, BLOCK_N, init_args=[])
                def body(n):
                    acc_init = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 0.0)
                    @tkl.for_loop(0, K, BLOCK_K, init_args=[acc_init])
                    def body_k(k, acc):
                        va = tkl.load(bA, (b, m, k), (BLOCK_M, BLOCK_K))
                        vb = tkl.load(bB, (b, k, n), (BLOCK_K, BLOCK_N)) 
                        acc = tkl.dot(va, vb, acc) # : (BLOCK_M, BLOCK_N)
                        return (acc)

                    res = body_k
                    tkl.store(bC, (b, m, n), res)
                    return
                return

        tA = torch.randn(vB, vM, vK).to(torch.float32)
        tB = torch.randn(vB, vK, vN).to(torch.float32)
        tC = torch.zeros(vB, vM, vN).to(torch.float32)

        # TODO: (suggested by Kunwar) we could build a tk.gen.EagerLaunchContext to emit pytorch via tracing.
        # This would give us 2 intermediate testing points:
        #   1. Language traced -> PyTorch vs PyTorch reference implementation.
        #   2. Language traced -> MLIR vs Language traced -> PyTorch, potentially instruction by instruction.
        # This should be very useful for debugging and teaching purposes.
        vBLOCK_M, vBLOCK_N, vBLOCK_K = 1, 1, 8
        assert vM % vBLOCK_M == 0, "only divisible sizes supported for now"
        assert vN % vBLOCK_N == 0, "only divisible sizes supported for now"
        assert vK % vBLOCK_K == 0, "only divisible sizes supported for now"
        with tk.gen.TestLaunchContext(
            {
                BLOCK_M: min(vBLOCK_M, vM),
                BLOCK_N: min(vBLOCK_N, vN),
                BLOCK_K: min(vBLOCK_K, vK),
            }
        ):
            batch_matmul(tA, tB, tC)
            ref = tA @ tB
            torch.testing.assert_close(tC, ref)


    def testBatchMatmulTranspose(self):
        @tk.gen.thread(B)
        def batch_matmul_transpose(
            bA: tkl.InputBuffer[B, M, K, tkl.f32],
            bB: tkl.InputBuffer[B, K, N, tkl.f32],
            bC: tkl.OutputBuffer[N, B, M, tkl.f32],
        ):
            # TODO: Naive mapping for now (1 along b per workgroup).
            # The goal is to use scf.forall mapping in the future and avoid having to think
            # about mod and div mapping.
            b = tkl.program_id(0)

            # This is really a forall that later should get mapped.
            @tkl.for_loop(0, M, BLOCK_M, init_args=[])
            def body(m):
                # This is really a forall that later should get mapped.
                @tkl.for_loop(0, N, BLOCK_N, init_args=[])
                def body(n):
                    acc_init = tkl.constant((BLOCK_B, BLOCK_M, BLOCK_N), tkl.f32, 0.0)
                    @tkl.for_loop(0, K, BLOCK_K, init_args=[acc_init])
                    def body_k(k, acc):
                        va = tkl.load(bA, (b, m, k), (BLOCK_B, BLOCK_M, BLOCK_K))
                        vb = tkl.load(bB, (b, k, n), (BLOCK_B, BLOCK_K, BLOCK_N)) 
                        acc = tkl.batched_dot(va, vb, acc) # : (BLOCK_B, BLOCK_M, BLOCK_N)
                        return (acc)

                    res = body_k
                    res = tkl.transpose(res, (2, 0, 1))
                    tkl.store(bC, (n, b, m), res)
                    return
                return

        tA = torch.randn(vB, vM, vK).to(torch.float32)
        tB = torch.randn(vB, vK, vN).to(torch.float32)
        tC = torch.zeros(vN, vB, vM).to(torch.float32)

        # TODO: (suggested by Kunwar) we could build a tk.gen.EagerLaunchContext to emit pytorch via tracing.
        # This would give us 2 intermediate testing points:
        #   1. Language traced -> PyTorch vs PyTorch reference implementation.
        #   2. Language traced -> MLIR vs Language traced -> PyTorch, potentially instruction by instruction.
        # This should be very useful for debugging and teaching purposes.
        vBLOCK_B, vBLOCK_M, vBLOCK_N, vBLOCK_K = 1, 1, 1, 8
        assert vB % vBLOCK_B == 0, "only divisible sizes supported for now"
        assert vM % vBLOCK_M == 0, "only divisible sizes supported for now"
        assert vN % vBLOCK_N == 0, "only divisible sizes supported for now"
        assert vK % vBLOCK_K == 0, "only divisible sizes supported for now"
        with tk.gen.TestLaunchContext(
            {
                BLOCK_B: min(vBLOCK_B, vB),
                BLOCK_M: min(vBLOCK_M, vM),
                BLOCK_N: min(vBLOCK_N, vN),
                BLOCK_K: min(vBLOCK_K, vK),
            }
        ):
            batch_matmul_transpose(tA, tB, tC)
            ref = tA @ tB
            torch.testing.assert_close(tC, ref.permute(2, 0, 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
