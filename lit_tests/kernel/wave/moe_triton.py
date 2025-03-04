# # Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

# """Fused MoE kernel."""

# import functools
# import json
# import logging
# import os
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import torch
# import triton
# import triton.language as tl
# from vllm import _custom_ops as ops

# from sglang.srt.layers.moe.topk import select_experts
# from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
# from sglang.srt.layers.quantization.int8_kernel import per_token_group_quant_int8
# from sglang.srt.utils import (
#     direct_register_custom_op,
#     get_device_name,
#     is_cuda_available,
#     is_hip,
# )

# is_hip_ = is_hip()


# logger = logging.getLogger(__name__)
# padding_size = 128 if bool(int(os.getenv("MOE_PADDING", "0"))) else 0

# enable_moe_align_block_size_triton = bool(
#     int(os.getenv("ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON", "0"))
# )

# _is_cuda = torch.cuda.is_available() and torch.version.cuda
# _is_rocm = torch.cuda.is_available() and torch.version.hip

# if _is_cuda:
#     from sgl_kernel import gelu_and_mul, silu_and_mul

#     from sglang.srt.layers.quantization.fp8_kernel import (
#         sglang_per_token_group_quant_fp8,
#     )

# if _is_cuda or _is_rocm:
#     from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size


# @triton.jit
# def fused_moe_kernel(
#     # Pointers to matrices
#     a_ptr,
#     b_ptr,
#     c_ptr,
#     a_scale_ptr,
#     b_scale_ptr,
#     topk_weights_ptr,
#     sorted_token_ids_ptr,
#     expert_ids_ptr,
#     num_tokens_post_padded_ptr,
#     # Matrix dimensions
#     N,
#     K,
#     EM,
#     num_valid_tokens,
#     # The stride variables represent how much to increase the ptr by when
#     # moving by 1 element in a particular dimension. E.g. `stride_am` is
#     # how much to increase `a_ptr` by to get the element one row down
#     # (A has M rows).
#     stride_am,
#     stride_ak,
#     stride_be,
#     stride_bk,
#     stride_bn,
#     stride_cm,
#     stride_cn,
#     stride_asm,
#     stride_ask,
#     stride_bse,
#     stride_bsk,
#     stride_bsn,
#     # Block size for block-wise quantization
#     group_n: tl.constexpr,
#     group_k: tl.constexpr,
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
#     MUL_ROUTED_WEIGHT: tl.constexpr,
#     top_k: tl.constexpr,
#     compute_type: tl.constexpr,
#     use_fp8_w8a8: tl.constexpr,
#     use_int8_w8a8: tl.constexpr,
#     use_int8_w8a16: tl.constexpr,
#     even_Ks: tl.constexpr,
# ):
#     """
#     Implements the fused computation for a Mixture of Experts (MOE) using
#     token and expert matrices.

#     Key Parameters:
#     - A: The input tensor representing tokens with shape (*, K), where '*' can
#         be any shape representing batches and K is the feature dimension of
#         each token.
#     - B: The stacked MOE weight tensor with shape (E, N, K), where E is
#         the number of experts, K is the input feature dimension, and N is
#         the output feature dimension.
#     - C: The output cache tensor with shape (M, topk, N), where M is the
#         total number of tokens post padding, topk is the number of times
#         each token is repeated, and N is the output feature dimension.
#     - sorted_token_ids: A tensor containing the sorted indices of tokens,
#         repeated topk times and arranged by the expert index they are
#         assigned to.
#     - expert_ids: A tensor containing the indices of the expert for each
#         block. It determines which expert matrix from B should be used for
#         each block in A.
#     This kernel performs the multiplication of a token by its corresponding
#     expert matrix as determined by `expert_ids`. The sorting of
#     `sorted_token_ids` by expert index and padding ensures divisibility by
#     BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
#     multiplication across different blocks processed by the same expert.
#     """
#     # -----------------------------------------------------------
#     # Map program ids `pid` to the block of C it should compute.
#     # This is done in a grouped ordering to promote L2 data reuse.
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     # We will advance this pointer as we move in the K direction
#     # and accumulate
#     # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
#     # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
#     num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
#     if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
#         return
#     offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
#     token_mask = offs_token < num_valid_tokens

#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + (
#         offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
#     )

#     off_experts = tl.load(expert_ids_ptr + pid_m)
#     b_ptrs = (
#         b_ptr
#         + off_experts * stride_be
#         + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#     )
#     # if use_int8_w8a16:
#     #     b_scale_ptrs = (
#     #         b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
#     #     )
#     #     b_scale = tl.load(b_scale_ptrs)

#     # if use_fp8_w8a8 or use_int8_w8a8:
#     #     if group_k > 0 and group_n > 0:
#     #         a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
#     #         offs_bsn = offs_bn // group_n
#     #         b_scale_ptrs = (
#     #             b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
#     #         )
#     #     else:
#     #         a_scale = tl.load(a_scale_ptr)
#     #         b_scale = tl.load(b_scale_ptr + off_experts)

#     # -----------------------------------------------------------
#     # Iterate to compute a block of the C matrix.
#     # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
#     # of fp32 values for higher accuracy.
#     # `accumulator` will be converted back to fp16 after the loop.
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Load the next block of A and B, generate a mask by checking the
#         # K dimension.
#         # if even_Ks:
#         #     a = tl.load(
#         #         a_ptrs,
#         #         mask=token_mask[:, None],
#         #         other=0.0,
#         #     )
#         #     b = tl.load(b_ptrs)
#         # else:
#         #     a = tl.load(
#         #         a_ptrs,
#         #         mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
#         #         other=0.0,
#         #     )
#         #     b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

#         a = tl.load(
#             a_ptrs,
#             mask=token_mask[:, None],
#             other=0.0,
#         )
#         b = tl.load(b_ptrs)

#         # We accumulate along the K dimension.
#         # if use_int8_w8a16:
#         #     accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
#         # elif use_fp8_w8a8 or use_int8_w8a8:
#         #     if group_k > 0 and group_n > 0:
#         #         k_start = k * BLOCK_SIZE_K
#         #         offs_ks = k_start // group_k
#         #         a_scale = tl.load(
#         #             a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
#         #         )
#         #         b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

#         #         accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
#         #     else:
#         #         accumulator = tl.dot(a, b, acc=accumulator)
#         # else:
#         #     accumulator += tl.dot(a, b)
#           accumulator += tl.dot(a, b)
#         # Advance the ptrs to the next K block.
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     # if MUL_ROUTED_WEIGHT:
#     #     moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
#     #     accumulator = accumulator * moe_weight[:, None]
#     # if use_int8_w8a16:
#     #     accumulator = (accumulator * b_scale).to(compute_type)
#     # elif use_fp8_w8a8 or use_int8_w8a8:
#     #     if group_k > 0 and group_n > 0:
#     #         accumulator = accumulator.to(compute_type)
#     #     else:
#     #         accumulator = (accumulator * a_scale * b_scale).to(compute_type)
#     # else:
#     #     accumulator = accumulator.to(compute_type)
#     accumulator = accumulator.to(compute_type)
#     # -----------------------------------------------------------
#     # Write back the block of the output
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, accumulator, mask=c_mask)


# def ceil_div(a, b):
#     return (a + b - 1) // b


# @triton.jit
# def moe_align_block_size_stage1(
#     topk_ids_ptr,
#     tokens_cnts_ptr,
#     num_experts: tl.constexpr,
#     numel: tl.constexpr,
#     tokens_per_thread: tl.constexpr,
# ):
#     pid = tl.program_id(0)

#     start_idx = pid * tokens_per_thread

#     off_c = (pid + 1) * num_experts

#     for i in range(tokens_per_thread):
#         if start_idx + i < numel:
#             idx = tl.load(topk_ids_ptr + start_idx + i)
#             token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
#             tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


# @triton.jit
# def moe_align_block_size_stage2(
#     tokens_cnts_ptr,
#     num_experts: tl.constexpr,
# ):
#     pid = tl.program_id(0)

#     last_cnt = 0
#     for i in range(1, num_experts + 1):
#         token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
#         last_cnt = last_cnt + token_cnt
#         tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


# @triton.jit
# def moe_align_block_size_stage3(
#     total_tokens_post_pad_ptr,
#     tokens_cnts_ptr,
#     cumsum_ptr,
#     num_experts: tl.constexpr,
#     block_size: tl.constexpr,
# ):
#     last_cumsum = 0
#     off_cnt = num_experts * num_experts
#     for i in range(1, num_experts + 1):
#         token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
#         last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
#         tl.store(cumsum_ptr + i, last_cumsum)
#     tl.store(total_tokens_post_pad_ptr, last_cumsum)


# @triton.jit
# def moe_align_block_size_stage4(
#     topk_ids_ptr,
#     sorted_token_ids_ptr,
#     expert_ids_ptr,
#     tokens_cnts_ptr,
#     cumsum_ptr,
#     num_experts: tl.constexpr,
#     block_size: tl.constexpr,
#     numel: tl.constexpr,
#     tokens_per_thread: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     start_idx = tl.load(cumsum_ptr + pid)
#     end_idx = tl.load(cumsum_ptr + pid + 1)

#     for i in range(start_idx, end_idx, block_size):
#         tl.store(expert_ids_ptr + i // block_size, pid)

#     start_idx = pid * tokens_per_thread
#     off_t = pid * num_experts

#     for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
#         expert_id = tl.load(topk_ids_ptr + i)
#         token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
#         rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
#         tl.store(sorted_token_ids_ptr + rank_post_pad, i)
#         tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# def moe_align_block_size_triton(
#     topk_ids: torch.Tensor,
#     num_experts: int,
#     block_size: int,
#     sorted_token_ids: torch.Tensor,
#     expert_ids: torch.Tensor,
#     num_tokens_post_pad: torch.Tensor,
# ) -> None:
#     numel = topk_ids.numel()
#     grid = (num_experts,)
#     tokens_cnts = torch.zeros(
#         (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
#     )
#     cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
#     tokens_per_thread = ceil_div(numel, num_experts)

#     moe_align_block_size_stage1[grid](
#         topk_ids,
#         tokens_cnts,
#         num_experts,
#         numel,
#         tokens_per_thread,
#     )
#     moe_align_block_size_stage2[grid](
#         tokens_cnts,
#         num_experts,
#     )
#     moe_align_block_size_stage3[(1,)](
#         num_tokens_post_pad,
#         tokens_cnts,
#         cumsum,
#         num_experts,
#         block_size,
#     )
#     moe_align_block_size_stage4[grid](
#         topk_ids,
#         sorted_token_ids,
#         expert_ids,
#         tokens_cnts,
#         cumsum,
#         num_experts,
#         block_size,
#         numel,
#         tokens_per_thread,
#     )


# def moe_align_block_size(
#     topk_ids: torch.Tensor, block_size: int, num_experts: int
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Aligns the token distribution across experts to be compatible with block
#     size for matrix multiplication.

#     Parameters:
#     - topk_ids: A tensor of shape [total_tokens, top_k] representing the
#         top-k expert indices for each token.
#     - block_size: The block size used in block matrix multiplication.
#     - num_experts: The total number of experts.

#     Returns:
#     - sorted_token_ids: A tensor containing the sorted token indices according
#         to their allocated expert.
#     - expert_ids: A tensor indicating the assigned expert index for each block.
#     - num_tokens_post_padded: The total number of tokens after padding,
#         ensuring divisibility by block_size.

#     This function pads the number of tokens that each expert needs to process
#     so that it is divisible by block_size.
#     Padding ensures that during block matrix multiplication, the dimensions
#     align correctly.

#     Example:
#     Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
#     block_size = 4, and num_experts = 4:
#     - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
#         with each expert needing to process 3 tokens.
#     - As block_size is 4, we pad 1 token for each expert.
#     - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
#     - Then append padding tokens [12, 12, 12, 12] for each block.
#     - After sorting by expert index, we obtain token_ids
#         [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
#         Tokens 12 are non-existent (padding) and are ignored in
#         the subsequent matrix multiplication.
#     - The padding ensures that the total number of tokens is now divisible
#         by block_size for proper block matrix operations.
#     """
#     max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
#     sorted_ids = torch.empty(
#         (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
#     )
#     sorted_ids.fill_(topk_ids.numel())
#     max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
#     expert_ids = torch.empty(
#         (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
#     )
#     num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
#     if num_experts >= 224:
#         if enable_moe_align_block_size_triton:
#             moe_align_block_size_triton(
#                 topk_ids,
#                 num_experts,
#                 block_size,
#                 sorted_ids,
#                 expert_ids,
#                 num_tokens_post_pad,
#             )
#         else:
#             token_cnts_buffer = torch.zeros(
#                 (num_experts + 1) * num_experts,
#                 dtype=torch.int32,
#                 device=topk_ids.device,
#             )
#             cumsum_buffer = torch.zeros(
#                 num_experts + 1, dtype=torch.int32, device=topk_ids.device
#             )

#             sgl_moe_align_block_size(
#                 topk_ids,
#                 num_experts,
#                 block_size,
#                 sorted_ids,
#                 expert_ids,
#                 num_tokens_post_pad,
#                 token_cnts_buffer,
#                 cumsum_buffer,
#             )
#     else:
#         ops.moe_align_block_size(
#             topk_ids,
#             num_experts,
#             block_size,
#             sorted_ids,
#             expert_ids,
#             num_tokens_post_pad,
#         )
#     return sorted_ids, expert_ids, num_tokens_post_pad


# def invoke_fused_moe_kernel(
#     A: torch.Tensor,
#     B: torch.Tensor,
#     C: torch.Tensor,
#     A_scale: Optional[torch.Tensor],
#     B_scale: Optional[torch.Tensor],
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     sorted_token_ids: torch.Tensor,
#     expert_ids: torch.Tensor,
#     num_tokens_post_padded: torch.Tensor,
#     mul_routed_weight: bool,
#     top_k: int,
#     config: Dict[str, Any],
#     compute_type: tl.dtype,
#     use_fp8_w8a8: bool,
#     use_int8_w8a8: bool,
#     use_int8_w8a16: bool,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ) -> None:
#     assert topk_weights.stride(1) == 1
#     assert sorted_token_ids.stride(0) == 1

#     padded_size = 0
#     if use_fp8_w8a8:
#         assert B_scale is not None
#         if block_shape is None:
#             padded_size = padding_size
#             A, A_scale = ops.scaled_fp8_quant(A, A_scale)
#         else:
#             assert len(block_shape) == 2
#             block_n, block_k = block_shape[0], block_shape[1]
#             if _is_cuda:
#                 A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
#             else:
#                 A, A_scale = per_token_group_quant_fp8(A, block_k)
#             assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
#             assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
#             assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
#     elif use_int8_w8a8:
#         assert B_scale is not None
#         if block_shape is None:
#             padded_size = padding_size
#             A, A_scale = ops.scaled_int8_quant(A, A_scale)
#         else:
#             assert len(block_shape) == 2
#             block_n, block_k = block_shape[0], block_shape[1]
#             A, A_scale = per_token_group_quant_int8(A, block_k)
#             assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
#             assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
#             assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
#     elif use_int8_w8a16:
#         assert B_scale is not None
#     else:
#         assert A_scale is None
#         assert B_scale is None

#     grid = lambda META: (
#         triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
#         * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
#     )

#     K = B.shape[2] - padded_size
#     if K % config["BLOCK_SIZE_K"] == 0:
#         even_Ks = True
#     else:
#         even_Ks = False

#     fused_moe_kernel[grid](
#         A,
#         B,
#         C,
#         A_scale,
#         B_scale,
#         topk_weights,
#         sorted_token_ids,
#         expert_ids,
#         num_tokens_post_padded,
#         B.shape[1],
#         B.shape[2] - padded_size,
#         sorted_token_ids.shape[0],
#         topk_ids.numel(),
#         A.stride(0),
#         A.stride(1),
#         B.stride(0),
#         B.stride(2),
#         B.stride(1),
#         C.stride(1),
#         C.stride(2),
#         A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
#         A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
#         B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
#         B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
#         B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
#         0 if block_shape is None else block_shape[0],
#         0 if block_shape is None else block_shape[1],
#         MUL_ROUTED_WEIGHT=mul_routed_weight,
#         top_k=top_k,
#         compute_type=compute_type,
#         use_fp8_w8a8=use_fp8_w8a8,
#         use_int8_w8a8=use_int8_w8a8,
#         use_int8_w8a16=use_int8_w8a16,
#         even_Ks=even_Ks,
#         **config,
#     )


# def get_config_file_name(
#     E: int, N: int, dtype: Optional[str], block_shape: Optional[int] = None
# ) -> str:
#     device_name = get_device_name().replace(" ", "_")
#     dtype_selector = "" if not dtype else f",dtype={dtype}"
#     block_shape_selector = (
#         "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
#     )
#     return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"


# @functools.lru_cache
# def get_moe_configs(
#     E: int,
#     N: int,
#     dtype: Optional[str],
#     block_n: Optional[int] = 0,
#     block_k: Optional[int] = 0,
# ) -> Optional[Dict[int, Any]]:
#     """
#     Return optimized configurations for the fused MoE kernel.

#     The return value will be a dictionary that maps an irregular grid of
#     batch sizes to configurations of the fused_moe kernel. To evaluate the
#     kernel on a given batch size bs, the closest batch size in the grid should
#     be picked and the associated configuration chosen to invoke the kernel.
#     """

#     # First look up if an optimized configuration is available in the configs
#     # directory
#     json_file_name = get_config_file_name(E, N, dtype, [block_n, block_k])

#     config_file_path = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
#     )
#     if os.path.exists(config_file_path):
#         with open(config_file_path) as f:
#             logger.info("Using configuration from %s for MoE layer.", config_file_path)
#             # If a configuration has been found, return it
#             return {int(key): val for key, val in json.load(f).items()}

#     # If no optimized configuration is available, we will use the default
#     # configuration
#     logger.warning(
#         (
#             "Using default MoE config. Performance might be sub-optimal! "
#             "Config file not found at %s"
#         ),
#         config_file_path,
#     )
#     return None


# def get_default_config(
#     M: int,
#     E: int,
#     N: int,
#     K: int,
#     topk: int,
#     dtype: Optional[str],
#     is_marlin: bool,
#     block_shape: Optional[List[int]] = None,
# ) -> Dict[str, int]:
#     if dtype == "fp8_w8a8":
#         if block_shape is None:
#             config = {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 32,
#                 "num_warps": 8,
#                 "num_stages": 2 if is_hip_ else 4,
#             }
#             if M <= E:
#                 config = {
#                     "BLOCK_SIZE_M": 64,
#                     "BLOCK_SIZE_N": 128,
#                     "BLOCK_SIZE_K": 128,
#                     "GROUP_SIZE_M": 1,
#                     "num_warps": 4,
#                     "num_stages": 2 if is_hip_ else 4,
#                 }
#         else:
#             # Block-wise quant: BLOCK_SIZE_K must be divisable by block_shape[1]
#             config = {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": block_shape[0],
#                 "BLOCK_SIZE_K": block_shape[1],
#                 "GROUP_SIZE_M": 32,
#                 "num_warps": 4,
#                 "num_stages": 2 if is_hip_ else 3,
#             }
#     else:
#         config = {
#             "BLOCK_SIZE_M": 64,
#             "BLOCK_SIZE_N": 64,
#             "BLOCK_SIZE_K": 32,
#             "GROUP_SIZE_M": 8,
#         }
#         # A heuristic: fused marlin works faster with this config for small M
#         if M <= E or (is_marlin and M <= 32):
#             config = {
#                 "BLOCK_SIZE_M": 16,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 1,
#             }
#     return config


# def try_get_optimal_moe_config(
#     w1_shape: Tuple[int, ...],
#     w2_shape: Tuple[int, ...],
#     top_k: int,
#     dtype: Optional[str],
#     M: int,
#     is_marlin: bool = False,
#     block_shape: Optional[List[int]] = None,
# ):
#     from sglang.srt.layers.moe.fused_moe_triton import get_config

#     override_config = get_config()
#     if override_config:
#         config = override_config
#     else:
#         # First try to load optimal config from the file
#         E, _, N = w2_shape
#         block_n = block_shape[0] if block_shape else 0
#         block_k = block_shape[1] if block_shape else 0
#         configs = get_moe_configs(E, N, dtype, block_n, block_k)

#         if configs:
#             # If an optimal configuration map has been found, look up the
#             # optimal config
#             config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
#         else:
#             # Else use the default config
#             config = get_default_config(
#                 M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
#             )
#     return config


# def get_config_dtype_str(
#     dtype: torch.dtype,
#     use_int8_w8a16: Optional[bool] = False,
#     use_fp8_w8a8: Optional[bool] = False,
#     use_int8_w8a8: Optional[bool] = False,
# ):
#     if use_fp8_w8a8:
#         return "fp8_w8a8"
#     elif use_int8_w8a8:
#         return "int8_w8a8"
#     elif use_int8_w8a16:
#         return "int8_w8a16"
#     elif dtype == torch.float:
#         # avoiding cases where kernel fails when float32 MoE
#         # use fp16/bfloat16 configs
#         return "float32"
#     return None


# def inplace_fused_experts(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
# ) -> None:
#     fused_experts_impl(
#         hidden_states,
#         w1,
#         w2,
#         topk_weights,
#         topk_ids,
#         True,
#         activation,
#         use_fp8_w8a8,
#         use_int8_w8a8,
#         use_int8_w8a16,
#         w1_scale,
#         w2_scale,
#         a1_scale,
#         a2_scale,
#         block_shape,
#     )


# def inplace_fused_experts_fake(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
# ) -> None:
#     pass


# direct_register_custom_op(
#     op_name="inplace_fused_experts",
#     op_func=inplace_fused_experts,
#     mutates_args=["hidden_states"],
#     fake_impl=inplace_fused_experts_fake,
# )


# def outplace_fused_experts(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ) -> torch.Tensor:
#     return fused_experts_impl(
#         hidden_states,
#         w1,
#         w2,
#         topk_weights,
#         topk_ids,
#         False,
#         activation,
#         use_fp8_w8a8,
#         use_int8_w8a8,
#         use_int8_w8a16,
#         w1_scale,
#         w2_scale,
#         a1_scale,
#         a2_scale,
#         block_shape,
#         no_combine=no_combine,
#     )


# def outplace_fused_experts_fake(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ) -> torch.Tensor:
#     return torch.empty_like(hidden_states)


# direct_register_custom_op(
#     op_name="outplace_fused_experts",
#     op_func=outplace_fused_experts,
#     mutates_args=[],
#     fake_impl=outplace_fused_experts_fake,
# )


# def fused_experts(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     inplace: bool = False,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ):
#     if inplace:
#         assert not no_combine, "no combine + inplace makes no sense"
#         torch.ops.sglang.inplace_fused_experts(
#             hidden_states,
#             w1,
#             w2,
#             topk_weights,
#             topk_ids,
#             activation,
#             use_fp8_w8a8,
#             use_int8_w8a8,
#             use_int8_w8a16,
#             w1_scale,
#             w2_scale,
#             a1_scale,
#             a2_scale,
#             block_shape,
#         )
#         return hidden_states
#     else:
#         return torch.ops.sglang.outplace_fused_experts(
#             hidden_states,
#             w1,
#             w2,
#             topk_weights,
#             topk_ids,
#             activation,
#             use_fp8_w8a8,
#             use_int8_w8a8,
#             use_int8_w8a16,
#             w1_scale,
#             w2_scale,
#             a1_scale,
#             a2_scale,
#             block_shape,
#             no_combine=no_combine,
#         )


# def fused_experts_impl(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     inplace: bool = False,
#     activation: str = "silu",
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ):
#     padded_size = padding_size
#     if not use_fp8_w8a8 or not use_int8_w8a8 or block_shape is not None:
#         padded_size = 0

#     # Check constraints.
#     assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
#     assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
#     assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
#     assert w1.is_contiguous(), "Expert weights1 must be contiguous"
#     assert w2.is_contiguous(), "Expert weights2 must be contiguous"
#     assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

#     num_tokens, _ = hidden_states.shape
#     E, N, _ = w1.shape
#     # We execute the fused_moe kernel in chunks to circumvent this issue:
#     # https://github.com/vllm-project/vllm/issues/5938
#     CHUNK_SIZE = 64 * 1024
#     M = min(num_tokens, CHUNK_SIZE)
#     config_dtype = get_config_dtype_str(
#         use_fp8_w8a8=use_fp8_w8a8,
#         use_int8_w8a8=use_int8_w8a8,
#         use_int8_w8a16=use_int8_w8a16,
#         dtype=hidden_states.dtype,
#     )

#     get_config_func = functools.partial(
#         try_get_optimal_moe_config,
#         w1.shape,
#         (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
#         topk_ids.shape[1],
#         config_dtype,
#         block_shape=block_shape,
#     )

#     config = get_config_func(M)

#     cache = torch.empty(
#         M * topk_ids.shape[1] * max(N, w2.shape[1]),
#         device=hidden_states.device,
#         dtype=hidden_states.dtype,
#     )
#     intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
#         (M, topk_ids.shape[1], N),
#     )
#     intermediate_cache2 = torch.empty(
#         (M * topk_ids.shape[1], N // 2),
#         device=hidden_states.device,
#         dtype=hidden_states.dtype,
#     )
#     intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
#         (M, topk_ids.shape[1], w2.shape[1]),
#     )

#     compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

#     if no_combine:
#         assert not inplace
#         out_hidden_states = torch.empty(
#             (num_tokens, topk_ids.shape[1], w2.shape[1]),
#             device=hidden_states.device,
#             dtype=hidden_states.dtype,
#         )
#     elif inplace:
#         out_hidden_states = hidden_states
#     else:
#         out_hidden_states = torch.empty_like(hidden_states)

#     for chunk in range((num_tokens // CHUNK_SIZE) + 1):
#         begin_chunk_idx, end_chunk_idx = (
#             chunk * CHUNK_SIZE,
#             min((chunk + 1) * CHUNK_SIZE, num_tokens),
#         )
#         curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
#         tokens_in_chunk, _ = curr_hidden_states.shape

#         if tokens_in_chunk == 0:
#             break

#         if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
#             # Adjust the intermediate cache size and config for the last
#             # chunk. Note that in most cases we only have one chunk
#             # so the cache size and config are already set correctly and
#             # do not need to be adjusted.
#             intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
#             intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
#             intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
#             config = get_config_func(tokens_in_chunk)

#         curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
#         curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

#         sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
#             curr_topk_ids, config["BLOCK_SIZE_M"], E
#         )

#         invoke_fused_moe_kernel(
#             curr_hidden_states,
#             w1,
#             intermediate_cache1,
#             a1_scale,
#             w1_scale,
#             curr_topk_weights,
#             curr_topk_ids,
#             sorted_token_ids,
#             expert_ids,
#             num_tokens_post_padded,
#             False,
#             topk_ids.shape[1],
#             config,
#             compute_type=compute_type,
#             use_fp8_w8a8=use_fp8_w8a8,
#             use_int8_w8a8=use_int8_w8a8,
#             use_int8_w8a16=use_int8_w8a16,
#             block_shape=block_shape,
#         )

#         if activation == "silu":
#             if _is_cuda:
#                 silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
#             else:
#                 ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
#         elif activation == "gelu":
#             if _is_cuda:
#                 gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
#             else:
#                 ops.gelu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
#         else:
#             raise ValueError(f"Unsupported activation: {activation=}")

#         invoke_fused_moe_kernel(
#             intermediate_cache2,
#             w2,
#             (
#                 intermediate_cache3
#                 if not no_combine and topk_ids.shape[1] != 1
#                 else out_hidden_states[begin_chunk_idx:end_chunk_idx]
#             ),
#             a2_scale,
#             w2_scale,
#             curr_topk_weights,
#             curr_topk_ids,
#             sorted_token_ids,
#             expert_ids,
#             num_tokens_post_padded,
#             True,
#             1,
#             config,
#             compute_type=compute_type,
#             use_fp8_w8a8=use_fp8_w8a8,
#             use_int8_w8a8=use_int8_w8a8,
#             use_int8_w8a16=use_int8_w8a16,
#             block_shape=block_shape,
#         )

#         if no_combine:
#             pass
#         elif is_hip_:
#             ops.moe_sum(
#                 intermediate_cache3.view(*intermediate_cache3.shape),
#                 out_hidden_states[begin_chunk_idx:end_chunk_idx],
#             )
#         else:
#             if topk_ids.shape[1] == 1:
#                 pass  # we write directly into out_hidden_states
#             elif topk_ids.shape[1] == 2:
#                 torch.add(
#                     intermediate_cache3[:, 0],
#                     intermediate_cache3[:, 1],
#                     out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
#                 ).squeeze(dim=1)
#             elif topk_ids.shape[1] > 2:
#                 torch.sum(
#                     intermediate_cache3.view(*intermediate_cache3.shape),
#                     dim=1,
#                     out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
#                 )

#     return out_hidden_states


# def fused_moe(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     gating_output: torch.Tensor,
#     topk: int,
#     renormalize: bool,
#     inplace: bool = False,
#     activation: str = "silu",
#     use_grouped_topk: bool = False,
#     num_expert_group: Optional[int] = None,
#     topk_group: Optional[int] = None,
#     custom_routing_function: Optional[Callable] = None,
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     block_shape: Optional[List[int]] = None,
#     no_combine: bool = False,
# ) -> torch.Tensor:
#     """
#     This function computes a Mixture of Experts (MoE) layer using two sets of
#     weights, w1 and w2, and top-k gating mechanism.

#     Parameters:
#     - hidden_states (torch.Tensor): The input tensor to the MoE layer.
#     - w1 (torch.Tensor): The first set of expert weights.
#     - w2 (torch.Tensor): The second set of expert weights.
#     - gating_output (torch.Tensor): The output of the gating operation
#         (before softmax).
#     - topk (int): The number of top-k experts to select.
#     - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
#     - inplace (bool): If True, perform the operation in-place.
#         Defaults to False.
#     - num_expert_group: Optional[int]: additional parameter for grouped_topk
#     - topk_group: Optional[int]: additional parameter for grouped_topk
#     - use_grouped_topk: If True, use grouped_topk instead of fused_topk
#         note: Deepseek V2/V3/R1 series models use grouped_topk
#     - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
#         products for w1 and w2. Defaults to False.
#     - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
#         products for w1 and w2. Defaults to False.
#     - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
#         products for w1 and w2. Defaults to False.
#     - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
#         w1.
#     - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
#         w2.
#     - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
#         a1.
#     - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
#         a2.
#     - block_shape: (Optional[List[int]]): Optional block size for block-wise
#         quantization.

#     Returns:
#     - torch.Tensor: The output tensor after applying the MoE layer.
#     """
#     # Check constraints.
#     assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

#     topk_weights, topk_ids = select_experts(
#         hidden_states=hidden_states,
#         router_logits=gating_output,
#         use_grouped_topk=use_grouped_topk,
#         top_k=topk,
#         renormalize=renormalize,
#         topk_group=topk_group,
#         num_expert_group=num_expert_group,
#         custom_routing_function=custom_routing_function,
#     )

#     return fused_experts(
#         hidden_states,
#         w1,
#         w2,
#         topk_weights,
#         topk_ids,
#         inplace=inplace,
#         activation=activation,
#         use_fp8_w8a8=use_fp8_w8a8,
#         use_int8_w8a8=use_int8_w8a8,
#         use_int8_w8a16=use_int8_w8a16,
#         w1_scale=w1_scale,
#         w2_scale=w2_scale,
#         a1_scale=a1_scale,
#         a2_scale=a2_scale,
#         block_shape=block_shape,
#         no_combine=no_combine,
#     )
