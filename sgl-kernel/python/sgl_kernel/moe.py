from typing import Any, Dict, Optional

import torch


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: float,
    renormalize: bool = False,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, gating_output, renormalize
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts=0,
    routed_scaling_factor=0,
    apply_routed_scaling_factor_on_output=False,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select expert groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limited for now.
    # for non-supported case, we suggest to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # num_fused_shared_experts: if > 0, the last several experts will be
    #   replaced with shared experts. the shared experts will be divided by the
    #   routed_scaling_factor - this is intended to cancel out later when routed+shared
    #   output is scaled so that shared experts are not scaled.
    # routed_scaling_factor: if > 0, the experts will be scaled by this factor
    # apply_routed_scaling_factor_on_output: if true, output will be
    #   scaled by the routed_scaling_factor
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def ep_moe_pre_reorder(
    input_tensor,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    start_expert_id,
    end_expert_id,
    topk,
    use_per_token_if_dynamic,
):
    return torch.ops.sgl_kernel.ep_moe_pre_reorder.default(
        input_tensor,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        use_per_token_if_dynamic,
    )


def ep_moe_silu_and_mul(
    gateup_output,
    down_input,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
):
    return torch.ops.sgl_kernel.ep_moe_silu_and_mul.default(
        gateup_output,
        down_input,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
    )


def ep_moe_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    start_expert_id,
    end_expert_id,
    topk,
):
    return torch.ops.sgl_kernel.ep_moe_post_reorder.default(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
    )


def fp8_blockwise_scaled_grouped_mm(
    output,
    a_ptrs,
    b_ptrs,
    out_ptrs,
    a_scales_ptrs,
    b_scales_ptrs,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_c,
    layout_sfa,
    layout_sfb,
    problem_sizes,
    expert_offsets,
    workspace,
):
    torch.ops.sgl_kernel.fp8_blockwise_scaled_grouped_mm.default(
        output,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def prepare_moe_input(
    topk_ids,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    num_experts,
    n,
    k,
    blockscale_offsets: Optional[torch.Tensor] = None,
):
    torch.ops.sgl_kernel.prepare_moe_input.default(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
    )


def apply_shuffle_mul_sum(
    input,
    output,
    permutation,
    factors,
):
    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        input, output, permutation, factors
    )


def cutlass_fp4_group_mm(
    a_fp4,
    b_fp4,
    a_blockscale,
    b_blockscale,
    alphas,
    out_dtype,
    device,
    params: Dict[str, Any],
):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - ab_strides/c_strides: Strides for the a/b tensors between rows.
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    m_topk = a_fp4.shape[0]
    n = b_fp4.shape[1]
    c_shape = (m_topk, n)
    c = torch.empty(c_shape, device=device, dtype=out_dtype)
    torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
        c,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        params["ab_strides"],
        params["c_strides"],
        params["problem_sizes"],
        params["expert_offsets"],
        params["blockscale_offsets"],
    )
    return c.to(dtype=out_dtype)


def flashinfer_cutedsl_moe_masked(hidden_states: torch.Tensor, #3d, bf16
                                  input_global_scale: torch.Tensor, # (l,)
                                  w1: torch.Tensor, #fp4 [l, 2 * n, k // 2] in uint8
                                  w1_blockscale: torch.Tensor, #e4m3, [l, 2*n ,k // 16]
                                  w1_alpha, # (l,)
                                  w2: torch.Tensor, #fp4 [l, k, n // 2] in uint8
                                  a2_global_scale: torch.Tensor, # (l,)
                                  w2_blockscale: torch.Tensor, #e4m3, [l, k, n // 16]
                                  w2_alpha, # (l,)
                                  masked_m: torch.Tensor,
                                  topk_idx: torch.Tensor, # (bs, topk)
                                  routing_weights: torch.Tensor, # (bs, topk)
):
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
    from .elementwise import silu_and_mul
    from .gemm import scaled_fp4_grouped_quant

    n = w1.shape[-2] // 2 # intermediate dimension
    num_experts, m, k = hidden_states.shape
    assert max(masked_m) == m
    
    aq, aq_sf = scaled_fp4_grouped_quant(
      hidden_states,
      input_global_scale,
    )
    gateup_output = torch.zeros((num_experts, m, n * 2), dtype=hidden_states.dtype, device=aq.device)
    gateup_output = gateup_output.permute(1, 2, 0) # requirement of kernel
    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    # Gemm1

    grouped_gemm_nt_masked(
        (aq, aq_sf),
        (w1.permute(1,2,0), w1_blockscale),
        gateup_output,
        masked_m.to(aq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    ) # in logical [m, n, l]
    # TODO(shuw): alpha can be fused into gemm kernel pending https://github.com/flashinfer-ai/flashinfer/pull/1498
    gateup_output *= w1_alpha.view(1, 1, num_experts)
    
    # SILU
    gateup_output = gateup_output.permute(2,0,1).view(-1, 2*n)
    down_input_shape = (*gateup_output.shape[:-1], gateup_output.shape[-1]//2)
    down_input = torch.empty(*down_input_shape, dtype=gateup_output.dtype, device=gateup_output.device)
    silu_and_mul(gateup_output, down_input)
    
    down_input = down_input.view(num_experts, m, n) # [l, m, n * 2]

    # Quantize intermediate
    diq, diq_sf = scaled_fp4_grouped_quant(
      down_input,
      a2_global_scale,
    )

    # Gemm2
    out = torch.zeros_like(hidden_states)
    out = out.permute(1, 2, 0) # requirement of kernel
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1,2,0), w2_blockscale),
        out,
        masked_m.to(diq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    ) # in logical [m, k, l]
    out *= w2_alpha.view(1, 1, num_experts)
    out = out.permute(2,0,1)

    positions = torch.nonzero(masked_m[topk_idx], as_tuple=False)
    rows, cols = positions[:,0], positions[:,1]
    experts = topk_idx[rows, cols]
    for i in range(num_experts):
        mask = (experts == i)
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            r, c = rows[idx], cols[idx]
            out[i, :len(r), :] *= routing_weights[r, c].unsqueeze(-1)
    return out
