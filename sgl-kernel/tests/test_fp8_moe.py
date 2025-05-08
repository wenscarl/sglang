import random

import pytest
import torch
from sgl_kernel import fp8_grouped_mm
import pdb


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def baseline_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: type[torch.dtype],
) -> torch.Tensor:
    # pdb.set_trace()
    return torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32)
    ).to(out_dtype).t()


@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_fp8_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 16
    n_g = alignment * random.randint(1, 5) * 128
    k_g = alignment * random.randint(1, 5) * 128
    # n_g = alignment * 2 * 128
    # k_g = alignment * 2 * 128


    expert_offsets = torch.zeros((num_experts + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)


    a_tensors = []
    b_tensors = []
    baseline_tensors = []

    for g in range(num_experts):
        m_g = alignment * random.randint(1, 64)
        # m_g = alignment * 2 * 128
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        b_g = to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        baseline = baseline_mm(
            a_g, b_g, out_dtype
        )
        baseline_tensors.append(baseline)

    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn
    )
    b_stack = torch.empty(
        (num_experts, n_g, k_g), device=device, dtype=torch.float8_e4m3fn
    )

    for g in range(num_experts):
        a_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)

    c_out = torch.empty((expert_offsets[-1], n_g), device=device, dtype=out_dtype)
    a_strides = torch.full(
        (num_experts,), a_stack.stride(0), device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_experts,), c_out.stride(0), device=device, dtype=torch.int64
    )
    # pdb.set_trace()
    fp8_grouped_mm(
        c_out,
        a_stack,
        b_stack,
        a_strides,
        a_strides,
        c_strides,
        problem_sizes,
        expert_offsets[:-1],
    )
    # pdb.set_trace()
    for g in range(num_experts):
        baseline = baseline_tensors[g]
        actual = c_out[expert_offsets[g] : expert_offsets[g + 1]]
        print(actual)
        print(baseline)
        # torch.testing.assert_close(actual, baseline.t(), rtol=1e-2, atol=5e-4)
        # print(f"num_experts={num_experts}, out_dtype={out_dtype}: OK")


if __name__ == "__main__":
    pytest.main([__file__])