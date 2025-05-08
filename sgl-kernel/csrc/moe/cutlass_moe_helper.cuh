#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <torch/all.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <
    typename ElementAB,
    typename ElementC,
    typename ElementAccumulator>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    int* problem_sizes,
    int* problem_sizes_transpose,
    bool transpose = false) {
  int expert_id = threadIdx.x;

  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[expert_id * 3] = n;
    problem_sizes_transpose[expert_id * 3 + 1] = m;
    problem_sizes_transpose[expert_id * 3 + 2] = k;
  }

  int32_t expert_offset = expert_offsets[expert_id];
  int a_stride = 0;
  int b_stride = 0;
  if (!transpose) {
    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
  } else {
    a_stride = expert_id * k * n;
    b_stride = expert_offset * k;
  }
  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE) \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                        \
    get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float>\
        <<<1, num_experts, 0, stream>>>(                                 \
            static_cast<int32_t*>(expert_offsets.data_ptr()),             \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),      \
            static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),      \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                   \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),    \
            static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),     \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                 \
            static_cast<int*>(problem_sizes.data_ptr()),                  \
            static_cast<int*>(problem_sizes_transpose.data_ptr()),        \
            transpose);                                                  \
  }

namespace {
void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors,
    torch::Tensor const& problem_sizes,
    torch::Tensor& problem_sizes_transpose,
    bool transpose = false) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0);
  TORCH_CHECK(a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0);

  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}
}  // namespace