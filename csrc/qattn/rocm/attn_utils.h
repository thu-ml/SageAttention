/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "../../utils.cuh"
// #include <cuda_fp16.h>
// #include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

// #include "../cp_async.cuh"
// #include "../mma.cuh"
// #include "../permuted_smem.cuh"
// #include "../numeric_conversion.cuh"

#define WARP_SIZE_CUDA 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define div_ceil(M, N) (((M) + (N)-1) / (N))

enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

enum class DataType {
    kHalf,
    kInt8,
    kInt4,
    kE4M3,
    kE5M2,
};

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class ComputeUnit {
  kTensorCore,
  kCudaCore,
};

__device__ __forceinline__ uint32_t get_warp_id()
{
  return threadIdx.y;
}

__device__ __forceinline__ uint32_t get_lane_id()
{
  return threadIdx.x;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_q()
{
  return get_warp_id() / num_warps_k;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_k()
{
  return get_warp_id() % num_warps_k;
}
