/*
 * Copyright (c) 2024 by SageAttention team.
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/extension.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <ATen/hip/HIPContext.h>
#include <hip/amd_detail/amd_math_functions.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#else
#error "qk_int_sv_gfx12_native.cu is only intended for ROCm/HIP."
#endif

#include "../reduction_utils.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#if !defined(SAGEATTN_GFX12_BUILD_AUX) && \
    !defined(SAGEATTN_GFX12_BUILD_PREPARE) && \
    !defined(SAGEATTN_GFX12_BUILD_ATTN_F16) && \
    !defined(SAGEATTN_GFX12_BUILD_ATTN_FP8) && \
    !defined(SAGEATTN_GFX12_BUILD_RAWQ_FP8)
#define SAGEATTN_GFX12_BUILD_AUX 1
#define SAGEATTN_GFX12_BUILD_PREPARE 1
#define SAGEATTN_GFX12_BUILD_ATTN_F16 1
#define SAGEATTN_GFX12_BUILD_ATTN_FP8 1
#define SAGEATTN_GFX12_BUILD_RAWQ_FP8 1
#endif

namespace {

constexpr int kNHD = 0;
constexpr int kHND = 1;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kFp8SoftmaxOffset = 8.807f;
constexpr float kF16SoftmaxOffset = 0.0f;

#define SAGEATTN_NATIVE_HAS_GFX12_WMMA 1
#ifndef SAGEATTN_GFX12_NATIVE_WAVES_PER_EU_MAX
#define SAGEATTN_GFX12_NATIVE_WAVES_PER_EU_MAX 1
#endif
#ifndef SAGEATTN_GFX12_NATIVE_D64_2Q_WAVES_PER_EU_MAX
#define SAGEATTN_GFX12_NATIVE_D64_2Q_WAVES_PER_EU_MAX 2
#endif
#ifndef SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MAX
#define SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MAX 2
#endif
#ifndef SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MIN
#define SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MIN 1
#endif
#ifndef SAGEATTN_GFX12_NATIVE_D16_2Q_WAVES_PER_EU_MAX
#define SAGEATTN_GFX12_NATIVE_D16_2Q_WAVES_PER_EU_MAX 2
#endif
#ifndef SAGEATTN_GFX12_NATIVE_D128_2Q_WAVES_PER_EU_MAX
#define SAGEATTN_GFX12_NATIVE_D128_2Q_WAVES_PER_EU_MAX 1
#endif
#ifndef SAGEATTN_GFX12_NATIVE_F16_TV_PAD
#define SAGEATTN_GFX12_NATIVE_F16_TV_PAD 16
#endif
#ifndef SAGEATTN_GFX12_NATIVE_F16_2Q_MIN_BLOCKS
#define SAGEATTN_GFX12_NATIVE_F16_2Q_MIN_BLOCKS 1
#endif
#define SAGEATTN_NATIVE_WAVES_PER_EU \
  __attribute__((amdgpu_waves_per_eu(1, SAGEATTN_GFX12_NATIVE_WAVES_PER_EU_MAX)))
#define SAGEATTN_NATIVE_2Q_WAVES_PER_EU(HD_, CAUSAL_) \
  __attribute__((amdgpu_waves_per_eu( \
      ((HD_) == 64 && (CAUSAL_) ? SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MIN : 1), \
      ((HD_) == 16 ? SAGEATTN_GFX12_NATIVE_D16_2Q_WAVES_PER_EU_MAX : \
       ((HD_) == 64 ? ((CAUSAL_) ? SAGEATTN_GFX12_NATIVE_D64_2Q_CAUSAL_WAVES_PER_EU_MAX \
                                   : SAGEATTN_GFX12_NATIVE_D64_2Q_WAVES_PER_EU_MAX) \
                       : SAGEATTN_GFX12_NATIVE_D128_2Q_WAVES_PER_EU_MAX)))))
#define SAGEATTN_NATIVE_F16_2Q_LAUNCH_BOUNDS(BR_) \
  __launch_bounds__(BR_, SAGEATTN_GFX12_NATIVE_F16_2Q_MIN_BLOCKS)
#define SAGEATTN_F16_SCHED_BARRIER(MASK_) ((void)0)

using half8_vec = _Float16 __attribute__((ext_vector_type(8)));
using float8_vec = float __attribute__((ext_vector_type(8)));
using i16x8_vec = int16_t __attribute__((ext_vector_type(8)));
using i32x2_vec = int32_t __attribute__((ext_vector_type(2)));
using u32x4_vec = uint32_t __attribute__((ext_vector_type(4)));
using i32x8_vec = int32_t __attribute__((ext_vector_type(8)));

void hip_kernel_launch_check() {
  const hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
}

__device__ __forceinline__ float value_to_float(const __half value) {
  return __half2float(value);
}

__device__ __forceinline__ float value_to_float(const __hip_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __half value_from_float_half(const float value) {
  return __float2half_rn(value);
}

__device__ __forceinline__ __hip_bfloat16 value_from_float_bfloat16(const float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ int8_t float_to_int8_rn_gfx12(const float x) {
  int32_t rounded;
  asm volatile("v_cvt_i32_f32 %[dst], %[src]"
               : [dst] "=v"(rounded)
               : [src] "v"(x));
  rounded = rounded > 127 ? 127 : rounded;
  rounded = rounded < -128 ? -128 : rounded;
  return static_cast<int8_t>(rounded);
}

__device__ __forceinline__ int8_t float_to_int8_nearby_gfx12(const float x) {
  const float clipped = fminf(127.0f, fmaxf(-128.0f, nearbyintf(x)));
  return static_cast<int8_t>(clipped);
}

template <typename T, int HeadDim>
__global__ void quant_qk_int8_hnd_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    int8_t* __restrict__ query_out,
    int8_t* __restrict__ key_out,
    float* __restrict__ query_scale,
    float* __restrict__ key_scale,
    const int64_t batch,
    const int64_t q_heads,
    const int64_t kv_heads,
    const int64_t q_len,
    const int64_t kv_len,
    const int q_groups,
    const int k_groups) {
  constexpr int Threads = 256;
  __shared__ float shared_amax;

  const int group = blockIdx.x;
  const int head = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const bool is_q = group < q_groups;
  const int local_group = is_q ? group : group - q_groups;
  const int rows_per_group = is_q ? 32 : 64;
  const int64_t seq_len = is_q ? q_len : kv_len;
  const int64_t base_row = static_cast<int64_t>(local_group) * rows_per_group;
  const int active_heads = is_q ? static_cast<int>(q_heads) : static_cast<int>(kv_heads);
  if (b >= batch || head >= active_heads || base_row >= seq_len) {
    return;
  }

  const T* in = is_q ? query : key;
  int8_t* out = is_q ? query_out : key_out;
  float* scale_out = is_q ? query_scale : key_scale;
  const int64_t heads = is_q ? q_heads : kv_heads;
  const int scale_groups = is_q ? q_groups : k_groups;
  constexpr int PackElems = 8;
  static_assert((HeadDim % PackElems) == 0, "native quantization packs eight elements");
  const int packs = (rows_per_group * HeadDim) / PackElems;

  float local_amax = 0.0000001f;
  for (int pack = tid; pack < packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (seq < seq_len) {
      const int64_t off = ((static_cast<int64_t>(b) * heads + head) * seq_len + seq) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(in + off);
      const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        local_amax = fmaxf(local_amax, fabsf(value_to_float(values[i])));
      }
    }
  }
  const float block_amax = vllm::blockReduceMax(local_amax);
  if (tid == 0) {
    shared_amax = block_amax;
    scale_out[(static_cast<int64_t>(b) * active_heads + head) * scale_groups + local_group] =
        shared_amax / 127.0f;
  }
  __syncthreads();
  const float inv_scale = 127.0f / shared_amax;

  for (int pack = tid; pack < packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (seq < seq_len) {
      const int64_t off = ((static_cast<int64_t>(b) * heads + head) * seq_len + seq) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(in + off);
      const T* values = reinterpret_cast<const T*>(&raw);
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_rn_gfx12(value_to_float(values[0]) * inv_scale);
      out0.y = float_to_int8_rn_gfx12(value_to_float(values[1]) * inv_scale);
      out0.z = float_to_int8_rn_gfx12(value_to_float(values[2]) * inv_scale);
      out0.w = float_to_int8_rn_gfx12(value_to_float(values[3]) * inv_scale);
      out1.x = float_to_int8_rn_gfx12(value_to_float(values[4]) * inv_scale);
      out1.y = float_to_int8_rn_gfx12(value_to_float(values[5]) * inv_scale);
      out1.z = float_to_int8_rn_gfx12(value_to_float(values[6]) * inv_scale);
      out1.w = float_to_int8_rn_gfx12(value_to_float(values[7]) * inv_scale);
      *reinterpret_cast<char4*>(out + off) = out0;
      *reinterpret_cast<char4*>(out + off + 4) = out1;
    }
  }
}

template <typename T, int HeadDim>
__global__ void quant_q_nhd_per_warp_kernel(
    const T* __restrict__ query,
    int8_t* __restrict__ query_out,
    float* __restrict__ query_scale,
    const int64_t batch,
    const int64_t q_len,
    const int64_t q_heads,
    const int q_scale_groups) {
  constexpr int Threads = 256;
  constexpr int PackElems = 8;
  constexpr int QRows = 32;
  constexpr int GroupsPerBlock = 2;
  static_assert((HeadDim % PackElems) == 0, "native Q quantization packs eight elements");

  __shared__ float shared_amax[GroupsPerBlock];
  __shared__ float shared_pair_amax[GroupsPerBlock][32];

  const int group_base = static_cast<int>(blockIdx.x) * GroupsPerBlock;
  const int head = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  if (b >= batch || head >= q_heads ||
      static_cast<int64_t>(group_base) * QRows >= q_len) {
    return;
  }

  constexpr int Packs = (QRows * HeadDim) / PackElems;
  float local_amax0 = 0.0000001f;
  float local_amax1 = 0.0000001f;
  const bool has_group1 = (group_base + 1) < q_scale_groups;
  const int64_t base_row0 = static_cast<int64_t>(group_base) * QRows;
  const int64_t base_row1 = base_row0 + QRows;

  for (int pack = tid; pack < Packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq0 = base_row0 + row;
    if (seq0 < q_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * q_len + seq0) * q_heads + head) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
      const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        local_amax0 = fmaxf(local_amax0, fabsf(value_to_float(values[i])));
      }
    }
    const int64_t seq1 = base_row1 + row;
    if (has_group1 && seq1 < q_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * q_len + seq1) * q_heads + head) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
      const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        local_amax1 = fmaxf(local_amax1, fabsf(value_to_float(values[i])));
      }
    }
  }

  local_amax0 = vllm::warpReduceMax(local_amax0);
  local_amax1 = vllm::warpReduceMax(local_amax1);
  if (lane == 0) {
    shared_pair_amax[0][wid] = local_amax0;
    shared_pair_amax[1][wid] = local_amax1;
  }
  __syncthreads();

  const bool warp_lane_active = tid < (blockDim.x / 32);
  local_amax0 = warp_lane_active ? shared_pair_amax[0][lane] : -1e20f;
  local_amax1 = warp_lane_active ? shared_pair_amax[1][lane] : -1e20f;
  local_amax0 = vllm::warpReduceMax(local_amax0);
  local_amax1 = vllm::warpReduceMax(local_amax1);
  if (tid == 0) {
    shared_amax[0] = local_amax0;
    query_scale[(static_cast<int64_t>(b) * q_heads + head) * q_scale_groups +
                group_base] = local_amax0 / 127.0f;
    if (has_group1) {
      shared_amax[1] = local_amax1;
      query_scale[(static_cast<int64_t>(b) * q_heads + head) * q_scale_groups +
                  group_base + 1] = local_amax1 / 127.0f;
    }
  }
  __syncthreads();

  const float inv_scale0 = 127.0f / shared_amax[0];
  const float inv_scale1 = has_group1 ? (127.0f / shared_amax[1]) : 0.0f;
  for (int pack = tid; pack < Packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq0 = base_row0 + row;
    if (seq0 < q_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * q_len + seq0) * q_heads + head) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
      const T* values = reinterpret_cast<const T*>(&raw);
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_nearby_gfx12(value_to_float(values[0]) * inv_scale0);
      out0.y = float_to_int8_nearby_gfx12(value_to_float(values[1]) * inv_scale0);
      out0.z = float_to_int8_nearby_gfx12(value_to_float(values[2]) * inv_scale0);
      out0.w = float_to_int8_nearby_gfx12(value_to_float(values[3]) * inv_scale0);
      out1.x = float_to_int8_nearby_gfx12(value_to_float(values[4]) * inv_scale0);
      out1.y = float_to_int8_nearby_gfx12(value_to_float(values[5]) * inv_scale0);
      out1.z = float_to_int8_nearby_gfx12(value_to_float(values[6]) * inv_scale0);
      out1.w = float_to_int8_nearby_gfx12(value_to_float(values[7]) * inv_scale0);
      *reinterpret_cast<char4*>(query_out + off) = out0;
      *reinterpret_cast<char4*>(query_out + off + 4) = out1;
    }
    const int64_t seq1 = base_row1 + row;
    if (has_group1 && seq1 < q_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * q_len + seq1) * q_heads + head) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
      const T* values = reinterpret_cast<const T*>(&raw);
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_nearby_gfx12(value_to_float(values[0]) * inv_scale1);
      out0.y = float_to_int8_nearby_gfx12(value_to_float(values[1]) * inv_scale1);
      out0.z = float_to_int8_nearby_gfx12(value_to_float(values[2]) * inv_scale1);
      out0.w = float_to_int8_nearby_gfx12(value_to_float(values[3]) * inv_scale1);
      out1.x = float_to_int8_nearby_gfx12(value_to_float(values[4]) * inv_scale1);
      out1.y = float_to_int8_nearby_gfx12(value_to_float(values[5]) * inv_scale1);
      out1.z = float_to_int8_nearby_gfx12(value_to_float(values[6]) * inv_scale1);
      out1.w = float_to_int8_nearby_gfx12(value_to_float(values[7]) * inv_scale1);
      *reinterpret_cast<char4*>(query_out + off) = out0;
      *reinterpret_cast<char4*>(query_out + off + 4) = out1;
    }
  }
}

template <typename T, typename OutT, bool ToFp8>
__global__ void transpose_value_hnd_kernel(
    const T* __restrict__ value,
    OutT* __restrict__ output,
    const int64_t total_heads,
    const int64_t seq_len,
    const int64_t head_dim) {
  constexpr int TileS = 128;
  constexpr int TileD = 16;
  __shared__ OutT tile[TileS][TileD];

  const int tid = threadIdx.x;
  const int64_t bh = blockIdx.z;

  for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
    const int load_s = linear / TileD;
    const int load_d = linear - load_s * TileD;
    const int64_t s = static_cast<int64_t>(blockIdx.x) * TileS + load_s;
    const int64_t d = static_cast<int64_t>(blockIdx.y) * TileD + load_d;
    if (bh < total_heads && s < seq_len && d < head_dim) {
      const float v = value_to_float(value[(bh * seq_len + s) * head_dim + d]);
      if constexpr (ToFp8) {
        tile[load_s][load_d] =
            __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
      } else {
        tile[load_s][load_d] = __float2half_rn(v);
      }
    }
  }
  __syncthreads();

  for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
    const int store_d_local = linear / TileS;
    const int store_s_local = linear - store_d_local * TileS;
    const int64_t store_s = static_cast<int64_t>(blockIdx.x) * TileS + store_s_local;
    const int64_t store_d = static_cast<int64_t>(blockIdx.y) * TileD + store_d_local;
    if (bh < total_heads && store_s < seq_len && store_d < head_dim) {
      output[(bh * head_dim + store_d) * seq_len + store_s] =
          tile[store_s_local][store_d_local];
    }
  }
}

template <typename OutT, bool ToFp8>
torch::Tensor transpose_value_hnd_gfx12(torch::Tensor value) {
  TORCH_CHECK(value.is_cuda(), "gfx12 value transpose expects a CUDA/HIP tensor");
  TORCH_CHECK(value.dim() == 4, "gfx12 value transpose expects [B, H, S, D]");
  TORCH_CHECK(value.is_contiguous(), "gfx12 value transpose expects contiguous HND input");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16 || value.scalar_type() == torch::kBFloat16,
              "gfx12 value transpose supports fp16/bf16 input");

  const int64_t batch = value.size(0);
  const int64_t heads = value.size(1);
  const int64_t seq_len = value.size(2);
  const int64_t head_dim = value.size(3);
  auto out_options = value.options().dtype(ToFp8 ? torch::kUInt8 : torch::kFloat16);
  torch::Tensor output = torch::empty({batch, heads, head_dim, seq_len}, out_options);

  dim3 block(256);
  dim3 grid((seq_len + 127) / 128, (head_dim + 15) / 16, batch * heads);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == torch::kFloat16) {
    transpose_value_hnd_kernel<__half, OutT, ToFp8><<<grid, block, 0, stream>>>(
                       reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                       reinterpret_cast<OutT*>(output.data_ptr()),
                       batch * heads, seq_len, head_dim);
  } else {
    transpose_value_hnd_kernel<__hip_bfloat16, OutT, ToFp8><<<grid, block, 0, stream>>>(
                       reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                       reinterpret_cast<OutT*>(output.data_ptr()),
                       batch * heads, seq_len, head_dim);
  }
  hip_kernel_launch_check();
  return output;
}

template <typename T>
__global__ void transpose_value_fp8_scaled_hnd_kernel(
    const T* __restrict__ value,
    const float* __restrict__ value_scale,
    uint8_t* __restrict__ output,
    const int64_t total_heads,
    const int64_t seq_len,
    const int64_t head_dim) {
  constexpr int TileS = 128;
  constexpr int TileD = 16;
  __shared__ uint8_t tile[TileS][TileD];

  const int tid = threadIdx.x;
  const int64_t bh = blockIdx.z;

  for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
    const int load_s = linear / TileD;
    const int load_d = linear - load_s * TileD;
    const int64_t s = static_cast<int64_t>(blockIdx.x) * TileS + load_s;
    const int64_t d = static_cast<int64_t>(blockIdx.y) * TileD + load_d;
    if (bh < total_heads && s < seq_len && d < head_dim) {
      const float scale = value_scale[bh * head_dim + d];
      const float v = scale == 0.0f ? 0.0f :
          value_to_float(value[(bh * seq_len + s) * head_dim + d]) / scale;
      tile[load_s][load_d] =
          __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
    }
  }
  __syncthreads();

  for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
    const int store_d_local = linear / TileS;
    const int store_s_local = linear - store_d_local * TileS;
    const int64_t store_s = static_cast<int64_t>(blockIdx.x) * TileS + store_s_local;
    const int64_t store_d = static_cast<int64_t>(blockIdx.y) * TileD + store_d_local;
    if (bh < total_heads && store_s < seq_len && store_d < head_dim) {
      output[(bh * head_dim + store_d) * seq_len + store_s] =
          tile[store_s_local][store_d_local];
    }
  }
}

template <typename T>
__global__ void fp8_value_nhd_short_kernel(
    const T* __restrict__ value,
    uint8_t* __restrict__ output,
    float* __restrict__ value_scale,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim,
    const float scale_max) {
  constexpr int TileS = 128;
  constexpr int TileD = 16;
  __shared__ float partial_amax[256];
  __shared__ float scale_tile[TileD];
  __shared__ uint8_t tile[TileS][TileD];

  const int tid = threadIdx.x;
  const int d_local = tid & (TileD - 1);
  const int s_lane = tid >> 4;
  const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
  const int64_t h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t d = d_base + d_local;

  float local_amax = 0.0f;
  if (d < head_dim) {
    for (int64_t s = s_lane; s < seq_len; s += 16) {
      const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + d;
      local_amax = fmaxf(local_amax, fabsf(value_to_float(value[offset])));
    }
  }
  partial_amax[tid] = local_amax;
  __syncthreads();

  if (tid < TileD) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
      amax = fmaxf(amax, partial_amax[i * TileD + tid]);
    }
    const float scale = amax / scale_max;
    scale_tile[tid] = scale;
    const int64_t scale_d = d_base + tid;
    if (scale_d < head_dim) {
      value_scale[(b * heads + h) * head_dim + scale_d] = scale;
    }
  }
  __syncthreads();

  for (int64_t s_base = 0; s_base < seq_len; s_base += TileS) {
    for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
      const int load_s = linear / TileD;
      const int load_d = linear - load_s * TileD;
      const int64_t s = s_base + load_s;
      const int64_t value_d = d_base + load_d;
      uint8_t packed = 0;
      if (s < seq_len && value_d < head_dim) {
        const float scale = scale_tile[load_d];
        const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + value_d;
        const float v = scale == 0.0f ? 0.0f : value_to_float(value[offset]) / scale;
        packed = __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
      }
      tile[load_s][load_d] = packed;
    }
    __syncthreads();

    for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
      const int store_d_local = linear / TileS;
      const int store_s_local = linear - store_d_local * TileS;
      const int64_t s = s_base + store_s_local;
      const int64_t value_d = d_base + store_d_local;
      if (s < seq_len && value_d < head_dim) {
        output[((b * heads + h) * head_dim + value_d) * seq_len + s] =
            tile[store_s_local][store_d_local];
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void mean_nhd_kernel(
    const T* __restrict__ input,
    T* __restrict__ mean,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim) {
  constexpr int TileD = 16;
  __shared__ float partial_sum[256];

  const int tid = threadIdx.x;
  const int d_local = tid & (TileD - 1);
  const int s_lane = tid >> 4;
  const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
  const int64_t h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t d = d_base + d_local;

  float local_sum = 0.0f;
  if (d < head_dim) {
    for (int64_t s = s_lane; s < seq_len; s += 16) {
      const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + d;
      local_sum += value_to_float(input[offset]);
    }
  }
  partial_sum[tid] = local_sum;
  __syncthreads();

  if (tid < TileD) {
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
      sum += partial_sum[i * TileD + tid];
    }
    const int64_t mean_d = d_base + tid;
    if (mean_d < head_dim) {
      const float value = sum / static_cast<float>(seq_len);
      if constexpr (std::is_same<T, __half>::value) {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_half(value);
      } else {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_bfloat16(value);
      }
    }
  }
}

template <typename T, int TileD, int SeqLanes>
__global__ void mean_nhd_short_kernel(
    const T* __restrict__ input,
    T* __restrict__ mean,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim) {
  __shared__ float partial_sum[TileD * SeqLanes];

  const int tid = threadIdx.x;
  const int d_local = tid & (TileD - 1);
  const int s_lane = tid / TileD;
  const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
  const int64_t h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t d = d_base + d_local;

  float local_sum = 0.0f;
  if (d < head_dim) {
    for (int64_t s = s_lane; s < seq_len; s += SeqLanes) {
      const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + d;
      local_sum += value_to_float(input[offset]);
    }
  }
  partial_sum[tid] = local_sum;
  __syncthreads();

  if (tid < TileD) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < SeqLanes; ++i) {
      sum += partial_sum[i * TileD + tid];
    }
    const int64_t mean_d = d_base + tid;
    if (mean_d < head_dim) {
      const float value = sum / static_cast<float>(seq_len);
      if constexpr (std::is_same<T, __half>::value) {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_half(value);
      } else {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_bfloat16(value);
      }
    }
  }
}

template <typename T>
__global__ void mean_hnd_kernel(
    const T* __restrict__ input,
    T* __restrict__ mean,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim) {
  constexpr int TileD = 16;
  __shared__ float partial_sum[256];

  const int tid = threadIdx.x;
  const int d_local = tid & (TileD - 1);
  const int s_lane = tid >> 4;
  const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
  const int64_t h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t d = d_base + d_local;

  float local_sum = 0.0f;
  if (d < head_dim) {
    for (int64_t s = s_lane; s < seq_len; s += 16) {
      const int64_t offset = ((b * heads + h) * seq_len + s) * head_dim + d;
      local_sum += value_to_float(input[offset]);
    }
  }
  partial_sum[tid] = local_sum;
  __syncthreads();

  if (tid < TileD) {
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
      sum += partial_sum[i * TileD + tid];
    }
    const int64_t mean_d = d_base + tid;
    if (mean_d < head_dim) {
      const float value = sum / static_cast<float>(seq_len);
      if constexpr (std::is_same<T, __half>::value) {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_half(value);
      } else {
        mean[(b * heads + h) * head_dim + mean_d] = value_from_float_bfloat16(value);
      }
    }
  }
}

__device__ __forceinline__ int32_t pack_f32x4_to_ocp_fp8(
    const float x0,
    const float x1,
    const float x2,
    const float x3);

template <typename T, int SeqLanes>
__global__ void mean_and_fp8_value_nhd_short_kernel(
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ key_mean,
    uint8_t* __restrict__ output,
    float* __restrict__ value_scale,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim,
    const float scale_max) {
  constexpr int TileS = 128;
  constexpr int TileD = 32;
  __shared__ float partial_sum[TileD * SeqLanes];
  __shared__ float partial_amax[TileD * SeqLanes];
  __shared__ float scale_tile[TileD];
  __shared__ uint8_t tile[TileS][TileD];

  const int tid = threadIdx.x;
  const int d_local = tid & (TileD - 1);
  const int s_lane = tid / TileD;
  const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
  const int64_t h = blockIdx.y;
  const int64_t b = blockIdx.z;
  const int64_t d = d_base + d_local;

  float local_sum = 0.0f;
  float local_amax = 0.0f;
  if (d < head_dim) {
    for (int64_t s = s_lane; s < seq_len; s += SeqLanes) {
      const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + d;
      local_sum += value_to_float(key[offset]);
      local_amax = fmaxf(local_amax, fabsf(value_to_float(value[offset])));
    }
  }
  partial_sum[tid] = local_sum;
  partial_amax[tid] = local_amax;
  __syncthreads();

  if (tid < TileD) {
    float sum = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < SeqLanes; ++i) {
      const int partial_idx = i * TileD + tid;
      sum += partial_sum[partial_idx];
      amax = fmaxf(amax, partial_amax[partial_idx]);
    }
    const int64_t value_d = d_base + tid;
    if (value_d < head_dim) {
      const float mean = sum / static_cast<float>(seq_len);
      const int64_t mean_offset = (b * heads + h) * head_dim + value_d;
      if constexpr (std::is_same<T, __half>::value) {
        key_mean[mean_offset] = value_from_float_half(mean);
      } else {
        key_mean[mean_offset] = value_from_float_bfloat16(mean);
      }
      const float scale = amax / scale_max;
      scale_tile[tid] = scale == 0.0f ? 0.0f : 1.0f / scale;
      value_scale[mean_offset] = scale;
    }
  }
  __syncthreads();

  for (int64_t s_base = 0; s_base < seq_len; s_base += TileS) {
    constexpr int PackElems = 4;
    constexpr int PacksPerRow = TileD / PackElems;
    for (int pack = tid; pack < TileS * PacksPerRow; pack += blockDim.x) {
      const int load_s = pack / PacksPerRow;
      const int load_d = (pack - load_s * PacksPerRow) * PackElems;
      const int64_t s = s_base + load_s;
      const int64_t value_d = d_base + load_d;
      if (s < seq_len && value_d + PackElems - 1 < head_dim) {
        const int64_t offset = ((b * seq_len + s) * heads + h) * head_dim + value_d;
        const float scale0 = scale_tile[load_d + 0];
        const float scale1 = scale_tile[load_d + 1];
        const float scale2 = scale_tile[load_d + 2];
        const float scale3 = scale_tile[load_d + 3];
        const float v0 = value_to_float(value[offset + 0]) * scale0;
        const float v1 = value_to_float(value[offset + 1]) * scale1;
        const float v2 = value_to_float(value[offset + 2]) * scale2;
        const float v3 = value_to_float(value[offset + 3]) * scale3;
        const uint32_t packed = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(v0, v1, v2, v3));
        *reinterpret_cast<uint32_t*>(&tile[load_s][load_d]) = packed;
      } else {
#pragma unroll
        for (int i = 0; i < PackElems; ++i) {
          const int elem_d = load_d + i;
          uint8_t packed = 0;
          if (s < seq_len && d_base + elem_d < head_dim) {
            const float scale = scale_tile[elem_d];
            const int64_t offset =
                ((b * seq_len + s) * heads + h) * head_dim + d_base + elem_d;
            const float v = value_to_float(value[offset]) * scale;
            packed = __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
          }
          tile[load_s][elem_d] = packed;
        }
      }
    }
    __syncthreads();

    for (int linear = tid; linear < TileS * TileD; linear += blockDim.x) {
      const int store_d_local = linear / TileS;
      const int store_s_local = linear - store_d_local * TileS;
      const int64_t s = s_base + store_s_local;
      const int64_t value_d = d_base + store_d_local;
      if (s < seq_len && value_d < head_dim) {
        output[((b * heads + h) * head_dim + value_d) * seq_len + s] =
            tile[store_s_local][store_d_local];
      }
    }
    __syncthreads();
  }
}

template <typename T, int HeadDim, int NumPackPerThread>
__global__ void quant_k_nhd_fuse_sub_mean_short_kernel(
    const T* __restrict__ key,
    const T* __restrict__ mean,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    const int64_t seq_len,
    const int64_t heads) {
  static_assert(HeadDim == 64 || HeadDim == 128,
                "short NHD smooth-K quant supports D64/D128");
  static_assert(NumPackPerThread == 1 || NumPackPerThread == 2,
                "short NHD smooth-K quant supports pack1/pack2");
  constexpr int BlockSize = 64;
  constexpr int PackElems = 8;
  constexpr int ThreadsPerToken = HeadDim / PackElems;
  constexpr int IterStride = BlockSize / NumPackPerThread;

  T x_val[NumPackPerThread][PackElems];
  T mean_val[PackElems];
  float x_float[NumPackPerThread][PackElems];
  float mean_float[PackElems];

  const int k_block = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int local_token = tid / ThreadsPerToken;
  const int d = (tid % ThreadsPerToken) * PackElems;
  const int64_t token_base = static_cast<int64_t>(k_block) * BlockSize + local_token;
  const int64_t mean_off = (static_cast<int64_t>(b) * heads + h) * HeadDim + d;

  *reinterpret_cast<uint4*>(&mean_val[0]) =
      *reinterpret_cast<const uint4*>(mean + mean_off);
#pragma unroll
  for (int i = 0; i < PackElems; ++i) {
    mean_float[i] = value_to_float(mean_val[i]);
  }

  float local_amax = 0.0000001f;
#pragma unroll
  for (int pack = 0; pack < NumPackPerThread; ++pack) {
    const int64_t s = token_base + static_cast<int64_t>(pack) * IterStride;
    if (s < seq_len) {
      const int64_t off = ((static_cast<int64_t>(b) * seq_len + s) * heads + h) * HeadDim + d;
      *reinterpret_cast<uint4*>(&x_val[pack][0]) =
          *reinterpret_cast<const uint4*>(key + off);
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        const float centered = value_to_float(x_val[pack][i]) - mean_float[i];
        x_float[pack][i] = centered;
        local_amax = fmaxf(local_amax, fabsf(centered));
      }
    } else {
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        x_float[pack][i] = 0.0f;
      }
    }
  }

  __shared__ float shared_amax;
  const float block_amax = vllm::blockReduceMax(local_amax);
  if (tid == 0) {
    shared_amax = block_amax;
    scale[(static_cast<int64_t>(b) * heads + h) * ((seq_len + 63) / 64) + k_block] =
        block_amax / 127.0f;
  }
  __syncthreads();
  const float inv_scale = 127.0f / shared_amax;

#pragma unroll
  for (int pack = 0; pack < NumPackPerThread; ++pack) {
    const int64_t s = token_base + static_cast<int64_t>(pack) * IterStride;
    if (s < seq_len) {
      const int64_t off = ((static_cast<int64_t>(b) * seq_len + s) * heads + h) * HeadDim + d;
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_nearby_gfx12(x_float[pack][0] * inv_scale);
      out0.y = float_to_int8_nearby_gfx12(x_float[pack][1] * inv_scale);
      out0.z = float_to_int8_nearby_gfx12(x_float[pack][2] * inv_scale);
      out0.w = float_to_int8_nearby_gfx12(x_float[pack][3] * inv_scale);
      out1.x = float_to_int8_nearby_gfx12(x_float[pack][4] * inv_scale);
      out1.y = float_to_int8_nearby_gfx12(x_float[pack][5] * inv_scale);
      out1.z = float_to_int8_nearby_gfx12(x_float[pack][6] * inv_scale);
      out1.w = float_to_int8_nearby_gfx12(x_float[pack][7] * inv_scale);
      *reinterpret_cast<char4*>(output + off) = out0;
      *reinterpret_cast<char4*>(output + off + 4) = out1;
    }
  }
}

__device__ __forceinline__ int64_t qkv_offset(
    const int tensor_layout,
    const int64_t b,
    const int64_t h,
    const int64_t n,
    const int64_t d,
    const int64_t stride_b,
    const int64_t stride_n,
    const int64_t stride_h) {
  return tensor_layout == kNHD
      ? b * stride_b + n * stride_n + h * stride_h + d
      : b * stride_b + h * stride_h + n * stride_n + d;
}

template <int HeadDim, bool HndContiguous, bool StaticNhd = false>
__device__ __forceinline__ int64_t qkv_offset_dispatch(
    const int tensor_layout,
    const int64_t b,
    const int64_t h,
    const int64_t n,
    const int64_t d,
    const int64_t stride_b,
    const int64_t stride_n,
    const int64_t stride_h) {
  if constexpr (HndContiguous) {
    return b * stride_b + h * stride_h + n * HeadDim + d;
  } else if constexpr (StaticNhd) {
    return b * stride_b + n * stride_n + h * stride_h + d;
  } else {
    return qkv_offset(tensor_layout, b, h, n, d, stride_b, stride_n, stride_h);
  }
}

__device__ __forceinline__ int q_scale_col_per_warp(const int64_t q_idx) {
  return static_cast<int>((q_idx / 128) * 4 + ((q_idx & 127) / 32));
}

__device__ __forceinline__ int k_scale_col_per_warp(const int64_t k_idx) {
  return static_cast<int>(k_idx / 64);
}

__device__ __forceinline__ int wmma_f16_k_for_lane_elem(
    const int lane,
    const int elem);

__device__ __forceinline__ int ceil_div_i64_to_int(
    const int64_t value,
    const int64_t divisor) {
  return static_cast<int>((value + divisor - 1) / divisor);
}

__device__ __forceinline__ int q_scale_col_per_thread(
    const int64_t q_idx,
    const int64_t qo_len,
    const int64_t q_scale_groups) {
  const int q_blocks = ceil_div_i64_to_int(qo_len, 128);
  const int groups_per_128 = q_blocks > 0 ?
      static_cast<int>(q_scale_groups / q_blocks) : 32;
  const int warp_q = groups_per_128 >= 64 ? 16 : 32;
  return static_cast<int>((q_idx / warp_q) * 8 + (q_idx & 7));
}

__device__ __forceinline__ int k_scale_col_per_thread(
    const int64_t k_idx,
    const int64_t kv_len,
    const int64_t k_scale_groups) {
  const int k_blocks64 = ceil_div_i64_to_int(kv_len, 64);
  const int groups_per_64 = k_blocks64 > 0 ?
      static_cast<int>(k_scale_groups / k_blocks64) : 4;
  const int warp_k = groups_per_64 <= 2 ? 128 : 64;
  return static_cast<int>((k_idx / warp_k) * 4 + ((k_idx & 7) >> 1));
}

template <bool PerThreadQK = false, bool PvOrdered = false>
__device__ __forceinline__ float qk_score_scale_scalar(
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t b,
    const int64_t hq,
    const int64_t hkv,
    const int64_t q_start,
    const int64_t kb_base,
    const int col_tile,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const float sm_scale) {
  if constexpr (PerThreadQK) {
    return 1.0f;
  } else {
    const int q_scale_idx = q_scale_col_per_warp(q_start);
    const int k_scale_idx = k_scale_col_per_warp(kb_base + col_tile * 16);
    return q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx] *
        k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx] *
        sm_scale * kLog2e;
  }
}

template <bool PerThreadQK = false, bool PvOrdered = false>
__device__ __forceinline__ void apply_per_thread_qk_score_scale(
    float8_vec& scores,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t b,
    const int64_t hq,
    const int64_t hkv,
    const int64_t q_start,
    const int64_t kb_base,
    const int col_tile,
    const int lane,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const float sm_scale) {
  if constexpr (PerThreadQK) {
    const int64_t q_idx = q_start + (lane & 15);
    const int q_scale_idx = q_scale_col_per_thread(q_idx, qo_len, qs_stride_h);
    const float q_scale_local =
        q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx] *
        sm_scale * kLog2e;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int k_local = PvOrdered ?
          wmma_f16_k_for_lane_elem(lane, elem) :
          (((lane >> 4) << 3) + elem);
      const int64_t k_idx = kb_base + col_tile * 16 + k_local;
      const int k_scale_idx = k_scale_col_per_thread(k_idx, kv_len, ks_stride_h);
      scores[elem] *=
          q_scale_local *
          k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
    }
  }
}

template <bool IsCausal, int BlockRows>
__device__ __forceinline__ int64_t q_block_base_for_launch(
    const int64_t block_x,
    const int64_t qo_len) {
  if constexpr (IsCausal) {
    const int64_t q_blocks = (qo_len + BlockRows - 1) / BlockRows;
    return (q_blocks - 1 - block_x) * BlockRows;
  } else {
    return block_x * BlockRows;
  }
}

__device__ __forceinline__ float fast_exp2(float x) {
  return __builtin_amdgcn_exp2f(x);
}

template <bool IsCausal>
__device__ __forceinline__ void apply_tqk_causal_mask(
    float8_vec& scores,
    const int q_start,
    const int kb_base,
    const int col_tile,
    const int lane) {
  if constexpr (IsCausal) {
    const int tile_end = kb_base + col_tile * 16 + 15;
    if (tile_end <= q_start) {
      return;
    }
    const int q_idx = q_start + (lane & 15);
    const int k_base = kb_base + col_tile * 16 + ((lane >> 4) << 3);
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      scores[elem] = (k_base + elem) > q_idx ? -FLT_MAX * 0.5f : scores[elem];
    }
  }
}

template <bool ApplyCausalMask, int ColTiles>
__device__ __forceinline__ int active_causal_col_tiles(
    const int64_t q_start,
    const int64_t kb_base) {
  if constexpr (!ApplyCausalMask) {
    return ColTiles;
  } else {
    constexpr int BK = 16;
    constexpr int RM = 16;
    const int64_t q_end = q_start + RM;
    if (q_end <= kb_base) {
      return 0;
    }
    const int64_t cols = (q_end - kb_base + BK - 1) / BK;
    return static_cast<int>(cols < ColTiles ? cols : ColTiles);
  }
}

__device__ __forceinline__ int wmma_f16_k_for_lane_elem(
    const int lane,
    const int elem) {
  const int reg = elem >> 1;
  const int half = elem & 1;
  return ((reg >> 1) << 3) + (((lane >> 4) & 1) << 2) + ((reg & 1) << 1) + half;
}

template <bool PvOrdered = false>
__device__ __forceinline__ void apply_tqk_kv_tail_mask(
    float8_vec& scores,
    const int64_t valid_kv_len,
    const int64_t kb_base,
    const int col_tile,
    const int lane) {
  const int64_t col_base = kb_base + static_cast<int64_t>(col_tile) * 16;
  if (col_base + 15 < valid_kv_len) {
    return;
  }
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    const int k_local = PvOrdered ?
        wmma_f16_k_for_lane_elem(lane, elem) :
        (((lane >> 4) << 3) + elem);
    scores[elem] = (col_base + k_local) >= valid_kv_len ?
        -FLT_MAX * 0.5f : scores[elem];
  }
}

__device__ __forceinline__ int gfx12_tr_b128_source_row_for_lane(const int lane) {
  const int quad_group = lane >> 2;
  const int quad_pos = lane & 3;
  return quad_pos + ((quad_group & 1) << 3) + ((lane >> 4) << 2);
}

__device__ __forceinline__ int gfx12_tr_b128_source_col_for_lane(const int lane) {
  return ((lane >> 3) & 1) << 3;
}

__device__ __forceinline__ half8_vec gfx12_global_load_tr_b128_f16(const __half* ptr) {
#if defined(__gfx1200__) || defined(__gfx1201__)
  const i16x8_vec bits = __builtin_amdgcn_global_load_tr_b128_v8i16(
      reinterpret_cast<i16x8_vec*>(const_cast<__half*>(ptr)));
  return *reinterpret_cast<const half8_vec*>(&bits);
#else
  half8_vec regs;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    regs[elem] = static_cast<_Float16>(ptr[elem]);
  }
  return regs;
#endif
}

__device__ __forceinline__ int pv_k_order_for_acc_row(const int row) {
  return (row & 3) | ((row & 4) << 1) | ((row & 8) >> 1);
}

template <bool IsCausal>
__device__ __forceinline__ void apply_tqk_causal_mask_pv_order(
    float8_vec& scores,
    const int q_start,
    const int kb_base,
    const int col_tile,
    const int lane) {
  if constexpr (IsCausal) {
    const int tile_end = kb_base + col_tile * 16 + 15;
    if (tile_end <= q_start) {
      return;
    }
    const int q_idx = q_start + (lane & 15);
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int k_idx = kb_base + col_tile * 16 + wmma_f16_k_for_lane_elem(lane, elem);
      scores[elem] = k_idx > q_idx ? -FLT_MAX * 0.5f : scores[elem];
    }
  }
}

template <typename QueryT,
          int HeadDim,
          bool HndContiguous,
          bool StaticNhd = false,
          bool NoQueryTail = false>
__device__ __forceinline__ i32x2_vec pack_quant_q_i8_wmma_b_regs(
    const QueryT* __restrict__ q,
    const int tensor_layout,
    const int lane,
    const int64_t b,
    const int64_t h,
    const int64_t q_start,
    const int64_t qo_len,
    const int d_base,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const float inv_q_scale) {
  i32x2_vec regs;
  const int row = lane & 15;
  const int k_base = 8 * (lane >> 4);
  const int64_t q_idx = q_start + row;
  if constexpr (!NoQueryTail) if (q_idx >= qo_len) {
#pragma unroll
    for (int gpr = 0; gpr < 2; ++gpr) {
      regs[gpr] = 0;
    }
    return regs;
  }

  const int d = d_base + k_base;
  const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhd>(
      tensor_layout, b, h, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
  const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
  const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    uint32_t packed = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      packed |= static_cast<uint32_t>(static_cast<unsigned char>(
          float_to_int8_nearby_gfx12(value_to_float(values[4 * gpr + byte]) * inv_q_scale)))
          << (8 * byte);
    }
    regs[gpr] = static_cast<int32_t>(packed);
  }
  return regs;
}

template <int HeadDim,
          bool HndContiguous,
          bool StaticNhd = false,
          bool NoQueryTail = false>
__device__ __forceinline__ i32x2_vec pack_q_i8_wmma_b_regs(
    const int8_t* __restrict__ q,
    const int tensor_layout,
    const int lane,
    const int64_t b,
    const int64_t h,
    const int64_t q_start,
    const int64_t qo_len,
    const int d_base,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h) {
  i32x2_vec regs;
  const int row = lane & 15;
  const int k_base = 8 * (lane >> 4);
  const int64_t q_idx = q_start + row;
  if constexpr (!NoQueryTail) if (q_idx >= qo_len) {
#pragma unroll
    for (int gpr = 0; gpr < 2; ++gpr) {
      regs[gpr] = 0;
    }
    return regs;
  }

  const int d = d_base + k_base;
  const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhd>(
      tensor_layout, b, h, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
  const uint2 raw = *reinterpret_cast<const uint2*>(q + q_off);
  regs[0] = static_cast<int32_t>(raw.x);
  regs[1] = static_cast<int32_t>(raw.y);
  return regs;
}

template <int SharedHeadStride>
__device__ __forceinline__ i32x2_vec pack_k_i8_wmma_b_regs_from_shared(
    const int8_t* __restrict__ k_tile,
    const int lane,
    const int col_tile,
    const int d_base) {
  i32x2_vec regs;
  const int col = lane & 15;
  const int k_base = 8 * (lane >> 4);
  const int row = col_tile * 16 + col;
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    const int d = d_base + k_base + 4 * gpr;
    regs[gpr] = *reinterpret_cast<const int32_t*>(
        k_tile + row * SharedHeadStride + d);
  }
  return regs;
}

template <int SharedHeadStride>
__device__ __forceinline__ i32x2_vec pack_k_i8_wmma_b_regs_from_shared_pv_order(
    const int8_t* __restrict__ k_tile,
    const int lane,
    const int col_tile,
    const int d_base) {
  i32x2_vec regs;
  const int col = lane & 15;
  const int k_base = 8 * (lane >> 4);
  const int row = col_tile * 16 + pv_k_order_for_acc_row(col);
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    const int d = d_base + k_base + 4 * gpr;
    regs[gpr] = *reinterpret_cast<const int32_t*>(
        k_tile + row * SharedHeadStride + d);
  }
  return regs;
}

template <int DTiles>
__device__ __forceinline__ i32x2_vec pack_k_i8_wmma_b_regs_from_lane_major_global(
    const int8_t* __restrict__ k,
    const int64_t k_head_base,
    const int64_t k_group_stride,
    const int64_t kb_base,
    const int col_tile,
    const int d_tile,
    const int lane) {
  const int64_t group = kb_base >> 6;
  const int group_col_tile = ((static_cast<int>(kb_base) & 63) >> 4) + col_tile;
  const int64_t off = k_head_base + group * k_group_stride +
      (((static_cast<int64_t>(group_col_tile) * DTiles + d_tile) * 32 + lane) * 8);
  const uint2 raw = *reinterpret_cast<const uint2*>(k + off);
  i32x2_vec regs;
  regs[0] = static_cast<int32_t>(raw.x);
  regs[1] = static_cast<int32_t>(raw.y);
  return regs;
}

template <int DTiles>
__device__ __forceinline__ i32x2_vec pack_k_i8_wmma_b_regs_from_lane_major_shared(
    const uint2* __restrict__ k_lane_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  const uint2 raw = k_lane_tile[(col_tile * DTiles + d_tile) * 32 + lane];
  i32x2_vec regs;
  regs[0] = static_cast<int32_t>(raw.x);
  regs[1] = static_cast<int32_t>(raw.y);
  return regs;
}

__device__ __forceinline__ int wmma_fp8_k_for_lane_byte(
    const int lane,
    const int gpr,
    const int byte) {
  return 8 * (lane >> 4) + 4 * gpr + byte;
}

__device__ __forceinline__ int32_t pack_f32x4_to_ocp_fp8(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
  float a0 = x0;
  float a1 = x1;
  float a2 = x2;
  float a3 = x3;
  uint32_t packed;
  asm volatile("v_cvt_pk_fp8_f32 %[dst], %[a0], %[a1]\n"
               "v_cvt_pk_fp8_f32 %[dst], %[a2], %[a3], op_sel:[0, 0, 1]\n"
               : [dst] "=v"(packed), [a0] "+v"(a0), [a1] "+v"(a1),
                 [a2] "+v"(a2), [a3] "+v"(a3));
  return static_cast<int32_t>(packed);
}

__device__ __forceinline__ uint16_t half_to_u16_bits(const _Float16 value) {
  return __builtin_bit_cast(uint16_t, value);
}

__device__ __forceinline__ _Float16 u16_bits_to_half(const uint16_t value) {
  return __builtin_bit_cast(_Float16, value);
}

__device__ __forceinline__ i32x2_vec make_p_fp8_regs_from_tqk_prob_regs(
    const float8_vec prob_values,
    const int lane) {
  (void)lane;
  i32x2_vec regs;
  regs[0] = pack_f32x4_to_ocp_fp8(
      prob_values[0], prob_values[1], prob_values[2], prob_values[3]);
  regs[1] = pack_f32x4_to_ocp_fp8(
      prob_values[4], prob_values[5], prob_values[6], prob_values[7]);
  return regs;
}

__device__ __forceinline__ half8_vec make_p_regs_from_tqk_prob_regs(
    const half8_vec prob_values,
    const int lane) {
  half8_vec regs;
  const bool lane_upper = lane >= 16;
  u32x4_vec local_values;
  u32x4_vec peer_values;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t lo = half_to_u16_bits(prob_values[2 * i]);
    const uint32_t hi = half_to_u16_bits(prob_values[2 * i + 1]);
    local_values[i] = lo | (hi << 16);
    peer_values[i] = __shfl_xor(local_values[i], 16, 32);
  }
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    const int p_col = wmma_f16_k_for_lane_elem(lane, elem);
    const int source_elem = p_col & 7;
    const bool source_upper = p_col >= 8;
    const uint32_t packed = source_upper == lane_upper
        ? local_values[source_elem >> 1]
        : peer_values[source_elem >> 1];
    const uint16_t bits = static_cast<uint16_t>(packed >> (16 * (source_elem & 1)));
    regs[elem] = u16_bits_to_half(bits);
  }
  return regs;
}

template <int SharedHeadStride>
__device__ __forceinline__ i32x2_vec make_v_fp8_regs_from_shared(
    const uint8_t* __restrict__ v_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  i32x2_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    uint32_t packed = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      const int k_local = wmma_fp8_k_for_lane_byte(lane, gpr, byte);
      const int row = col_tile * BK + k_local;
      packed |= static_cast<uint32_t>(v_tile[row * SharedHeadStride + d]) << (8 * byte);
    }
    regs[gpr] = static_cast<int32_t>(packed);
  }
  return regs;
}

template <int SharedValueStride32>
__device__ __forceinline__ i32x2_vec make_v_fp8_regs_from_transposed_shared(
    const uint32_t* __restrict__ v_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  i32x2_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    const int k_local = wmma_fp8_k_for_lane_byte(lane, gpr, 0);
    const int n_group = (col_tile * BK + k_local) >> 2;
    regs[gpr] = static_cast<int32_t>(v_tile[d * SharedValueStride32 + n_group]);
  }
  return regs;
}

template <int SharedValueStride>
__device__ __forceinline__ i32x2_vec make_v_fp8_regs_from_transposed_shared(
    const uint8_t* __restrict__ v_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  i32x2_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int gpr = 0; gpr < 2; ++gpr) {
    uint32_t packed = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      const int k_local = wmma_fp8_k_for_lane_byte(lane, gpr, byte);
      const int n = col_tile * BK + k_local;
      packed |= static_cast<uint32_t>(v_tile[d * SharedValueStride + n]) << (8 * byte);
    }
    regs[gpr] = static_cast<int32_t>(packed);
  }
  return regs;
}

template <int DTiles>
__device__ __forceinline__ i32x2_vec make_v_fp8_regs_from_lane_major_global(
    const uint8_t* __restrict__ v,
    const int64_t v_head_base,
    const int64_t kb_base,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int ColTiles64 = 4;
  const int64_t group = kb_base >> 6;
  const int group_col_tile = ((static_cast<int>(kb_base) & 63) >> 4) + col_tile;
  const int64_t off = v_head_base +
      group * (ColTiles64 * DTiles * 32 * 8) +
      (((static_cast<int64_t>(group_col_tile) * DTiles + d_tile) * 32 + lane) * 8);
  const uint2 raw = *reinterpret_cast<const uint2*>(v + off);
  i32x2_vec regs;
  regs[0] = static_cast<int32_t>(raw.x);
  regs[1] = static_cast<int32_t>(raw.y);
  return regs;
}

template <
    int DTiles,
    int SharedHeadStride,
    typename FragK,
    typename FragQ,
    typename FragScoreT>
__device__ __forceinline__ float8_vec compute_tqk_score_regs(
    const int8_t* __restrict__ k_tile,
    const FragQ (&q_frag)[DTiles],
    const int col_tile,
    const float score_scale) {
  constexpr int BK = 16;
  FragScoreT score_acc;
  rocwmma::fill_fragment(score_acc, 0);
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    FragK k_frag;
    rocwmma::load_matrix_sync(
        k_frag,
        &k_tile[(col_tile * BK) * SharedHeadStride + dt * BK],
        static_cast<uint32_t>(SharedHeadStride));
    rocwmma::mma_sync(score_acc, k_frag, q_frag[dt], score_acc);
  }

  const auto score_rm = rocwmma::apply_data_layout<rocwmma::row_major>(score_acc);
  float8_vec scores;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores[elem] = static_cast<float>(score_rm[elem]) * score_scale;
  }
  return scores;
}

template <int DTiles, int SharedHeadStride>
__device__ __forceinline__ float8_vec compute_tqk_score_regs_raw_kq(
    const int8_t* __restrict__ k_tile,
    const i32x2_vec (&q_regs)[DTiles],
    const int lane,
    const int col_tile,
    const float score_scale) {
  constexpr int BK = 16;
  i32x8_vec score_acc;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    score_acc[elem] = 0;
  }
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    const i32x2_vec k_regs =
        pack_k_i8_wmma_b_regs_from_shared<SharedHeadStride>(
            k_tile, lane, col_tile, dt * BK);
    score_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true, k_regs, true, q_regs[dt], score_acc, true);
  }

  float8_vec scores;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores[elem] = static_cast<float>(score_acc[elem]) * score_scale;
  }
  return scores;
}

template <bool PvOrderedQK, int DTiles, int SharedHeadStride>
__device__ __forceinline__ float8_vec compute_tqk_score_regs_raw_kq_one(
    const int8_t* __restrict__ k_tile,
    const i32x2_vec (&q_regs)[DTiles],
    const int lane,
    const int col_tile,
    const float score_scale) {
  constexpr int BK = 16;
  i32x8_vec score_acc;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    score_acc[elem] = 0;
  }
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    i32x2_vec k_regs;
    if constexpr (PvOrderedQK) {
      k_regs = pack_k_i8_wmma_b_regs_from_shared_pv_order<SharedHeadStride>(
          k_tile, lane, col_tile, dt * BK);
    } else {
      k_regs = pack_k_i8_wmma_b_regs_from_shared<SharedHeadStride>(
          k_tile, lane, col_tile, dt * BK);
    }
    score_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true, k_regs, true, q_regs[dt], score_acc, true);
  }

  float8_vec scores;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores[elem] = static_cast<float>(score_acc[elem]) * score_scale;
  }
  return scores;
}

template <bool PvOrderedQK, int DTiles, int SharedHeadStride>
__device__ __forceinline__ void compute_tqk_score_regs_raw_kq_2(
    const int8_t* __restrict__ k_tile,
    const i32x2_vec (&q_regs)[2][DTiles],
    const int lane,
    const int col_tile,
    const float score_scale0,
    const float score_scale1,
    const bool do0,
    const bool do1,
    float8_vec& scores0,
    float8_vec& scores1) {
  constexpr int BK = 16;
  i32x8_vec score_acc0;
  i32x8_vec score_acc1;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    score_acc0[elem] = 0;
    score_acc1[elem] = 0;
  }
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    i32x2_vec k_regs;
    if constexpr (PvOrderedQK) {
      k_regs = pack_k_i8_wmma_b_regs_from_shared_pv_order<SharedHeadStride>(
          k_tile, lane, col_tile, dt * BK);
    } else {
      k_regs = pack_k_i8_wmma_b_regs_from_shared<SharedHeadStride>(
          k_tile, lane, col_tile, dt * BK);
    }
    if (do0) {
      score_acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[0][dt], score_acc0, true);
    }
    if (do1) {
      score_acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[1][dt], score_acc1, true);
    }
  }

#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores0[elem] = do0 ? static_cast<float>(score_acc0[elem]) * score_scale0 : -FLT_MAX * 0.5f;
    scores1[elem] = do1 ? static_cast<float>(score_acc1[elem]) * score_scale1 : -FLT_MAX * 0.5f;
  }
}

template <int DTiles>
__device__ __forceinline__ void compute_tqk_score_regs_raw_kq_2_lane_key(
    const int8_t* __restrict__ k,
    const int64_t k_head_base,
    const int64_t k_group_stride,
    const int64_t kb_base,
    const i32x2_vec (&q_regs)[2][DTiles],
    const int lane,
    const int col_tile,
    const float score_scale0,
    const float score_scale1,
    const bool do0,
    const bool do1,
    float8_vec& scores0,
    float8_vec& scores1) {
  i32x8_vec score_acc0;
  i32x8_vec score_acc1;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    score_acc0[elem] = 0;
    score_acc1[elem] = 0;
  }
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    const i32x2_vec k_regs =
        pack_k_i8_wmma_b_regs_from_lane_major_global<DTiles>(
            k, k_head_base, k_group_stride, kb_base, col_tile, dt, lane);
    if (do0) {
      score_acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[0][dt], score_acc0, true);
    }
    if (do1) {
      score_acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[1][dt], score_acc1, true);
    }
  }

#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores0[elem] = do0 ? static_cast<float>(score_acc0[elem]) * score_scale0 : -FLT_MAX * 0.5f;
    scores1[elem] = do1 ? static_cast<float>(score_acc1[elem]) * score_scale1 : -FLT_MAX * 0.5f;
  }
}

template <int DTiles>
__device__ __forceinline__ void compute_tqk_score_regs_raw_kq_2_lane_shared_key(
    const uint2* __restrict__ k_lane_tile,
    const i32x2_vec (&q_regs)[2][DTiles],
    const int lane,
    const int col_tile,
    const float score_scale0,
    const float score_scale1,
    const bool do0,
    const bool do1,
    float8_vec& scores0,
    float8_vec& scores1) {
  i32x8_vec score_acc0;
  i32x8_vec score_acc1;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    score_acc0[elem] = 0;
    score_acc1[elem] = 0;
  }
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    const i32x2_vec k_regs =
        pack_k_i8_wmma_b_regs_from_lane_major_shared<DTiles>(
            k_lane_tile, col_tile, dt, lane);
    if (do0) {
      score_acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[0][dt], score_acc0, true);
    }
    if (do1) {
      score_acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, k_regs, true, q_regs[1][dt], score_acc1, true);
    }
  }

#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    scores0[elem] = do0 ? static_cast<float>(score_acc0[elem]) * score_scale0 : -FLT_MAX * 0.5f;
    scores1[elem] = do1 ? static_cast<float>(score_acc1[elem]) * score_scale1 : -FLT_MAX * 0.5f;
  }
}

template <int SharedHeadStride>
__device__ __forceinline__ half8_vec make_v_regs_from_shared(
    const __half* __restrict__ v_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  half8_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    const int k_local = wmma_f16_k_for_lane_elem(lane, elem);
    const int row = col_tile * BK + k_local;
    regs[elem] = static_cast<_Float16>(v_tile[row * SharedHeadStride + d]);
  }
  return regs;
}

template <int SharedValueStride>
__device__ __forceinline__ half8_vec make_v_regs_from_transposed_shared(
    const __half* __restrict__ v_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  half8_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    const int k_local = wmma_f16_k_for_lane_elem(lane, elem);
    const int n = col_tile * BK + k_local;
    regs[elem] = static_cast<_Float16>(v_tile[d * SharedValueStride + n]);
  }
  return regs;
}

__device__ __forceinline__ half8_vec make_v_regs_from_transposed_global(
    const __half* __restrict__ v,
    const int64_t v_head_base,
    const int64_t v_stride_n,
    const int64_t kb_base,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  half8_vec regs;
  const int d = d_tile * BK + (lane & 15);
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    const int k_local = wmma_f16_k_for_lane_elem(lane, elem);
    const int64_t n = kb_base + col_tile * BK + k_local;
    regs[elem] = static_cast<_Float16>(v[v_head_base + static_cast<int64_t>(d) * v_stride_n + n]);
  }
  return regs;
}

__device__ __forceinline__ half8_vec make_v_regs_from_hnd_global(
    const __half* __restrict__ v,
    const int64_t v_head_base,
    const int64_t v_stride_n,
    const int64_t kb_base,
    const int col_tile,
    const int d_tile,
    const int lane) {
  constexpr int BK = 16;
  const int source_row = gfx12_tr_b128_source_row_for_lane(lane);
  const int source_col = gfx12_tr_b128_source_col_for_lane(lane);
  const int64_t n = kb_base + col_tile * BK + source_row;
  const int64_t d = d_tile * BK + source_col;
  return gfx12_global_load_tr_b128_f16(v + v_head_base + n * v_stride_n + d);
}

template <int DTiles>
__device__ __forceinline__ half8_vec make_v_regs_from_lane_major_shared(
    const uint4* __restrict__ v_lane_tile,
    const int col_tile,
    const int d_tile,
    const int lane) {
  const uint4 packed = v_lane_tile[(col_tile * DTiles + d_tile) * 32 + lane];
  return *reinterpret_cast<const half8_vec*>(&packed);
}

template <int DTiles>
__device__ __forceinline__ half8_vec make_v_regs_from_lane_major_global(
    const __half* __restrict__ v,
    const int64_t v_head_base,
    const int64_t v_group_stride,
    const int64_t kb_base,
    const int col_tile,
    const int d_tile,
    const int lane) {
  const int64_t group = kb_base >> 6;
  const int group_col_tile = ((static_cast<int>(kb_base) & 63) >> 4) + col_tile;
  const int64_t off = v_head_base + group * v_group_stride +
      (((static_cast<int64_t>(group_col_tile) * DTiles + d_tile) * 32 + lane) * 8);
  const uint4 packed = *reinterpret_cast<const uint4*>(v + off);
  return *reinterpret_cast<const half8_vec*>(&packed);
}

__device__ __forceinline__ void store_half(__half* output, const int64_t offset, const float value) {
  output[offset] = __float2half_rn(value);
}

__device__ __forceinline__ void store_output_value(
    __half* output,
    const int64_t offset,
    const float value) {
  output[offset] = __float2half_rn(value);
}

__device__ __forceinline__ void store_output_value(
    __hip_bfloat16* output,
    const int64_t offset,
    const float value) {
  output[offset] = __float2bfloat16(value);
}

template <
    int BlockCols,
    int BlockRows,
    bool HndContiguous = false,
    bool ValueTransposed = false,
    int ValuePad = SAGEATTN_GFX12_NATIVE_F16_TV_PAD,
    bool IsCausal = false,
    bool TransposeValueOnLoad = false,
    bool F16PvAccum = false,
    bool PvOrderedQK = false,
    typename QueryT = int8_t,
    bool QuantizeQuery = false,
    bool SplitCausalPrefix = false,
    bool PerThreadQK = false>
SAGEATTN_NATIVE_WAVES_PER_EU __global__ __launch_bounds__((BlockRows / 16) * 32, 1) void qk_int8_sv_f16_d64_native_kernel(
    const QueryT* __restrict__ q,
    const int8_t* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout,
    const float sm_scale,
    const bool per_thread_qk = false) {
  constexpr int HeadDim = 64;
  constexpr int BR = BlockRows;
  constexpr int RM = 16;
  constexpr int BK = 16;
  constexpr int BC = BlockCols;
  constexpr int Threads = (BlockRows / RM) * 32;
  constexpr int DTiles = HeadDim / BK;
  constexpr int ColTiles = BC / BK;
  constexpr int SharedHeadStride = HeadDim + 16;
  constexpr bool UseTransposedValueLayout = ValueTransposed || TransposeValueOnLoad;
  constexpr bool UseTrLoadLaneMajorValue = false;
  constexpr int SharedValueRows = UseTransposedValueLayout ? HeadDim : BC;
  constexpr int SharedValueStride = UseTransposedValueLayout ? (BC + ValuePad) : SharedHeadStride;
  constexpr int LaneMajorValueElems = ColTiles * DTiles * 32;
  constexpr int PackedRows = 4;
  static_assert(BlockCols == 64 || BlockCols == 128,
                "native gfx12 D64 kernel supports BC64/BC128.");
  static_assert(BlockRows == 64 || BlockRows == 128,
                "native gfx12 D64 kernel supports BR64/BR128.");
  static_assert(!UseTransposedValueLayout || HndContiguous,
                "transposed fp16 value path requires contiguous HND tensors.");
  static_assert(!IsCausal || ((BlockRows == 64 || BlockRows == 128) && BlockCols == 64),
                "native gfx12 D64 single-q causal path supports BR64/BR128/BC64.");
  static_assert(!QuantizeQuery || HndContiguous,
                "direct fp16 Q quantization requires contiguous HND tensors.");

  __shared__ int8_t k_tile[BC][SharedHeadStride];
  __shared__ __half v_tile[UseTrLoadLaneMajorValue ? 1 : SharedValueRows]
                          [UseTrLoadLaneMajorValue ? 1 : SharedValueStride];
  __shared__ uint4 v_lane_tile[UseTrLoadLaneMajorValue ? LaneMajorValueElems : 1];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int row_base = (lane >> 4) << 3;
  const int col = lane & 15;
  const int64_t q_base =
      q_block_base_for_launch<IsCausal, BR>(static_cast<int64_t>(blockIdx.x), qo_len);
  const int64_t hq = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) {
    return;
  }

  const int64_t hkv = hq / (num_qo_heads / num_kv_heads);
  const int64_t q_start = q_base + static_cast<int64_t>(wave) * RM;
  float qs = 1.0f;

  using FragK = rocwmma::fragment<rocwmma::matrix_a, RM, BK, BK, int8_t, rocwmma::row_major>;
  using FragQ = rocwmma::fragment<rocwmma::matrix_b, RM, BK, BK, int8_t, rocwmma::col_major>;
  using FragScoreT = rocwmma::fragment<rocwmma::accumulator, RM, BK, BK, int32_t>;
  constexpr bool UseRawPreparedQ = !QuantizeQuery && HndContiguous;

  i32x2_vec q_regs[DTiles];
  FragQ q_frag[DTiles];
  if constexpr (QuantizeQuery) {
    constexpr int QPackElems = 8;
    constexpr int QPacks = (RM * HeadDim) / QPackElems;
    float local_q_amax = 0.0000001f;
    for (int pack = lane; pack < QPacks; pack += 32) {
      const int elem_base = pack * QPackElems;
      const int row = elem_base / HeadDim;
      const int d = elem_base - row * HeadDim;
      const int64_t q_idx = q_start + row;
      if (q_idx < qo_len) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous>(
            tensor_layout, b, hq, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
        const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
        for (int i = 0; i < QPackElems; ++i) {
          local_q_amax = fmaxf(local_q_amax, fabsf(value_to_float(values[i])));
        }
      }
    }
    local_q_amax = vllm::warpReduceMax(local_q_amax);
    const float q_amax = __shfl(local_q_amax, 0, 32);
    const float inv_q_scale = 127.0f / q_amax;
    qs = (q_amax / 127.0f) * sm_scale * kLog2e;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
      q_regs[dt] = pack_quant_q_i8_wmma_b_regs<QueryT, HeadDim, HndContiguous>(
          q, tensor_layout, lane, b, hq, q_start, qo_len, dt * BK,
          q_stride_b, q_stride_n, q_stride_h, inv_q_scale);
    }
  } else {
    if constexpr (PerThreadQK) {
      qs = 1.0f;
    } else {
      const int q_scale_idx = q_scale_col_per_warp(q_start);
      qs = q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx] *
          sm_scale * kLog2e;
    }
    if constexpr (UseRawPreparedQ) {
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        q_regs[dt] = pack_q_i8_wmma_b_regs<HeadDim, HndContiguous>(
            reinterpret_cast<const int8_t*>(q), tensor_layout, lane, b, hq, q_start,
            qo_len, dt * BK, q_stride_b, q_stride_n, q_stride_h);
      }
    } else {
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous>(
            tensor_layout, b, hq, q_start, dt * BK, q_stride_b, q_stride_n, q_stride_h);
        rocwmma::load_matrix_sync(q_frag[dt], q + q_off, static_cast<uint32_t>(q_stride_n));
      }
    }
  }

  using PvAccumVec = std::conditional_t<F16PvAccum, half8_vec, float8_vec>;
  PvAccumVec out_frag[DTiles];
#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      out_frag[dt][elem] = 0.0f;
    }
  }
  float m = -FLT_MAX * 0.5f;
  float l = 0.0f;

  auto process_kv_tile = [&](const int64_t kb_base, auto causal_mask_tag) {
    constexpr bool ApplyCausalMask = decltype(causal_mask_tag)::value;
    constexpr int KVecBytes = 16;
    constexpr int KVecsPerRow = HeadDim / KVecBytes;
    for (int vec = tid; vec < BC * KVecsPerRow; vec += Threads) {
      const int n = vec / KVecsPerRow;
      const int d = (vec - n * KVecsPerRow) * KVecBytes;
      const int64_t k_off = qkv_offset_dispatch<HeadDim, HndContiguous>(
          tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
      *reinterpret_cast<uint4*>(&k_tile[n][d]) =
          *reinterpret_cast<const uint4*>(k + k_off);
    }

    if constexpr (UseTrLoadLaneMajorValue) {
      for (int idx = tid; idx < LaneMajorValueElems; idx += Threads) {
        const int lane_local = idx & 31;
        const int d_tile = (idx >> 5) % DTiles;
        const int col_tile = idx / (DTiles * 32);
        const half8_vec regs = make_v_regs_from_hnd_global(
            v, b * v_stride_b + hkv * v_stride_h, v_stride_n,
            kb_base, col_tile, d_tile, lane_local);
        v_lane_tile[idx] = *reinterpret_cast<const uint4*>(&regs);
      }
    } else if constexpr (ValueTransposed) {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerD = BC / VElemsPerVec;
      for (int vec = tid; vec < HeadDim * VVecsPerD; vec += Threads) {
        const int d = vec / VVecsPerD;
        const int n = (vec - d * VVecsPerD) * VElemsPerVec;
        const int64_t v_off = b * v_stride_b + hkv * v_stride_h +
            static_cast<int64_t>(d) * v_stride_n + kb_base + n;
        *reinterpret_cast<uint4*>(&v_tile[d][n]) =
            *reinterpret_cast<const uint4*>(v + v_off);
      }
    } else if constexpr (TransposeValueOnLoad) {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerRow = HeadDim / VElemsPerVec;
      for (int vec = tid; vec < BC * VVecsPerRow; vec += Threads) {
        const int n = vec / VVecsPerRow;
        const int d = (vec - n * VVecsPerRow) * VElemsPerVec;
        const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous>(
            tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
        const uint4 packed = *reinterpret_cast<const uint4*>(v + v_off);
        const __half* vals = reinterpret_cast<const __half*>(&packed);
#pragma unroll
        for (int elem = 0; elem < VElemsPerVec; ++elem) {
          v_tile[d + elem][n] = vals[elem];
        }
      }
    } else {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerRow = HeadDim / VElemsPerVec;
      for (int vec = tid; vec < BC * VVecsPerRow; vec += Threads) {
        const int n = vec / VVecsPerRow;
        const int d = (vec - n * VVecsPerRow) * VElemsPerVec;
        const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous>(
            tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
        *reinterpret_cast<uint4*>(&v_tile[n][d]) =
            *reinterpret_cast<const uint4*>(v + v_off);
      }
    }
      __syncthreads();

    if constexpr (IsCausal) {
      float8_vec score_cache[ColTiles];
      half8_vec prob_cache[ColTiles];
      float local_max = -FLT_MAX * 0.5f;
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        const float score_scale = qk_score_scale_scalar<PerThreadQK, PvOrderedQK>(q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        float8_vec scores;
        const int64_t k_col_start = kb_base + static_cast<int64_t>(col_tile) * BK;
        const bool fully_future =
            ApplyCausalMask && (k_col_start > q_start + RM - 1);
        if constexpr (ApplyCausalMask) {
          if (fully_future) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              scores[elem] = -FLT_MAX * 0.5f;
            }
          } else if constexpr (QuantizeQuery || UseRawPreparedQ) {
            scores = compute_tqk_score_regs_raw_kq_one<PvOrderedQK, DTiles, SharedHeadStride>(
                &k_tile[0][0], q_regs, lane, col_tile, score_scale);
          } else {
            scores = compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                &k_tile[0][0], q_frag, col_tile, score_scale);
          }
        } else if constexpr (QuantizeQuery || UseRawPreparedQ) {
          scores = compute_tqk_score_regs_raw_kq_one<PvOrderedQK, DTiles, SharedHeadStride>(
              &k_tile[0][0], q_regs, lane, col_tile, score_scale);
        } else {
          scores = compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
              &k_tile[0][0], q_frag, col_tile, score_scale);
        }
        apply_per_thread_qk_score_scale<PerThreadQK, PvOrderedQK>(scores, q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile, lane,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        if constexpr (ApplyCausalMask) {
          if (!fully_future && k_col_start + BK > q_start) {
            if constexpr (PvOrderedQK) {
              apply_tqk_causal_mask_pv_order<true>(
                  scores, static_cast<int>(q_start), static_cast<int>(kb_base), col_tile, lane);
            } else {
              apply_tqk_causal_mask<true>(
                  scores, static_cast<int>(q_start), static_cast<int>(kb_base), col_tile, lane);
            }
          }
        }
        if (k_col_start + BK > kv_len) {
          apply_tqk_kv_tail_mask<PvOrderedQK>(scores, kv_len, kb_base, col_tile, lane);
        }
        score_cache[col_tile] = scores;
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          local_max = fmaxf(local_max, scores[elem]);
        }
      }
      const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
      const float old_m = m;
      const float new_m = fmaxf(old_m, tile_max);
      const float alpha = l == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
      m = new_m;
      l *= alpha;

#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          const int row = row_base + elem;
          out_frag[dt][elem] *= __shfl(alpha, row, 32);
        }
      }

      float local_sum = 0.0f;
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        half8_vec prob_values;
        const int64_t k_col_start = kb_base + static_cast<int64_t>(col_tile) * BK;
        const bool fully_future =
            ApplyCausalMask && (k_col_start > q_start + RM - 1);
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          const float prob = fully_future ? 0.0f :
              fast_exp2(score_cache[col_tile][elem] - m + kF16SoftmaxOffset);
          local_sum += prob;
          prob_values[elem] = static_cast<_Float16>(prob);
        }
        if constexpr (PvOrderedQK) {
          prob_cache[col_tile] = prob_values;
        } else {
          prob_cache[col_tile] = make_p_regs_from_tqk_prob_regs(prob_values, lane);
        }
      }
      l += local_sum + __shfl_xor(local_sum, 16, 32);
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        const int64_t k_col_start = kb_base + static_cast<int64_t>(col_tile) * BK;
        const bool fully_future =
            ApplyCausalMask && (k_col_start > q_start + RM - 1);
        if constexpr (ApplyCausalMask) {
          if (fully_future) {
            continue;
          }
        }
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          half8_vec v_regs;
          if constexpr (UseTrLoadLaneMajorValue) {
            v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                v_lane_tile, col_tile, dt, lane);
          } else if constexpr (UseTransposedValueLayout) {
            v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          } else {
            v_regs = make_v_regs_from_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          }
          PvAccumVec acc;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            acc[elem] = out_frag[dt][elem];
          }
          PvAccumVec pv_acc;
          if constexpr (F16PvAccum) {
            pv_acc =
                __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                    prob_cache[col_tile], v_regs, acc);
          } else {
            pv_acc =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    prob_cache[col_tile], v_regs, acc);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[dt][elem] = pv_acc[elem];
          }
        }
      }
    } else {
      float local_max = -FLT_MAX * 0.5f;
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        const float score_scale = qk_score_scale_scalar<PerThreadQK, false>(q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        const int64_t k_col_start = kb_base + static_cast<int64_t>(col_tile) * BK;
        float8_vec scores;
        if constexpr (QuantizeQuery || UseRawPreparedQ) {
          scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
              &k_tile[0][0], q_regs, lane, col_tile, score_scale);
        } else {
          scores = compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
              &k_tile[0][0], q_frag, col_tile, score_scale);
        }
        apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile, lane,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        if (k_col_start + BK > kv_len) {
          apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
        }
        for (int elem = 0; elem < 8; ++elem) {
          local_max = fmaxf(local_max, scores[elem]);
        }
      }
      const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
      const float old_m = m;
      const float new_m = fmaxf(old_m, tile_max);
      const float alpha = l == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
      m = new_m;
      l *= alpha;

#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          const int row = row_base + elem;
          out_frag[dt][elem] *= __shfl(alpha, row, 32);
        }
      }

      float local_sum = 0.0f;
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        const float score_scale = qk_score_scale_scalar<PerThreadQK, false>(q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        const int64_t k_col_start = kb_base + static_cast<int64_t>(col_tile) * BK;
        float8_vec scores;
          if constexpr (QuantizeQuery || UseRawPreparedQ) {
            scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
                &k_tile[0][0], q_regs, lane, col_tile, score_scale);
          } else {
          scores = compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
              &k_tile[0][0], q_frag, col_tile, score_scale);
        }
        apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start, kb_base, col_tile, lane,
            qo_len, kv_len, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
            sm_scale);
        if (k_col_start + BK > kv_len) {
          apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
        }
        half8_vec prob_values;
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          const float prob = fast_exp2(scores[elem] - m + kF16SoftmaxOffset);
          local_sum += prob;
          prob_values[elem] = static_cast<_Float16>(prob);
        }

#if SAGEATTN_NATIVE_HAS_GFX12_WMMA
        const half8_vec p_regs = make_p_regs_from_tqk_prob_regs(prob_values, lane);
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          half8_vec v_regs;
          if constexpr (UseTrLoadLaneMajorValue) {
            v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                v_lane_tile, col_tile, dt, lane);
          } else if constexpr (UseTransposedValueLayout) {
            v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          } else {
            v_regs = make_v_regs_from_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          }
          PvAccumVec acc;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            acc[elem] = out_frag[dt][elem];
          }
          PvAccumVec pv_acc;
          if constexpr (F16PvAccum) {
            pv_acc =
                __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(p_regs, v_regs, acc);
          } else {
            pv_acc =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_regs, v_regs, acc);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[dt][elem] = pv_acc[elem];
          }
        }
#else
      static_assert(SAGEATTN_NATIVE_HAS_GFX12_WMMA, "native gfx12 kernel requires gfx12 WMMA builtins");
#endif
      }
      l += local_sum + __shfl_xor(local_sum, 16, 32);
    }
    __syncthreads();
  };

  const int64_t kv_limit = IsCausal && (q_base + BR) < kv_len ? q_base + BR : kv_len;
  if constexpr (IsCausal && SplitCausalPrefix) {
    const int64_t prefix_limit = q_base < kv_limit ? q_base : kv_limit;
    for (int64_t kb_base = 0; kb_base < prefix_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::false_type{});
    }
    for (int64_t kb_base = prefix_limit; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::true_type{});
    }
  } else {
    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::integral_constant<bool, IsCausal>{});
    }
  }

#pragma unroll
  for (int dt = 0; dt < DTiles; ++dt) {
    const int d = dt * BK + col;
#pragma unroll
    for (int pair = 0; pair < PackedRows; ++pair) {
      const int elem = pair * 2;
      const int64_t q_idx0 = q_start + row_base + elem;
      const int64_t q_idx1 = q_idx0 + 1;
      const float l_sum0 = __shfl(l, row_base + elem, 32);
      const float l_sum1 = __shfl(l, row_base + elem + 1, 32);
      const float value0 = l_sum0 == 0.0f ? 0.0f : out_frag[dt][elem] / l_sum0;
      const float value1 = l_sum1 == 0.0f ? 0.0f : out_frag[dt][elem + 1] / l_sum1;
      store_half(output, qkv_offset_dispatch<HeadDim, HndContiguous>(
          tensor_layout, b, hq, q_idx0, d, o_stride_b, o_stride_n, o_stride_h), value0);
      store_half(output, qkv_offset_dispatch<HeadDim, HndContiguous>(
          tensor_layout, b, hq, q_idx1, d, o_stride_b, o_stride_n, o_stride_h), value1);
    }
  }
}

template <int BlockCols,
          bool HndContiguous = false,
          int BlockRows = 128,
          bool ValueTransposed = false,
          int ValuePad = SAGEATTN_GFX12_NATIVE_F16_TV_PAD,
          bool IsCausal = false,
          bool TransposeValueOnLoad = false,
          bool F16PvAccum = false,
          typename QueryT = int8_t,
          bool QuantizeQuery = false,
          typename KeyT = int8_t,
          bool QuantizeKey = false,
          bool PvOrderedQK = false,
          bool LaneMajorValue = false,
          bool StreamColTiles = false,
          bool LaneMajorKey = false,
          int HeadDim = 64,
          bool FlatCausalSchedule = false,
          bool PerThreadQK = false,
          bool StaticNhdLayout = false,
          bool NoKvTail = false,
          bool SameQKHeads = false,
          bool NoQueryTail = false,
          bool PrefetchStreamVRegs = false,
          bool DirectStreamProbs = false,
          bool DirectPvOutFragAccum = false>
SAGEATTN_NATIVE_2Q_WAVES_PER_EU(HeadDim, IsCausal) __global__
SAGEATTN_NATIVE_F16_2Q_LAUNCH_BOUNDS(BlockRows) void qk_int8_sv_f16_d64_native_2q_kernel(
    const QueryT* __restrict__ q,
    const KeyT* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout,
    const float sm_scale,
    const bool per_thread_qk = false) {
  static_assert(HeadDim == 16 || HeadDim == 64 || HeadDim == 128 || HeadDim == 256,
                "native gfx12 fp16 2q kernel supports D16/D64/D128/D256.");
  constexpr int BR = BlockRows;
  constexpr int RM = 16;
  constexpr int RowsPerWave = 32;
  constexpr int QGroups = 2;
  constexpr int BK = 16;
  constexpr int BC = BlockCols;
  constexpr int Threads = BlockRows;
  constexpr int DTiles = HeadDim / BK;
  constexpr int ColTiles = BC / BK;
  constexpr int SharedHeadStride = HeadDim + 16;
  constexpr int SharedQKHeadStride = SharedHeadStride;
  constexpr bool UseTransposedValueLayout = ValueTransposed || TransposeValueOnLoad;
  constexpr bool UseLaneMajorKey =
      LaneMajorKey && !QuantizeKey && HndContiguous && BlockCols == 64 && PvOrderedQK;
  constexpr bool UseLaneMajorValue =
      LaneMajorValue && HndContiguous &&
      (ValueTransposed || TransposeValueOnLoad);
  constexpr bool StageValueInShared = !UseLaneMajorValue;
  constexpr bool UsesTileSharedMemory = true;
  constexpr bool UsesKeySharedMemory = true;
  constexpr bool UsesValueSharedMemory = StageValueInShared || UseLaneMajorValue;
  constexpr int SharedValueRows =
      StageValueInShared ? (UseTransposedValueLayout ? HeadDim : BC) : 1;
  constexpr int SharedValueStride =
      StageValueInShared ? (UseTransposedValueLayout ? (BC + ValuePad) : SharedHeadStride) : 1;
  constexpr int PackedRows = 4;
  static_assert(BlockCols == 32 || BlockCols == 64 || BlockCols == 128,
                "native gfx12 fp16 2q kernel supports BC32/BC64/BC128.");
  static_assert(BlockRows == 32 || BlockRows == 64 || BlockRows == 128 ||
                    BlockRows == 256 || BlockRows == 512 || BlockRows == 1024,
                "native gfx12 fp16 2q kernel supports BR32/BR64/BR128/BR256/BR512/BR1024.");
  static_assert(!UseTransposedValueLayout || HndContiguous,
                "transposed fp16 value layout requires contiguous HND tensors.");
  static_assert(!F16PvAccum || BlockCols <= 64,
                "fp16 PV accumulation currently supports the BC64 2q path.");
  static_assert(!QuantizeKey || (HndContiguous && BlockCols == 64),
                "direct fp16 K quantization currently requires contiguous HND BC64 tensors.");
  static_assert(!LaneMajorValue ||
                    (HndContiguous && (ValueTransposed || TransposeValueOnLoad)),
                "lane-major fp16 value staging requires contiguous HND values.");
  static_assert(!LaneMajorKey ||
                     (!QuantizeKey && HndContiguous && BlockCols == 64 && PvOrderedQK),
                "lane-major fp16 key staging requires prepared HND BC64 PvOrderedQK.");
  static_assert(!StreamColTiles || (BlockCols <= 128 && (QuantizeQuery || HndContiguous)),
                "streaming col-tile softmax is specialized for raw/quantized QK.");
  static_assert(!NoQueryTail || StaticNhdLayout,
                "full-query fp16 path requires a static dispatch.");
  __shared__ int8_t k_tile[UseLaneMajorKey ? 1 : BC]
                        [SharedQKHeadStride];
  __shared__ uint2 k_lane_tile[UseLaneMajorKey ? (ColTiles * DTiles * 32) : 1];
  __shared__ __half v_tile[SharedValueRows][SharedValueStride];
  __shared__ uint4 v_lane_tile[UseLaneMajorValue ? (ColTiles * DTiles * 32) : 1];
  __shared__ float raw_k_amax_shared;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int row_base = (lane >> 4) << 3;
  const int col = lane & 15;
  const int64_t hb_count = num_qo_heads * batch_size;
  for (;;) {
    int64_t q_block = static_cast<int64_t>(blockIdx.x);
    int64_t hq = blockIdx.y;
    int64_t b = blockIdx.z;
    if constexpr (FlatCausalSchedule) {
      static_assert(IsCausal, "flat q scheduling is causal-only");
      const int64_t hb = static_cast<int64_t>(blockIdx.x) % hb_count;
      q_block = static_cast<int64_t>(blockIdx.x) / hb_count;
      hq = hb % num_qo_heads;
      b = hb / num_qo_heads;
    }
    const int64_t q_base = q_block_base_for_launch<IsCausal, BR>(q_block, qo_len);
    if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) {
      return;
    }

    const int64_t hkv = SameQKHeads ? hq : hq / (num_qo_heads / num_kv_heads);
    const int64_t k_head_base = b * k_stride_b + hkv * k_stride_h;
    const int64_t v_head_base = b * v_stride_b + hkv * v_stride_h;
    int64_t q_start[QGroups];
    float qs[QGroups];
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
      q_start[qg] = q_base + static_cast<int64_t>(wave) * RowsPerWave + qg * RM;
    }

  using FragK = rocwmma::fragment<rocwmma::matrix_a, RM, BK, BK, int8_t, rocwmma::row_major>;
  using FragQ = rocwmma::fragment<rocwmma::matrix_b, RM, BK, BK, int8_t, rocwmma::col_major>;
  using FragScoreT = rocwmma::fragment<rocwmma::accumulator, RM, BK, BK, int32_t>;
  constexpr bool UseRawPreparedQ = !QuantizeQuery && HndContiguous;

  i32x2_vec q_regs[QGroups][DTiles];
  if constexpr (QuantizeQuery) {
    constexpr int QPackElems = 8;
    constexpr int QPacksPerWave = (RowsPerWave * HeadDim) / QPackElems;
    const int local_q_row_base = wave * RowsPerWave;
    float local_q_amax = 0.0000001f;
    for (int pack = lane; pack < QPacksPerWave; pack += 32) {
      const int elem_base = pack * QPackElems;
      const int row = elem_base / HeadDim;
      const int d = elem_base - row * HeadDim;
      const int64_t q_idx = q_base + local_q_row_base + row;
      if constexpr (NoQueryTail) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
        const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
        for (int i = 0; i < QPackElems; ++i) {
          local_q_amax = fmaxf(local_q_amax, fabsf(value_to_float(values[i])));
        }
      } else if (q_idx < qo_len) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
        const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
        for (int i = 0; i < QPackElems; ++i) {
          local_q_amax = fmaxf(local_q_amax, fabsf(value_to_float(values[i])));
        }
      }
    }
    local_q_amax = vllm::warpReduceMax(local_q_amax);
    const float q_scale_local = __shfl(local_q_amax, 0, 32) / 127.0f;
    const float inv_q_scale = 127.0f / __shfl(local_q_amax, 0, 32);
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
      qs[qg] = q_scale_local * sm_scale * kLog2e;
      const int64_t qg_start = q_start[qg];
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        q_regs[qg][dt] =
            pack_quant_q_i8_wmma_b_regs<QueryT, HeadDim, HndContiguous, StaticNhdLayout, NoQueryTail>(
            q, tensor_layout, lane, b, hq, qg_start, qo_len, dt * BK,
            q_stride_b, q_stride_n, q_stride_h, inv_q_scale);
      }
    }
  } else {
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
      if constexpr (PerThreadQK) {
        qs[qg] = 1.0f;
      } else {
        const int q_scale_idx = q_scale_col_per_warp(q_start[qg]);
        qs[qg] = q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx] *
            sm_scale * kLog2e;
      }
    }
    if constexpr (UseRawPreparedQ) {
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          q_regs[qg][dt] =
              pack_q_i8_wmma_b_regs<HeadDim, HndContiguous, StaticNhdLayout, NoQueryTail>(
              reinterpret_cast<const int8_t*>(q), tensor_layout, lane, b, hq, q_start[qg],
              qo_len, dt * BK, q_stride_b, q_stride_n, q_stride_h);
        }
      }
    }
  }

  FragQ q_frag[QGroups][DTiles];
  if constexpr (!QuantizeQuery && !UseRawPreparedQ) {
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_start[qg], dt * BK, q_stride_b, q_stride_n, q_stride_h);
        rocwmma::load_matrix_sync(q_frag[qg][dt], q + q_off, static_cast<uint32_t>(q_stride_n));
      }
    }
  }

  using PvAccumVec = std::conditional_t<F16PvAccum, half8_vec, float8_vec>;
  PvAccumVec out_frag[QGroups][DTiles];
  float m[QGroups];
  float l[QGroups];
#pragma unroll
  for (int qg = 0; qg < QGroups; ++qg) {
    m[qg] = -FLT_MAX * 0.5f;
    l[qg] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        out_frag[qg][dt][elem] = 0.0f;
      }
    }
  }

  constexpr bool ExactStaticCausalBlock =
      IsCausal && StaticNhdLayout && NoKvTail && NoQueryTail &&
      (BR % BC == 0);
  const int64_t kv_limit =
      ExactStaticCausalBlock ? (q_base + BR) :
      (IsCausal && (q_base + BR) < kv_len ? q_base + BR : kv_len);
  auto process_kv_tile = [&](const int64_t kb_base, auto apply_causal_mask_tag) {
    constexpr int KVecBytes = 16;
    constexpr int KBytesPerRow = HeadDim;
    constexpr int KVecsPerRow = KBytesPerRow / KVecBytes;
    float k_scale_tile = 1.0f;
    if constexpr (QuantizeKey) {
      constexpr int PackElems = 8;
      constexpr int Packs = (BC * HeadDim) / PackElems;
      float local_k_amax = 0.0000001f;
      for (int pack = tid; pack < Packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int n = elem_base / HeadDim;
        const int d = elem_base - n * HeadDim;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(k + k_off);
        const KeyT* values = reinterpret_cast<const KeyT*>(&raw);
#pragma unroll
        for (int i = 0; i < PackElems; ++i) {
          local_k_amax = fmaxf(local_k_amax, fabsf(value_to_float(values[i])));
        }
      }
      const float block_k_amax = vllm::blockReduceMax(local_k_amax);
      if (tid == 0) {
        raw_k_amax_shared = block_k_amax;
      }
      __syncthreads();
      const float raw_k_amax = raw_k_amax_shared;
      k_scale_tile = raw_k_amax / 127.0f;
      const float inv_k_scale = 127.0f / raw_k_amax;

      for (int pack = tid; pack < Packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int n = elem_base / HeadDim;
        const int d = elem_base - n * HeadDim;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(k + k_off);
        const KeyT* values = reinterpret_cast<const KeyT*>(&raw);
        char4 out0;
        char4 out1;
        out0.x = float_to_int8_rn_gfx12(value_to_float(values[0]) * inv_k_scale);
        out0.y = float_to_int8_rn_gfx12(value_to_float(values[1]) * inv_k_scale);
        out0.z = float_to_int8_rn_gfx12(value_to_float(values[2]) * inv_k_scale);
        out0.w = float_to_int8_rn_gfx12(value_to_float(values[3]) * inv_k_scale);
        out1.x = float_to_int8_rn_gfx12(value_to_float(values[4]) * inv_k_scale);
        out1.y = float_to_int8_rn_gfx12(value_to_float(values[5]) * inv_k_scale);
        out1.z = float_to_int8_rn_gfx12(value_to_float(values[6]) * inv_k_scale);
        out1.w = float_to_int8_rn_gfx12(value_to_float(values[7]) * inv_k_scale);
        *reinterpret_cast<char4*>(&k_tile[n][d]) = out0;
        *reinterpret_cast<char4*>(&k_tile[n][d + 4]) = out1;
      }
    } else if constexpr (UseLaneMajorKey) {
      constexpr int LaneMajorElems = ColTiles * DTiles * 32;
      for (int idx = tid; idx < LaneMajorElems; idx += Threads) {
        const int lane_local = idx & 31;
        const int d_tile = (idx >> 5) % DTiles;
        const int col_tile = idx / (DTiles * 32);
        const int col_local = lane_local & 15;
        const int row = col_tile * BK + pv_k_order_for_acc_row(col_local);
        const int d = d_tile * BK + 8 * (lane_local >> 4);
        const int64_t k_off = k_head_base + (kb_base + row) * HeadDim + d;
        k_lane_tile[idx] = *reinterpret_cast<const uint2*>(k + k_off);
      }
    } else {
      for (int vec = tid; vec < BC * KVecsPerRow; vec += Threads) {
        const int n = vec / KVecsPerRow;
        const int d = (vec - n * KVecsPerRow) * KVecBytes;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        *reinterpret_cast<uint4*>(&k_tile[n][d]) =
            *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(k) + k_off);
      }
    }

    float prepared_k_scale_tile = k_scale_tile;
    if constexpr (!QuantizeKey && BC <= 64) {
      if constexpr (!PerThreadQK) {
        const int k_scale_idx = k_scale_col_per_warp(kb_base);
        prepared_k_scale_tile =
            k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
      }
    }

    auto stage_value_tile = [&]() {
    if constexpr (UseLaneMajorValue) {
      if constexpr (ValueTransposed) {
        constexpr int LaneMajorElems = ColTiles * DTiles * 32;
        for (int idx = tid; idx < LaneMajorElems; idx += Threads) {
          const int lane_local = idx & 31;
          const int d_tile = (idx >> 5) % DTiles;
          const int col_tile = idx / (DTiles * 32);
          const int d = d_tile * BK + (lane_local & 15);
          const int high_half = (lane_local >> 4) & 1;
          const int n0 = col_tile * BK + high_half * 4;
          const int n1 = col_tile * BK + 8 + high_half * 4;
          const int64_t base = v_head_base + static_cast<int64_t>(d) * v_stride_n + kb_base;
          const uint2 raw0 = *reinterpret_cast<const uint2*>(v + base + n0);
          const uint2 raw1 = *reinterpret_cast<const uint2*>(v + base + n1);
          uint4 packed;
          packed.x = raw0.x;
          packed.y = raw0.y;
          packed.z = raw1.x;
          packed.w = raw1.y;
          v_lane_tile[idx] = packed;
        }
      } else {
        constexpr int VElemsPerVec = 8;
        constexpr int VVecsPerRow = HeadDim / VElemsPerVec;
        __half* lane_values = reinterpret_cast<__half*>(v_lane_tile);
        for (int vec = tid; vec < BC * VVecsPerRow; vec += Threads) {
          const int n = vec / VVecsPerRow;
          const int d_base = (vec - n * VVecsPerRow) * VElemsPerVec;
          const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
              tensor_layout, b, hkv, kb_base + n, d_base, v_stride_b, v_stride_n, v_stride_h);
          const uint4 packed = *reinterpret_cast<const uint4*>(v + v_off);
          const __half* vals = reinterpret_cast<const __half*>(&packed);
          const int col_tile = n / BK;
          const int k_local = n - col_tile * BK;
          const int dst_elem = ((k_local & 8) >> 1) | (k_local & 3);
          const int dst_lane_hi = ((k_local >> 2) & 1) << 4;
#pragma unroll
          for (int elem = 0; elem < VElemsPerVec; ++elem) {
            const int d = d_base + elem;
            const int d_tile = d >> 4;
            const int dst_lane = (d & 15) | dst_lane_hi;
            const int slot = ((col_tile * DTiles + d_tile) * 32 + dst_lane) * 8 + dst_elem;
            lane_values[slot] = vals[elem];
          }
        }
      }
    } else if constexpr (StageValueInShared && ValueTransposed) {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerD = BC / VElemsPerVec;
      for (int vec = tid; vec < HeadDim * VVecsPerD; vec += Threads) {
        const int d = vec / VVecsPerD;
        const int n = (vec - d * VVecsPerD) * VElemsPerVec;
        const int64_t v_off = v_head_base + static_cast<int64_t>(d) * v_stride_n + kb_base + n;
        *reinterpret_cast<uint4*>(&v_tile[d][n]) =
            *reinterpret_cast<const uint4*>(v + v_off);
      }
    } else if constexpr (StageValueInShared && TransposeValueOnLoad) {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerRow = HeadDim / VElemsPerVec;
      for (int vec = tid; vec < BC * VVecsPerRow; vec += Threads) {
        const int n = vec / VVecsPerRow;
        const int d = (vec - n * VVecsPerRow) * VElemsPerVec;
        const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
        const uint4 packed = *reinterpret_cast<const uint4*>(v + v_off);
        const __half* vals = reinterpret_cast<const __half*>(&packed);
#pragma unroll
        for (int elem = 0; elem < VElemsPerVec; ++elem) {
          v_tile[d + elem][n] = vals[elem];
        }
      }
    } else if constexpr (StageValueInShared) {
      constexpr int VElemsPerVec = 8;
      constexpr int VVecsPerRow = HeadDim / VElemsPerVec;
      for (int vec = tid; vec < BC * VVecsPerRow; vec += Threads) {
        const int n = vec / VVecsPerRow;
        const int d = (vec - n * VVecsPerRow) * VElemsPerVec;
        const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
        *reinterpret_cast<uint4*>(&v_tile[n][d]) =
            *reinterpret_cast<const uint4*>(v + v_off);
      }
    }
    };
    stage_value_tile();
    if constexpr (UsesKeySharedMemory || UsesValueSharedMemory) {
      __syncthreads();
    }
    SAGEATTN_F16_SCHED_BARRIER(0);

    auto compute_loaded_tile = [&](auto causal_mask_tag) {
      constexpr bool ApplyCausalMask = decltype(causal_mask_tag)::value;
    if constexpr (BlockCols <= 128) {
      if constexpr ((QuantizeQuery || UseRawPreparedQ) && StreamColTiles) {
        constexpr int StreamGroupCols = ColTiles >= 2 ? 2 : 1;
#pragma unroll
        for (int group_base = 0; group_base < ColTiles; group_base += StreamGroupCols) {
          float8_vec scores0[StreamGroupCols];
          float8_vec scores1[StreamGroupCols];
          bool fully_future[QGroups][StreamGroupCols];
          bool any_work = false;
#pragma unroll
          for (int gc = 0; gc < StreamGroupCols; ++gc) {
            const int col_tile = group_base + gc;
            const int64_t k_col_start = kb_base + col_tile * BK;
            const bool fully_future0 =
                ApplyCausalMask && k_col_start >= q_start[0] + RM;
            const bool fully_future1 =
                ApplyCausalMask && k_col_start >= q_start[1] + RM;
            fully_future[0][gc] = fully_future0;
            fully_future[1][gc] = fully_future1;
            any_work = any_work || !(fully_future0 && fully_future1);
            if (fully_future0 && fully_future1) {
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                scores0[gc][elem] = -FLT_MAX * 0.5f;
                scores1[gc][elem] = -FLT_MAX * 0.5f;
              }
              continue;
            }

            float k_scale_local = prepared_k_scale_tile;
            if constexpr (!QuantizeKey && BC > 64) {
              if constexpr (!PerThreadQK) {
                const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                k_scale_local =
                    k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
              }
            }
            if constexpr (UseLaneMajorKey) {
              compute_tqk_score_regs_raw_kq_2_lane_shared_key<DTiles>(
                  k_lane_tile, q_regs, lane, col_tile,
                  qs[0] * k_scale_local, qs[1] * k_scale_local,
                  !fully_future0, !fully_future1,
                  scores0[gc], scores1[gc]);
            } else {
              compute_tqk_score_regs_raw_kq_2<PvOrderedQK, DTiles, SharedQKHeadStride>(
                  &k_tile[0][0], q_regs, lane, col_tile,
                  qs[0] * k_scale_local, qs[1] * k_scale_local,
                  !fully_future0, !fully_future1,
                  scores0[gc], scores1[gc]);
            }
            apply_per_thread_qk_score_scale<PerThreadQK, PvOrderedQK>(scores0[gc], q_scale, k_scale, b, hq, hkv, q_start[0], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            apply_per_thread_qk_score_scale<PerThreadQK, PvOrderedQK>(scores1[gc], q_scale, k_scale, b, hq, hkv, q_start[1], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            if constexpr (ApplyCausalMask) {
              if (fully_future0) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  scores0[gc][elem] = -FLT_MAX * 0.5f;
                }
              } else if (k_col_start + BK > q_start[0]) {
                if constexpr (PvOrderedQK) {
                  apply_tqk_causal_mask_pv_order<true>(
                      scores0[gc], static_cast<int>(q_start[0]),
                      static_cast<int>(kb_base), col_tile, lane);
                } else {
                  apply_tqk_causal_mask<true>(
                      scores0[gc], static_cast<int>(q_start[0]),
                      static_cast<int>(kb_base), col_tile, lane);
                }
              }
              if (fully_future1) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  scores1[gc][elem] = -FLT_MAX * 0.5f;
                }
              } else if (k_col_start + BK > q_start[1]) {
                if constexpr (PvOrderedQK) {
                  apply_tqk_causal_mask_pv_order<true>(
                      scores1[gc], static_cast<int>(q_start[1]),
                      static_cast<int>(kb_base), col_tile, lane);
                } else {
                  apply_tqk_causal_mask<true>(
                      scores1[gc], static_cast<int>(q_start[1]),
                      static_cast<int>(kb_base), col_tile, lane);
                }
              }
            }
            if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
              apply_tqk_kv_tail_mask<PvOrderedQK>(
                  scores0[gc], kv_len, kb_base, col_tile, lane);
              apply_tqk_kv_tail_mask<PvOrderedQK>(
                  scores1[gc], kv_len, kb_base, col_tile, lane);
            }
          }
          if (!any_work) {
            continue;
          }

          if constexpr (DirectStreamProbs) {
            float local_sums[QGroups];
#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              float local_max = -FLT_MAX * 0.5f;
#pragma unroll
              for (int gc = 0; gc < StreamGroupCols; ++gc) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  const float score = qg == 0 ? scores0[gc][elem] : scores1[gc][elem];
                  local_max = fmaxf(local_max, score);
                }
              }
              const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
              const float old_m = m[qg];
              const float new_m = fmaxf(old_m, tile_max);
              const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
              m[qg] = new_m;
              l[qg] *= alpha;
              local_sums[qg] = 0.0f;

              float8_vec alpha_rows;
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
              }

#pragma unroll
              for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  out_frag[qg][dt][elem] *= alpha_rows[elem];
                }
              }
            }

#pragma unroll
            for (int gc = 0; gc < StreamGroupCols; ++gc) {
              if (fully_future[0][gc] && fully_future[1][gc]) {
                continue;
              }
              half8_vec p_regs_current[QGroups];
#pragma unroll
              for (int qg = 0; qg < QGroups; ++qg) {
                half8_vec prob_values;
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  const float score = qg == 0 ? scores0[gc][elem] : scores1[gc][elem];
                  float prob = 0.0f;
                  if (!fully_future[qg][gc]) {
                    prob = fast_exp2(score - m[qg] + kF16SoftmaxOffset);
                    local_sums[qg] += prob;
                  }
                  prob_values[elem] = static_cast<_Float16>(prob);
                }
                if constexpr (PvOrderedQK) {
                  p_regs_current[qg] = prob_values;
                } else {
                  p_regs_current[qg] = make_p_regs_from_tqk_prob_regs(prob_values, lane);
                }
              }
              const int col_tile = group_base + gc;
              auto load_stream_v_regs = [&](const int dt) {
                half8_vec v_regs;
                if constexpr (UseLaneMajorValue) {
                  v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                      v_lane_tile, col_tile, dt, lane);
                } else if constexpr (UseTransposedValueLayout) {
                  v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                      &v_tile[0][0], col_tile, dt, lane);
                } else {
                  v_regs = make_v_regs_from_shared<SharedValueStride>(
                      &v_tile[0][0], col_tile, dt, lane);
                }
                return v_regs;
              };
              auto apply_stream_pv = [&](const int dt, const half8_vec v_regs) {
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  if (fully_future[qg][gc]) {
                    continue;
                  }
                  if constexpr (DirectPvOutFragAccum) {
                    if constexpr (F16PvAccum) {
                      out_frag[qg][dt] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                          p_regs_current[qg], v_regs, out_frag[qg][dt]);
                    } else {
                      out_frag[qg][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                          p_regs_current[qg], v_regs, out_frag[qg][dt]);
                    }
                  } else {
                  PvAccumVec acc;
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    acc[elem] = out_frag[qg][dt][elem];
                  }
                  if constexpr (F16PvAccum) {
                    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                        p_regs_current[qg], v_regs, acc);
                  } else {
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        p_regs_current[qg], v_regs, acc);
                  }
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    out_frag[qg][dt][elem] = acc[elem];
                  }
                  }
                }
              };
              if constexpr (PrefetchStreamVRegs) {
                half8_vec v_regs = load_stream_v_regs(0);
#pragma unroll
                for (int dt = 0; dt < DTiles - 1; ++dt) {
                  const half8_vec next_v_regs = load_stream_v_regs(dt + 1);
                  apply_stream_pv(dt, v_regs);
                  v_regs = next_v_regs;
                }
                apply_stream_pv(DTiles - 1, v_regs);
              } else {
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                  const half8_vec v_regs = load_stream_v_regs(dt);
                  apply_stream_pv(dt, v_regs);
                }
              }
            }
#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              l[qg] += local_sums[qg] + __shfl_xor(local_sums[qg], 16, 32);
            }
          } else {
            half8_vec p_regs[QGroups][StreamGroupCols];
#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              float local_max = -FLT_MAX * 0.5f;
#pragma unroll
              for (int gc = 0; gc < StreamGroupCols; ++gc) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  const float score = qg == 0 ? scores0[gc][elem] : scores1[gc][elem];
                  local_max = fmaxf(local_max, score);
                }
              }
              const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
              const float old_m = m[qg];
              const float new_m = fmaxf(old_m, tile_max);
              const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
              m[qg] = new_m;
              l[qg] *= alpha;

              float8_vec alpha_rows;
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
              }

#pragma unroll
              for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  out_frag[qg][dt][elem] *= alpha_rows[elem];
                }
              }

              float local_sum = 0.0f;
#pragma unroll
              for (int gc = 0; gc < StreamGroupCols; ++gc) {
                half8_vec prob_values;
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  const float score = qg == 0 ? scores0[gc][elem] : scores1[gc][elem];
                  float prob = 0.0f;
                  if (!fully_future[qg][gc]) {
                    prob = fast_exp2(score - m[qg] + kF16SoftmaxOffset);
                    local_sum += prob;
                  }
                  prob_values[elem] = static_cast<_Float16>(prob);
                }
                if constexpr (PvOrderedQK) {
                  p_regs[qg][gc] = prob_values;
                } else {
                  p_regs[qg][gc] = make_p_regs_from_tqk_prob_regs(prob_values, lane);
                }
              }
              l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
            }

#pragma unroll
            for (int gc = 0; gc < StreamGroupCols; ++gc) {
              if (fully_future[0][gc] && fully_future[1][gc]) {
                continue;
              }
              const int col_tile = group_base + gc;
              auto load_stream_v_regs = [&](const int dt) {
                half8_vec v_regs;
                if constexpr (UseLaneMajorValue) {
                  v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                      v_lane_tile, col_tile, dt, lane);
                } else if constexpr (UseTransposedValueLayout) {
                  v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                      &v_tile[0][0], col_tile, dt, lane);
                } else {
                  v_regs = make_v_regs_from_shared<SharedValueStride>(
                      &v_tile[0][0], col_tile, dt, lane);
                }
                return v_regs;
              };
              auto apply_stream_pv = [&](const int dt, const half8_vec v_regs) {
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  if (fully_future[qg][gc]) {
                    continue;
                  }
                  if constexpr (DirectPvOutFragAccum) {
                    if constexpr (F16PvAccum) {
                      out_frag[qg][dt] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                          p_regs[qg][gc], v_regs, out_frag[qg][dt]);
                    } else {
                      out_frag[qg][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                          p_regs[qg][gc], v_regs, out_frag[qg][dt]);
                    }
                  } else {
                  PvAccumVec acc;
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    acc[elem] = out_frag[qg][dt][elem];
                  }
                  if constexpr (F16PvAccum) {
                    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                        p_regs[qg][gc], v_regs, acc);
                  } else {
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        p_regs[qg][gc], v_regs, acc);
                  }
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    out_frag[qg][dt][elem] = acc[elem];
                  }
                  }
                }
              };
              if constexpr (PrefetchStreamVRegs) {
                half8_vec v_regs = load_stream_v_regs(0);
#pragma unroll
                for (int dt = 0; dt < DTiles - 1; ++dt) {
                  const half8_vec next_v_regs = load_stream_v_regs(dt + 1);
                  apply_stream_pv(dt, v_regs);
                  v_regs = next_v_regs;
                }
                apply_stream_pv(DTiles - 1, v_regs);
              } else {
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                  const half8_vec v_regs = load_stream_v_regs(dt);
                  apply_stream_pv(dt, v_regs);
                }
              }
            }
          }
        }
        return;
      }
      half8_vec prob_cache[QGroups][ColTiles];
      if constexpr (QuantizeQuery || UseRawPreparedQ) {
        float8_vec score_cache[QGroups][ColTiles];
        float local_max[QGroups];
#pragma unroll
        for (int qg = 0; qg < QGroups; ++qg) {
          local_max[qg] = -FLT_MAX * 0.5f;
        }
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future0 =
              ApplyCausalMask && k_col_start >= q_start[0] + RM;
          const bool fully_future1 =
              ApplyCausalMask && k_col_start >= q_start[1] + RM;
          float8_vec scores0;
          float8_vec scores1;
          if (fully_future0 && fully_future1) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              scores0[elem] = -FLT_MAX * 0.5f;
              scores1[elem] = -FLT_MAX * 0.5f;
            }
          } else {
            float k_scale_local = prepared_k_scale_tile;
            if constexpr (!QuantizeKey && BC > 64) {
              if constexpr (!PerThreadQK) {
                const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                k_scale_local =
                    k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
              }
            }
            if constexpr (UseLaneMajorKey) {
              compute_tqk_score_regs_raw_kq_2_lane_shared_key<DTiles>(
                  k_lane_tile, q_regs, lane, col_tile,
                  qs[0] * k_scale_local, qs[1] * k_scale_local,
                  !fully_future0, !fully_future1, scores0, scores1);
            } else {
              compute_tqk_score_regs_raw_kq_2<PvOrderedQK, DTiles, SharedQKHeadStride>(
                  &k_tile[0][0], q_regs, lane, col_tile,
                  qs[0] * k_scale_local, qs[1] * k_scale_local,
                  !fully_future0, !fully_future1, scores0, scores1);
            }
            apply_per_thread_qk_score_scale<PerThreadQK, PvOrderedQK>(scores0, q_scale, k_scale, b, hq, hkv, q_start[0], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            apply_per_thread_qk_score_scale<PerThreadQK, PvOrderedQK>(scores1, q_scale, k_scale, b, hq, hkv, q_start[1], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            if constexpr (ApplyCausalMask) {
              if (fully_future0) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  scores0[elem] = -FLT_MAX * 0.5f;
                }
              } else if (k_col_start + BK > q_start[0]) {
                if constexpr (PvOrderedQK) {
                  apply_tqk_causal_mask_pv_order<true>(
                      scores0, static_cast<int>(q_start[0]), static_cast<int>(kb_base),
                      col_tile, lane);
                } else {
                  apply_tqk_causal_mask<true>(
                      scores0, static_cast<int>(q_start[0]), static_cast<int>(kb_base),
                      col_tile, lane);
                }
              }
              if (fully_future1) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  scores1[elem] = -FLT_MAX * 0.5f;
                }
              } else if (k_col_start + BK > q_start[1]) {
                if constexpr (PvOrderedQK) {
                  apply_tqk_causal_mask_pv_order<true>(
                      scores1, static_cast<int>(q_start[1]), static_cast<int>(kb_base),
                      col_tile, lane);
                } else {
                  apply_tqk_causal_mask<true>(
                      scores1, static_cast<int>(q_start[1]), static_cast<int>(kb_base),
                      col_tile, lane);
                }
              }
            }
          }
          if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
            apply_tqk_kv_tail_mask<PvOrderedQK>(
                scores0, kv_len, kb_base, col_tile, lane);
            apply_tqk_kv_tail_mask<PvOrderedQK>(
                scores1, kv_len, kb_base, col_tile, lane);
          }
          score_cache[0][col_tile] = scores0;
          score_cache[1][col_tile] = scores1;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            local_max[0] = fmaxf(local_max[0], scores0[elem]);
            local_max[1] = fmaxf(local_max[1], scores1[elem]);
          }
        }
#pragma unroll
        for (int qg = 0; qg < QGroups; ++qg) {
          const float tile_max = fmaxf(local_max[qg], __shfl_xor(local_max[qg], 16, 32));
          const float old_m = m[qg];
          const float new_m = fmaxf(old_m, tile_max);
          const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
          m[qg] = new_m;
          l[qg] *= alpha;

          float8_vec alpha_rows;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
          }

#pragma unroll
          for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              out_frag[qg][dt][elem] *= alpha_rows[elem];
            }
          }

          float local_sum = 0.0f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          half8_vec prob_values;
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future =
                ApplyCausalMask && k_col_start >= q_start[qg] + RM;
          const float8_vec scores = score_cache[qg][col_tile];
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              float prob = 0.0f;
              if (!fully_future) {
                prob = fast_exp2(scores[elem] - m[qg] + kF16SoftmaxOffset);
                local_sum += prob;
              }
              prob_values[elem] = static_cast<_Float16>(prob);
            }
            if constexpr (PvOrderedQK) {
              prob_cache[qg][col_tile] = prob_values;
            } else {
              prob_cache[qg][col_tile] = make_p_regs_from_tqk_prob_regs(prob_values, lane);
            }
          }
          l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
        }
      } else {
#pragma unroll
        for (int qg = 0; qg < QGroups; ++qg) {
          float8_vec score_cache[ColTiles];
          float local_max = -FLT_MAX * 0.5f;
#pragma unroll
          for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
            const int64_t k_col_start = kb_base + col_tile * BK;
            const bool fully_future =
                ApplyCausalMask && k_col_start >= q_start[qg] + RM;
            float8_vec scores;
            if (fully_future) {
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                scores[elem] = -FLT_MAX * 0.5f;
              }
            } else {
              float k_scale_local = prepared_k_scale_tile;
              if constexpr (!QuantizeKey && BC > 64) {
                if constexpr (!PerThreadQK) {
                  const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                  k_scale_local =
                      k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
                }
              }
              const float score_scale = qs[qg] * k_scale_local;
              scores =
                  compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                      &k_tile[0][0], q_frag[qg], col_tile, score_scale);
              apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                  col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                  ks_stride_b, ks_stride_h, sm_scale);
              if constexpr (ApplyCausalMask) {
                const bool needs_causal_mask = k_col_start + BK > q_start[qg];
                if (needs_causal_mask) {
                  apply_tqk_causal_mask<true>(
                      scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base),
                      col_tile, lane);
                }
              }
            }
            if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
              apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
            }
            score_cache[col_tile] = scores;
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              local_max = fmaxf(local_max, scores[elem]);
            }
          }
          const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
          const float old_m = m[qg];
          const float new_m = fmaxf(old_m, tile_max);
          const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
          m[qg] = new_m;
          l[qg] *= alpha;

          float8_vec alpha_rows;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
          }

#pragma unroll
          for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              out_frag[qg][dt][elem] *= alpha_rows[elem];
            }
          }

          float local_sum = 0.0f;
#pragma unroll
          for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
            half8_vec prob_values;
            const int64_t k_col_start = kb_base + col_tile * BK;
            const bool fully_future =
                ApplyCausalMask && k_col_start >= q_start[qg] + RM;
            const float8_vec scores = score_cache[col_tile];
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              float prob = 0.0f;
              if (!fully_future) {
                prob = fast_exp2(scores[elem] - m[qg] + kF16SoftmaxOffset);
                local_sum += prob;
              }
              prob_values[elem] = static_cast<_Float16>(prob);
            }
            prob_cache[qg][col_tile] = make_p_regs_from_tqk_prob_regs(prob_values, lane);
          }
          l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
        }
      }

      SAGEATTN_F16_SCHED_BARRIER(0);
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        if constexpr (ApplyCausalMask) {
          if (kb_base + col_tile * BK >= q_start[QGroups - 1] + RM) {
            continue;
          }
        }
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          half8_vec v_regs;
          if constexpr (UseLaneMajorValue) {
            v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                v_lane_tile, col_tile, dt, lane);
          } else if constexpr (UseTransposedValueLayout) {
            v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          } else {
            v_regs = make_v_regs_from_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          }
          const bool fully_future0 =
              ApplyCausalMask && kb_base + col_tile * BK >= q_start[0] + RM;
          const half8_vec p_regs0 = prob_cache[0][col_tile];
          if (!fully_future0) {
          PvAccumVec acc0;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            acc0[elem] = out_frag[0][dt][elem];
          }
          if constexpr (F16PvAccum) {
            acc0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                p_regs0, v_regs, acc0);
          } else {
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                p_regs0, v_regs, acc0);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[0][dt][elem] = acc0[elem];
          }
          }
          const bool fully_future1 =
              ApplyCausalMask && kb_base + col_tile * BK >= q_start[1] + RM;
          const half8_vec p_regs1 = prob_cache[1][col_tile];
          if (!fully_future1) {
          PvAccumVec acc1;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            acc1[elem] = out_frag[1][dt][elem];
          }
          if constexpr (F16PvAccum) {
            acc1 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                p_regs1, v_regs, acc1);
          } else {
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                p_regs1, v_regs, acc1);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[1][dt][elem] = acc1[elem];
          }
          }
        }
      }
    } else {
      const bool needs_causal_mask = ApplyCausalMask && (kb_base + BC > q_base);
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
        float local_max = -FLT_MAX * 0.5f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          float k_scale_local = prepared_k_scale_tile;
          if constexpr (!QuantizeKey && BC > 64) {
            if constexpr (!PerThreadQK) {
              const int k_scale_idx = k_scale_col_per_warp(kb_base + col_tile * BK);
              k_scale_local =
                  k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
            }
          }
          const float score_scale = qs[qg] * k_scale_local;
          float8_vec scores;
          if constexpr (QuantizeQuery || UseRawPreparedQ) {
            scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
                &k_tile[0][0], q_regs[qg], lane, col_tile, score_scale);
          } else {
            scores =
                compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                    &k_tile[0][0], q_frag[qg], col_tile, score_scale);
          }
          apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
              col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
              ks_stride_b, ks_stride_h, sm_scale);
          if constexpr (ApplyCausalMask) {
            if (needs_causal_mask) {
              apply_tqk_causal_mask<true>(
                  scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base), col_tile, lane);
            }
          }
          if constexpr (!NoKvTail) if (kb_base + static_cast<int64_t>(col_tile) * BK + BK > kv_len) {
            apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            local_max = fmaxf(local_max, scores[elem]);
          }
        }
        const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        const float old_m = m[qg];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
        m[qg] = new_m;
        l[qg] *= alpha;

        float8_vec alpha_rows;
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
        }

#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[qg][dt][elem] *= alpha_rows[elem];
          }
        }

        float local_sum = 0.0f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          float k_scale_local = prepared_k_scale_tile;
          if constexpr (!QuantizeKey && BC > 64) {
            if constexpr (!PerThreadQK) {
              const int k_scale_idx = k_scale_col_per_warp(kb_base + col_tile * BK);
              k_scale_local =
                  k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
            }
          }
          const float score_scale = qs[qg] * k_scale_local;
          float8_vec scores;
          if constexpr (QuantizeQuery || UseRawPreparedQ) {
            scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
                &k_tile[0][0], q_regs[qg], lane, col_tile, score_scale);
          } else {
            scores =
                compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                    &k_tile[0][0], q_frag[qg], col_tile, score_scale);
          }
          apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
              col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
              ks_stride_b, ks_stride_h, sm_scale);
          if constexpr (ApplyCausalMask) {
            if (needs_causal_mask) {
              apply_tqk_causal_mask<true>(
                  scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base), col_tile, lane);
            }
          }
          if constexpr (!NoKvTail) if (kb_base + static_cast<int64_t>(col_tile) * BK + BK > kv_len) {
            apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
          }
          half8_vec prob_values;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            const float prob = fast_exp2(scores[elem] - m[qg] + kF16SoftmaxOffset);
            local_sum += prob;
            prob_values[elem] = static_cast<_Float16>(prob);
          }

          const half8_vec p_regs = make_p_regs_from_tqk_prob_regs(prob_values, lane);
#pragma unroll
          for (int dt = 0; dt < DTiles; ++dt) {
            half8_vec v_regs;
            if constexpr (UseLaneMajorValue) {
              v_regs = make_v_regs_from_lane_major_shared<DTiles>(
                  v_lane_tile, col_tile, dt, lane);
            } else if constexpr (UseTransposedValueLayout) {
              v_regs = make_v_regs_from_transposed_shared<SharedValueStride>(
                  &v_tile[0][0], col_tile, dt, lane);
          } else {
            v_regs = make_v_regs_from_shared<SharedValueStride>(
                &v_tile[0][0], col_tile, dt, lane);
          }
            PvAccumVec acc;
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              acc[elem] = out_frag[qg][dt][elem];
            }
            if constexpr (F16PvAccum) {
              acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                  p_regs, v_regs, acc);
            } else {
              acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                  p_regs, v_regs, acc);
            }
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              out_frag[qg][dt][elem] = acc[elem];
            }
          }
        }
        l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
      }
    }
    };
    compute_loaded_tile(apply_causal_mask_tag);
    if constexpr (UsesTileSharedMemory) {
      __syncthreads();
    }
  };

  if constexpr (IsCausal) {
    const int64_t prefix_limit = ExactStaticCausalBlock ?
        q_base : (((q_base / BC) * BC) < kv_limit ? ((q_base / BC) * BC) : kv_limit);
#pragma unroll 2
    for (int64_t kb_base = 0; kb_base < prefix_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::false_type{});
    }
#pragma unroll 2
    for (int64_t kb_base = prefix_limit; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::true_type{});
    }
  } else {
#pragma unroll 2
    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::false_type{});
    }
  }

#pragma unroll
  for (int qg = 0; qg < QGroups; ++qg) {
    float8_vec inv_l_rows;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const float l_sum = __shfl(l[qg], row_base + elem, 32);
      inv_l_rows[elem] = l_sum == 0.0f ? 0.0f : 1.0f / l_sum;
    }
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
      const int d = dt * BK + col;
#pragma unroll
      for (int pair = 0; pair < PackedRows; ++pair) {
        const int elem = pair * 2;
        const int64_t q_idx0 = q_start[qg] + row_base + elem;
        const int64_t q_idx1 = q_idx0 + 1;
        const float value0 = static_cast<float>(out_frag[qg][dt][elem]) * inv_l_rows[elem];
        const float value1 = static_cast<float>(out_frag[qg][dt][elem + 1]) * inv_l_rows[elem + 1];
        const int64_t o_off0 = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx0, d, o_stride_b, o_stride_n, o_stride_h);
        const int64_t o_off1 = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx1, d, o_stride_b, o_stride_n, o_stride_h);
        if constexpr (NoQueryTail) {
          store_half(output, o_off0, value0);
          store_half(output, o_off1, value1);
        } else {
          if (q_idx0 < qo_len) {
            store_half(output, o_off0, value0);
          }
          if (q_idx1 < qo_len) {
            store_half(output, o_off1, value1);
          }
        }
      }
    }
  }
    return;
  }
}

template <int BlockCols,
          int HeadDim,
          int ValueTileBase,
          int ValueTiles,
          bool HndContiguous = false,
          int BlockRows = 128,
          bool ValueTransposed = false,
          bool IsCausal = false,
          typename OutputT = __half,
          typename QueryT = int8_t,
          bool QuantizeQuery = false,
          typename KeyT = int8_t,
          typename ValueT = uint8_t,
          bool QuantizeKeyValue = false,
          bool PrepackedLaneMajorKV = false,
          int StreamColsOverride = 0,
          bool PrepackedLaneMajorKeyOnly = false,
          bool PrepackedLaneMajorValueOnly = false,
          int QGroupsParam = 2,
          bool LowPressureQGroups = false,
          bool PerThreadQK = false,
          bool KeyHndContiguous = HndContiguous,
          bool StaticNhdLayout = false,
          bool NoKvTail = false,
          bool SameQKHeads = false,
          bool NoQueryTail = false,
          bool InvLRowsEpilogue = false>
SAGEATTN_NATIVE_2Q_WAVES_PER_EU(HeadDim, IsCausal)
__global__ __launch_bounds__(BlockRows * (2 / QGroupsParam), 1) void qk_int8_sv_f8_native_2q_kernel(
    const QueryT* __restrict__ q,
    const KeyT* __restrict__ k,
    const ValueT* __restrict__ v,
    OutputT* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout,
    const float sm_scale,
    const bool per_thread_qk = false) {
  static_assert(HeadDim == 16 || HeadDim == 64 || HeadDim == 128 || HeadDim == 256,
                "native gfx12 fp8 2q kernel supports D16/D64/D128/D256.");
  static_assert(BlockCols == 16 || BlockCols == 32 || BlockCols == 64 ||
                    BlockCols == 128,
                "native gfx12 fp8 2q kernel supports BC16/BC32/BC64/BC128.");
  static_assert(BlockRows == 32 || BlockRows == 64 || BlockRows == 128 ||
                    BlockRows == 256 || BlockRows == 512,
                "native fp8 2q supports 32, 64, 128, 256, or 512 query rows per CTA.");
  static_assert(QGroupsParam == 1 || QGroupsParam == 2,
                "native fp8 kernel supports one or two query groups per wave.");
  static_assert(!LowPressureQGroups ||
                    (HeadDim == 128 && BlockCols == 64 && BlockRows == 128 &&
                     IsCausal && ValueTransposed && QGroupsParam == 2),
                "low-pressure fp8 path is specialized for D128 BC64 BR128 causal.");
  constexpr int BR = BlockRows;
  constexpr int RM = 16;
  constexpr int QGroups = QGroupsParam;
  constexpr int RowsPerWave = RM * QGroups;
  constexpr int BK = 16;
  constexpr int BC = BlockCols;
  constexpr int Threads = BlockRows * (2 / QGroups);
  constexpr int DTiles = HeadDim / BK;
  constexpr int ColTiles = BC / BK;
  constexpr int SharedHeadStride = HeadDim + 16;
  constexpr bool PackedTransposedValue = ValueTransposed && HeadDim == 64;
  constexpr int SharedValueRows = ValueTransposed && !PackedTransposedValue ? HeadDim : (ValueTransposed ? 1 : BC);
  constexpr int SharedValueStride = ValueTransposed && !PackedTransposedValue ? (BC + 16) : (ValueTransposed ? 1 : SharedHeadStride);
  constexpr int SharedValueRowsT = PackedTransposedValue ? HeadDim : 1;
  constexpr int SharedValueStrideT = PackedTransposedValue ? (BC / 4 + 1) : 1;
  constexpr int PackedRows = 4;
  constexpr bool UseLowPressureQGroups =
      LowPressureQGroups && HeadDim == 128 && BlockCols == 64 &&
      BlockRows == 128 && IsCausal && ValueTransposed &&
      !QuantizeQuery && !QuantizeKeyValue && QGroups == 2;
  constexpr bool UsePrepackedLaneMajorKV =
      PrepackedLaneMajorKV && HeadDim == 64 &&
      BlockCols == 64 && HndContiguous &&
      ValueTransposed && !QuantizeKeyValue;
  constexpr bool UsePrepackedLaneMajorKey =
      PrepackedLaneMajorKeyOnly && (HeadDim == 64 || HeadDim == 128) &&
      BlockCols == 64 && HndContiguous &&
      ValueTransposed && !QuantizeKeyValue;
  constexpr bool UsePrepackedLaneMajorK =
      UsePrepackedLaneMajorKV || UsePrepackedLaneMajorKey;
  constexpr bool UsePrepackedLaneMajorValue =
      UsePrepackedLaneMajorKV ||
      (PrepackedLaneMajorValueOnly && (HeadDim == 64 || HeadDim == 128) &&
       BlockCols == 64 && HndContiguous &&
       ValueTransposed && !QuantizeKeyValue);
  constexpr bool UsesTileSharedMemory =
      !UsePrepackedLaneMajorK || !UsePrepackedLaneMajorValue;
  constexpr bool PreloadQFragments =
      !UseLowPressureQGroups && !UsePrepackedLaneMajorK && (HeadDim == 64 ||
      (HeadDim == 128 && IsCausal && BlockRows == 128 &&
       (BlockCols == 64 || BlockCols == 128) &&
       ValueTransposed));
  constexpr bool UseStreamedFp8Pv =
      StreamColsOverride >= 0 &&
      (BlockCols == 64 || (BlockCols == 128 && HeadDim == 128)) &&
      ValueTransposed &&
      (QuantizeQuery || (HeadDim == 128 && BlockCols == 128 &&
                         PreloadQFragments)) &&
      (HeadDim == 128 ||
       (HeadDim == 64 && IsCausal &&
        (UsePrepackedLaneMajorK || UsePrepackedLaneMajorValue)));
  static_assert(!PrepackedLaneMajorKV || HndContiguous,
                "prepacked transposed fp8 K/V requires contiguous HND tensors.");
  static_assert(!QuantizeKeyValue ||
                    ((HeadDim == 64 || HeadDim == 128) &&
                     BlockCols == 64 && !ValueTransposed),
                "raw K/V fp8 staging currently supports D64/D128 BC64 tensors.");
  static_assert(!UsePrepackedLaneMajorKV ||
                    (HeadDim == 64 && HndContiguous && ValueTransposed &&
                     !QuantizeKeyValue),
                "lane-major prepared fp8 K/V requires prepared transposed D64 HND tensors.");
  static_assert(!UsePrepackedLaneMajorKey ||
                    ((HeadDim == 64 || HeadDim == 128) &&
                     HndContiguous && ValueTransposed &&
                      !QuantizeKeyValue),
                "lane-major prepared fp8 K requires prepared transposed D64/D128 HND tensors.");
  static_assert(!UsePrepackedLaneMajorValue ||
                    (HndContiguous && ValueTransposed && !QuantizeKeyValue),
                "lane-major prepared fp8 V requires prepared transposed HND tensors.");
  static_assert(ValueTiles == 1 || ValueTiles == 4 || ValueTiles == 8,
                "native fp8 2q stores one D16, D64, or D128 value slice per launch.");
  static_assert(ValueTileBase + ValueTiles <= DTiles, "invalid fp8 value tile slice.");
  static_assert(!NoQueryTail || StaticNhdLayout,
                "full-query fp8 path requires a static dispatch.");

  __shared__ int8_t k_tile[UsePrepackedLaneMajorK ? 1 : BC][SharedHeadStride];
  __shared__ uint8_t v_tile[UsePrepackedLaneMajorValue ? 1 : SharedValueRows]
                           [UsePrepackedLaneMajorValue ? 1 : SharedValueStride];
  __shared__ uint32_t v_tile_t[UsePrepackedLaneMajorValue ? 1 : SharedValueRowsT]
                               [UsePrepackedLaneMajorValue ? 1 : SharedValueStrideT];
  __shared__ float raw_k_amax_shared;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int row_base = (lane >> 4) << 3;
  const int col = lane & 15;
  const int64_t q_base =
      q_block_base_for_launch<IsCausal, BR>(static_cast<int64_t>(blockIdx.x), qo_len);
  const int64_t hq = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) {
    return;
  }

  const int64_t hkv = SameQKHeads ? hq : hq / (num_qo_heads / num_kv_heads);
  const int64_t k_head_base = b * k_stride_b + hkv * k_stride_h;
  const int64_t v_head_base = b * v_stride_b + hkv * v_stride_h;
  int64_t q_start[QGroups];
  float qs[QGroups];
  const int64_t wave_q_start = q_base + static_cast<int64_t>(wave) * RowsPerWave;
#pragma unroll
  for (int qg = 0; qg < QGroups; ++qg) {
    q_start[qg] = wave_q_start + qg * RM;
  }

  i32x2_vec q_regs[QGroups][DTiles];
  if constexpr (QuantizeQuery) {
    constexpr int QPackElems = 8;
    constexpr int QPacksPerWave = (RowsPerWave * HeadDim) / QPackElems;
    const int local_q_row_base = wave * RowsPerWave;
    float local_q_amax = 0.0000001f;
    for (int pack = lane; pack < QPacksPerWave; pack += 32) {
      const int elem_base = pack * QPackElems;
      const int row = elem_base / HeadDim;
      const int d = elem_base - row * HeadDim;
      const int64_t q_idx = q_base + local_q_row_base + row;
      if constexpr (NoQueryTail) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
        const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
        for (int i = 0; i < QPackElems; ++i) {
          local_q_amax = fmaxf(local_q_amax, fabsf(value_to_float(values[i])));
        }
      } else if (q_idx < qo_len) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx, d, q_stride_b, q_stride_n, q_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(q + q_off);
        const QueryT* values = reinterpret_cast<const QueryT*>(&raw);
#pragma unroll
        for (int i = 0; i < QPackElems; ++i) {
          local_q_amax = fmaxf(local_q_amax, fabsf(value_to_float(values[i])));
        }
      }
    }
    local_q_amax = vllm::warpReduceMax(local_q_amax);
    float q_amax_for_scale = __shfl(local_q_amax, 0, 32);
    if constexpr (QGroups == 1) {
      __shared__ float q_amax_shared[Threads / 32];
      if (lane == 0) {
        q_amax_shared[wave] = q_amax_for_scale;
      }
      __syncthreads();
      const int pair_wave = wave & ~1;
      q_amax_for_scale =
          fmaxf(q_amax_shared[pair_wave], q_amax_shared[pair_wave + 1]);
      __syncthreads();
    }
    const float q_scale_local = q_amax_for_scale / 127.0f;
    const float inv_q_scale = 127.0f / q_amax_for_scale;
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
      qs[qg] = q_scale_local * sm_scale * kLog2e;
      const int64_t qg_start = q_start[qg];
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        q_regs[qg][dt] =
            pack_quant_q_i8_wmma_b_regs<QueryT, HeadDim, HndContiguous, StaticNhdLayout, NoQueryTail>(
            q, tensor_layout, lane, b, hq, qg_start, qo_len, dt * BK,
            q_stride_b, q_stride_n, q_stride_h, inv_q_scale);
      }
    }
  } else {
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
      if constexpr (PerThreadQK) {
        qs[qg] = 1.0f;
      } else {
        const int q_scale_idx = q_scale_col_per_warp(q_start[qg]);
        qs[qg] = q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx] *
            sm_scale * kLog2e;
      }
    }
    if constexpr (UsePrepackedLaneMajorK) {
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          q_regs[qg][dt] =
              pack_q_i8_wmma_b_regs<HeadDim, HndContiguous, StaticNhdLayout, NoQueryTail>(
              q, tensor_layout, lane, b, hq, q_start[qg], qo_len, dt * BK,
              q_stride_b, q_stride_n, q_stride_h);
        }
      }
    }
  }

  using FragK = rocwmma::fragment<rocwmma::matrix_a, RM, BK, BK, int8_t, rocwmma::row_major>;
  using FragQ = rocwmma::fragment<rocwmma::matrix_b, RM, BK, BK, int8_t, rocwmma::col_major>;
  using FragScoreT = rocwmma::fragment<rocwmma::accumulator, RM, BK, BK, int32_t>;

  FragQ q_frag[QGroups][DTiles];
  if constexpr (PreloadQFragments && !QuantizeQuery) {
#pragma unroll
    for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
      for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_start[qg], dt * BK, q_stride_b, q_stride_n, q_stride_h);
        rocwmma::load_matrix_sync(q_frag[qg][dt], q + q_off, static_cast<uint32_t>(q_stride_n));
      }
    }
  }

  float out_frag[QGroups][ValueTiles][8];
  float m[QGroups];
  float l[QGroups];
#pragma unroll
  for (int qg = 0; qg < QGroups; ++qg) {
    m[qg] = -FLT_MAX * 0.5f;
    l[qg] = 0.0f;
#pragma unroll
    for (int vdt = 0; vdt < ValueTiles; ++vdt) {
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        out_frag[qg][vdt][elem] = 0.0f;
      }
    }
  }

  const int64_t kv_limit = IsCausal && (q_base + BR) < kv_len ? q_base + BR : kv_len;
  auto process_kv_tile = [&](const int64_t kb_base, auto causal_mask_tag) {
    constexpr bool ApplyCausalMask = decltype(causal_mask_tag)::value;
    constexpr int VecBytes = 16;
    constexpr int VecsPerRow = HeadDim / VecBytes;
    constexpr bool UseActiveCausalColSkip = false;
    if constexpr (UsePrepackedLaneMajorK) {
      if constexpr (ApplyCausalMask) {
        if (kb_base >= q_start[QGroups - 1] + RM) {
          return;
        }
      }
    }
    float k_scale_tile = 1.0f;
    if constexpr (QuantizeKeyValue) {
      constexpr int PackElems = 8;
      constexpr int Packs = (BC * HeadDim) / PackElems;
      float local_k_amax = 0.0000001f;
      for (int pack = tid; pack < Packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int n = elem_base / HeadDim;
        const int d = elem_base - n * HeadDim;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, KeyHndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        const uint4 raw = *reinterpret_cast<const uint4*>(k + k_off);
        const KeyT* values = reinterpret_cast<const KeyT*>(&raw);
#pragma unroll
        for (int i = 0; i < PackElems; ++i) {
          local_k_amax = fmaxf(local_k_amax, fabsf(value_to_float(values[i])));
        }
      }
      const float block_k_amax = vllm::blockReduceMax(local_k_amax);
      if (tid == 0) {
        raw_k_amax_shared = block_k_amax;
      }
      __syncthreads();
      const float raw_k_amax = raw_k_amax_shared;
      k_scale_tile = raw_k_amax / 127.0f;
      const float inv_k_scale = 127.0f / raw_k_amax;

      for (int pack = tid; pack < Packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int n = elem_base / HeadDim;
        const int d = elem_base - n * HeadDim;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, KeyHndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
        const uint4 raw_k = *reinterpret_cast<const uint4*>(k + k_off);
        const uint4 raw_v = *reinterpret_cast<const uint4*>(v + v_off);
        const KeyT* k_values = reinterpret_cast<const KeyT*>(&raw_k);
        const ValueT* v_values = reinterpret_cast<const ValueT*>(&raw_v);
        char4 out0;
        char4 out1;
        out0.x = float_to_int8_rn_gfx12(value_to_float(k_values[0]) * inv_k_scale);
        out0.y = float_to_int8_rn_gfx12(value_to_float(k_values[1]) * inv_k_scale);
        out0.z = float_to_int8_rn_gfx12(value_to_float(k_values[2]) * inv_k_scale);
        out0.w = float_to_int8_rn_gfx12(value_to_float(k_values[3]) * inv_k_scale);
        out1.x = float_to_int8_rn_gfx12(value_to_float(k_values[4]) * inv_k_scale);
        out1.y = float_to_int8_rn_gfx12(value_to_float(k_values[5]) * inv_k_scale);
        out1.z = float_to_int8_rn_gfx12(value_to_float(k_values[6]) * inv_k_scale);
        out1.w = float_to_int8_rn_gfx12(value_to_float(k_values[7]) * inv_k_scale);
        *reinterpret_cast<char4*>(&k_tile[n][d]) = out0;
        *reinterpret_cast<char4*>(&k_tile[n][d + 4]) = out1;

        const uint32_t v_pack0 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[0]), value_to_float(v_values[1]),
            value_to_float(v_values[2]), value_to_float(v_values[3])));
        const uint32_t v_pack1 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[4]), value_to_float(v_values[5]),
            value_to_float(v_values[6]), value_to_float(v_values[7])));
        *reinterpret_cast<uint32_t*>(&v_tile[n][d]) = v_pack0;
        *reinterpret_cast<uint32_t*>(&v_tile[n][d + 4]) = v_pack1;
      }
    } else if constexpr (!UsePrepackedLaneMajorK) {
      for (int vec = tid; vec < BC * VecsPerRow; vec += Threads) {
        const int n = vec / VecsPerRow;
        const int d = (vec - n * VecsPerRow) * VecBytes;
        const int64_t k_off = qkv_offset_dispatch<HeadDim, KeyHndContiguous, StaticNhdLayout>(
            tensor_layout, b, hkv, kb_base + n, d, k_stride_b, k_stride_n, k_stride_h);
        *reinterpret_cast<uint4*>(&k_tile[n][d]) =
            *reinterpret_cast<const uint4*>(k + k_off);
        if constexpr (!ValueTransposed) {
          const int64_t v_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
              tensor_layout, b, hkv, kb_base + n, d, v_stride_b, v_stride_n, v_stride_h);
          *reinterpret_cast<uint4*>(&v_tile[n][d]) =
              *reinterpret_cast<const uint4*>(v + v_off);
        }
      }
    }
    if constexpr (ValueTransposed && !QuantizeKeyValue && !UsePrepackedLaneMajorValue) {
      constexpr int VVecBytes = 16;
      constexpr int VVecsPerD = BC / VVecBytes;
      for (int vec = tid; vec < HeadDim * VVecsPerD; vec += Threads) {
        const int d = vec / VVecsPerD;
        const int n = (vec - d * VVecsPerD) * VVecBytes;
        const int64_t v_off = b * v_stride_b + hkv * v_stride_h +
            static_cast<int64_t>(d) * v_stride_n + kb_base + n;
        const uint4 packed = *reinterpret_cast<const uint4*>(v + v_off);
        if constexpr (PackedTransposedValue) {
          const int group = n >> 2;
          v_tile_t[d][group + 0] = packed.x;
          v_tile_t[d][group + 1] = packed.y;
          v_tile_t[d][group + 2] = packed.z;
          v_tile_t[d][group + 3] = packed.w;
        } else {
          *reinterpret_cast<uint4*>(&v_tile[d][n]) = packed;
        }
      }
    }
    if constexpr (UsesTileSharedMemory) {
      __syncthreads();
    }

    if constexpr (!PreloadQFragments && !QuantizeQuery && !UsePrepackedLaneMajorK) {
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
          const int64_t q_off = qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
              tensor_layout, b, hq, q_start[qg], dt * BK, q_stride_b, q_stride_n, q_stride_h);
          rocwmma::load_matrix_sync(q_frag[qg][dt], q + q_off, static_cast<uint32_t>(q_stride_n));
        }
      }
    }

    int active_cols0 = ColTiles;
    int active_cols1 = ColTiles;
    if constexpr (UseActiveCausalColSkip) {
      active_cols0 =
          active_causal_col_tiles<ApplyCausalMask, ColTiles>(q_start[0], kb_base);
      active_cols1 =
          active_causal_col_tiles<ApplyCausalMask, ColTiles>(q_start[1], kb_base);
    }
    const int active_cols_any = active_cols0 > active_cols1 ? active_cols0 : active_cols1;

    if constexpr (BlockCols <= 64 || UseStreamedFp8Pv) {
      i32x2_vec prob_cache[QGroups][ColTiles];
      if constexpr (QuantizeQuery || UsePrepackedLaneMajorKV || UseStreamedFp8Pv) {
        if constexpr (UseStreamedFp8Pv) {
          constexpr int StreamCols = StreamColsOverride > 0 ? StreamColsOverride : 2;
          static_assert(StreamCols == 1 || StreamCols == 2 || StreamCols == 4,
                        "fp8 streaming supports one, two, or four col tiles per group.");
#pragma unroll
          for (int stream_col = 0; stream_col < ColTiles; stream_col += StreamCols) {
            float8_vec score_cache_stream[QGroups][StreamCols];
            i32x2_vec prob_cache_stream[QGroups][StreamCols];
            float local_max_stream[QGroups];
#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              local_max_stream[qg] = -FLT_MAX * 0.5f;
            }
            float prepared_k_scale_tile = k_scale_tile;
            if constexpr (!QuantizeKeyValue && BC <= 64) {
              if constexpr (!PerThreadQK) {
                const int k_scale_idx = k_scale_col_per_warp(kb_base);
                prepared_k_scale_tile =
                    k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
              }
            }
#pragma unroll
            for (int sc = 0; sc < StreamCols; ++sc) {
              const int col_tile = stream_col + sc;
              if constexpr (UseActiveCausalColSkip) {
                if (col_tile >= active_cols_any) {
                  continue;
                }
              }
              const int64_t k_col_start = kb_base + col_tile * BK;
              const bool skip_all =
                  ApplyCausalMask && k_col_start >= q_start[QGroups - 1] + RM;
              if (skip_all) {
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  float8_vec scores;
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    scores[elem] = -FLT_MAX * 0.5f;
                  }
                  score_cache_stream[qg][sc] = scores;
                }
                continue;
              }
              bool fully_future[QGroups];
#pragma unroll
              for (int qg = 0; qg < QGroups; ++qg) {
                if constexpr (UseActiveCausalColSkip) {
                  const int active_cols = qg == 0 ? active_cols0 : active_cols1;
                  fully_future[qg] = col_tile >= active_cols;
                } else {
                  fully_future[qg] =
                      ApplyCausalMask && k_col_start >= q_start[qg] + RM;
                }
              }
              float k_scale_local = k_scale_tile;
                if constexpr (!QuantizeKeyValue && BC <= 64) {
                if constexpr (!PerThreadQK) {
                  k_scale_local = prepared_k_scale_tile;
                }
              } else if constexpr (!QuantizeKeyValue) {
                if constexpr (!PerThreadQK) {
                  const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                  k_scale_local = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
                }
              }
              if constexpr (QuantizeQuery || UsePrepackedLaneMajorK) {
              i32x8_vec score_acc[QGroups];
#pragma unroll
              for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  score_acc[qg][elem] = 0;
                }
              }
#pragma unroll
              for (int dt = 0; dt < DTiles; ++dt) {
                i32x2_vec k_regs;
                if constexpr (UsePrepackedLaneMajorK) {
                  k_regs = pack_k_i8_wmma_b_regs_from_lane_major_global<DTiles>(
                      k, k_head_base, k_stride_n * 64, kb_base, col_tile, dt, lane);
                } else {
                  k_regs = pack_k_i8_wmma_b_regs_from_shared<SharedHeadStride>(
                      &k_tile[0][0], lane, col_tile, dt * BK);
                }
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  if (!fully_future[qg]) {
                    score_acc[qg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, k_regs, true, q_regs[qg][dt], score_acc[qg], true);
                  }
                }
              }
#pragma unroll
              for (int qg = 0; qg < QGroups; ++qg) {
                const float score_scale = qs[qg] * k_scale_local;
                float8_vec scores;
                if (fully_future[qg]) {
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    scores[elem] = -FLT_MAX * 0.5f;
                  }
                } else {
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    scores[elem] = static_cast<float>(score_acc[qg][elem]) * score_scale;
                  }
                  apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                      col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                      ks_stride_b, ks_stride_h, sm_scale);
                  if constexpr (ApplyCausalMask) {
                    const bool needs_causal_mask = k_col_start + BK > q_start[qg];
                    if (needs_causal_mask) {
                      apply_tqk_causal_mask<true>(
                          scores, static_cast<int>(q_start[qg]),
                          static_cast<int>(kb_base), col_tile, lane);
                    }
                  }
                }
                if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
                  apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
                }
                score_cache_stream[qg][sc] = scores;
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  local_max_stream[qg] = fmaxf(local_max_stream[qg], scores[elem]);
                }
              }
              } else {
                FragScoreT score_acc[QGroups];
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  rocwmma::fill_fragment(score_acc[qg], 0);
                }
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                  FragK k_frag;
                  rocwmma::load_matrix_sync(
                      k_frag,
                      &k_tile[0][0] + (col_tile * BK) * SharedHeadStride + dt * BK,
                      static_cast<uint32_t>(SharedHeadStride));
#pragma unroll
                  for (int qg = 0; qg < QGroups; ++qg) {
                    if (!fully_future[qg]) {
                      rocwmma::mma_sync(score_acc[qg], k_frag, q_frag[qg][dt], score_acc[qg]);
                    }
                  }
                }
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  const float score_scale = qs[qg] * k_scale_local;
                  float8_vec scores;
                  if (fully_future[qg]) {
#pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                      scores[elem] = -FLT_MAX * 0.5f;
                    }
                  } else {
                    const auto score_rm =
                        rocwmma::apply_data_layout<rocwmma::row_major>(score_acc[qg]);
#pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                      scores[elem] = static_cast<float>(score_rm[elem]) * score_scale;
                    }
                    apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                        col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                        ks_stride_b, ks_stride_h, sm_scale);
                    if constexpr (ApplyCausalMask) {
                      const bool needs_causal_mask = k_col_start + BK > q_start[qg];
                      if (needs_causal_mask) {
                        apply_tqk_causal_mask<true>(
                            scores, static_cast<int>(q_start[qg]),
                            static_cast<int>(kb_base), col_tile, lane);
                      }
                    }
                  }
                  if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
                    apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
                  }
                  score_cache_stream[qg][sc] = scores;
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    local_max_stream[qg] = fmaxf(local_max_stream[qg], scores[elem]);
                  }
                }
              }
            }

#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              const float tile_max =
                  fmaxf(local_max_stream[qg], __shfl_xor(local_max_stream[qg], 16, 32));
              const float old_m = m[qg];
              const float new_m = fmaxf(old_m, tile_max);
              const bool has_previous_sum = l[qg] != 0.0f;
              const float alpha = has_previous_sum ? fast_exp2(old_m - new_m) : 0.0f;
              m[qg] = new_m;
              l[qg] *= alpha;

              if (has_previous_sum) {
                float alpha_rows[8];
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
                }
#pragma unroll
                for (int vdt = 0; vdt < ValueTiles; ++vdt) {
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    out_frag[qg][vdt][elem] *= alpha_rows[elem];
                  }
                }
              }

              float local_sum = 0.0f;
#pragma unroll
              for (int sc = 0; sc < StreamCols; ++sc) {
                const int col_tile = stream_col + sc;
                if constexpr (UseActiveCausalColSkip) {
                  const int active_cols = qg == 0 ? active_cols0 : active_cols1;
                  if (col_tile >= active_cols) {
                    continue;
                  }
                }
                const int64_t k_col_start = kb_base + col_tile * BK;
                bool fully_future =
                    ApplyCausalMask && k_col_start >= q_start[qg] + RM;
                if constexpr (UseActiveCausalColSkip) {
                  fully_future = false;
                }
                const float8_vec scores = score_cache_stream[qg][sc];
                float8_vec prob_values;
#pragma unroll
                for (int elem = 0; elem < 8; ++elem) {
                  float prob = 0.0f;
                  if (!fully_future) {
                    prob = fast_exp2(scores[elem] - m[qg] + kFp8SoftmaxOffset);
                    local_sum += prob;
                  }
                  prob_values[elem] = prob;
                }
                prob_cache_stream[qg][sc] = make_p_fp8_regs_from_tqk_prob_regs(
                    prob_values, lane);
              }
              l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
            }

#pragma unroll
            for (int sc = 0; sc < StreamCols; ++sc) {
              const int col_tile = stream_col + sc;
              if constexpr (UseActiveCausalColSkip) {
                if (col_tile >= active_cols_any) {
                  continue;
                }
              }
              const int64_t k_col_start = kb_base + col_tile * BK;
              bool fully_future[QGroups];
              bool skip_all = true;
#pragma unroll
              for (int qg = 0; qg < QGroups; ++qg) {
                if constexpr (UseActiveCausalColSkip) {
                  const int active_cols = qg == 0 ? active_cols0 : active_cols1;
                  fully_future[qg] = col_tile >= active_cols;
                } else {
                  fully_future[qg] =
                      ApplyCausalMask && k_col_start >= q_start[qg] + RM;
                }
                skip_all = skip_all && fully_future[qg];
              }
              if (skip_all) {
                continue;
              }
#pragma unroll
              for (int vdt = 0; vdt < ValueTiles; ++vdt) {
                const int dt = ValueTileBase + vdt;
                i32x2_vec v_regs;
                if constexpr (UsePrepackedLaneMajorValue) {
                  v_regs = make_v_fp8_regs_from_lane_major_global<DTiles>(
                      v, v_head_base, kb_base, col_tile, dt, lane);
                } else if constexpr (PackedTransposedValue) {
                  v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStrideT>(
                      &v_tile_t[0][0], col_tile, dt, lane);
                } else {
                  v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStride>(
                      &v_tile[0][0], col_tile, dt, lane);
                }
#pragma unroll
                for (int qg = 0; qg < QGroups; ++qg) {
                  if (fully_future[qg]) {
                    continue;
                  }
                  float8_vec acc;
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    acc[elem] = out_frag[qg][vdt][elem];
                  }
                  acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                      prob_cache_stream[qg][sc], v_regs, acc);
#pragma unroll
                  for (int elem = 0; elem < 8; ++elem) {
                    out_frag[qg][vdt][elem] = acc[elem];
                  }
                }
              }
            }
          }
        } else {
        float8_vec score_cache[QGroups][ColTiles];
        float local_max[QGroups];
        float prepared_k_scale_tile = k_scale_tile;
        if constexpr (!QuantizeKeyValue && BC <= 64) {
          if constexpr (!PerThreadQK) {
            const int k_scale_idx = k_scale_col_per_warp(kb_base);
            prepared_k_scale_tile =
                k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
          }
        }
#pragma unroll
        for (int qg = 0; qg < QGroups; ++qg) {
          local_max[qg] = -FLT_MAX * 0.5f;
        }
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          if constexpr (UseActiveCausalColSkip) {
            if (col_tile >= active_cols_any) {
              continue;
            }
          }
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool skip_all =
              ApplyCausalMask && k_col_start >= q_start[QGroups - 1] + RM;
          if (skip_all) {
#pragma unroll
            for (int qg = 0; qg < QGroups; ++qg) {
              float8_vec scores;
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                scores[elem] = -FLT_MAX * 0.5f;
              }
              score_cache[qg][col_tile] = scores;
            }
            continue;
          }
          bool q0_fully_future =
              ApplyCausalMask && k_col_start >= q_start[0] + RM;
          bool q1_fully_future =
              ApplyCausalMask && k_col_start >= q_start[1] + RM;
          if constexpr (UseActiveCausalColSkip) {
            q0_fully_future = col_tile >= active_cols0;
            q1_fully_future = col_tile >= active_cols1;
          }
          float k_scale_local = k_scale_tile;
          if constexpr (!QuantizeKeyValue && BC <= 64) {
            if constexpr (!PerThreadQK) {
              k_scale_local = prepared_k_scale_tile;
            }
          } else if constexpr (!QuantizeKeyValue) {
            if constexpr (!PerThreadQK) {
              const int k_scale_idx = k_scale_col_per_warp(k_col_start);
              k_scale_local = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
            }
          }
          i32x8_vec score_acc[QGroups];
#pragma unroll
          for (int qg = 0; qg < QGroups; ++qg) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              score_acc[qg][elem] = 0;
            }
          }
#pragma unroll
          for (int dt = 0; dt < DTiles; ++dt) {
            i32x2_vec k_regs;
            if constexpr (UsePrepackedLaneMajorK) {
              k_regs = pack_k_i8_wmma_b_regs_from_lane_major_global<DTiles>(
                  k, k_head_base, k_stride_n * 64, kb_base, col_tile, dt, lane);
            } else {
              k_regs = pack_k_i8_wmma_b_regs_from_shared<SharedHeadStride>(
                  &k_tile[0][0], lane, col_tile, dt * BK);
            }
            if (!q0_fully_future) {
              score_acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                  true, k_regs, true, q_regs[0][dt], score_acc[0], true);
            }
            if (!q1_fully_future) {
              score_acc[1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                  true, k_regs, true, q_regs[1][dt], score_acc[1], true);
            }
          }
#pragma unroll
          for (int qg = 0; qg < QGroups; ++qg) {
            const float score_scale = qs[qg] * k_scale_local;
            float8_vec scores;
            const bool fully_future =
                qg == 0 ? q0_fully_future : q1_fully_future;
            if constexpr (UseActiveCausalColSkip) {
              if (fully_future) {
                continue;
              }
            }
            if (fully_future) {
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                scores[elem] = -FLT_MAX * 0.5f;
              }
            } else {
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                scores[elem] = static_cast<float>(score_acc[qg][elem]) * score_scale;
              }
              apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                  col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                  ks_stride_b, ks_stride_h, sm_scale);
              if constexpr (ApplyCausalMask) {
                const bool needs_causal_mask = k_col_start + BK > q_start[qg];
                if (needs_causal_mask) {
                  apply_tqk_causal_mask<true>(
                      scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base), col_tile, lane);
                }
              }
            }
            if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
              apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
            }
            score_cache[qg][col_tile] = scores;
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              local_max[qg] = fmaxf(local_max[qg], scores[elem]);
            }
          }
        }

#pragma unroll
        for (int qg = 0; qg < QGroups; ++qg) {
          const float tile_max = fmaxf(local_max[qg], __shfl_xor(local_max[qg], 16, 32));
          const float old_m = m[qg];
          const float new_m = fmaxf(old_m, tile_max);
          const bool has_previous_sum = l[qg] != 0.0f;
          const float alpha = has_previous_sum ? fast_exp2(old_m - new_m) : 0.0f;
          m[qg] = new_m;
          l[qg] *= alpha;

          if (has_previous_sum) {
            float alpha_rows[8];
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
            }

#pragma unroll
            for (int vdt = 0; vdt < ValueTiles; ++vdt) {
#pragma unroll
              for (int elem = 0; elem < 8; ++elem) {
                out_frag[qg][vdt][elem] *= alpha_rows[elem];
              }
            }
          }

          float local_sum = 0.0f;
#pragma unroll
          for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
            if constexpr (UseActiveCausalColSkip) {
              const int active_cols = qg == 0 ? active_cols0 : active_cols1;
              if (col_tile >= active_cols) {
                continue;
              }
            }
            const int64_t k_col_start = kb_base + col_tile * BK;
            bool fully_future =
                ApplyCausalMask && k_col_start >= q_start[qg] + RM;
            if constexpr (UseActiveCausalColSkip) {
              fully_future = false;
            }
            const float8_vec scores = score_cache[qg][col_tile];
            float8_vec prob_values;
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              float prob = 0.0f;
              if (!fully_future) {
                prob = fast_exp2(scores[elem] - m[qg] + kFp8SoftmaxOffset);
                local_sum += prob;
              }
              prob_values[elem] = prob;
            }
            prob_cache[qg][col_tile] = make_p_fp8_regs_from_tqk_prob_regs(prob_values, lane);
          }
          l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
        }
        }
      } else {
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
          float8_vec score_cache[ColTiles];
          float local_max = -FLT_MAX * 0.5f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future =
              ApplyCausalMask && k_col_start >= q_start[qg] + RM;
          float8_vec scores;
          if (fully_future) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              scores[elem] = -FLT_MAX * 0.5f;
            }
          } else {
            float k_scale_local = k_scale_tile;
            if constexpr (!QuantizeKeyValue) {
              if constexpr (!PerThreadQK) {
                const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                k_scale_local = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
              }
            }
            const float score_scale = qs[qg] * k_scale_local;
            if constexpr (QuantizeQuery) {
              scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
                  &k_tile[0][0], q_regs[qg], lane, col_tile, score_scale);
            } else {
              scores =
                  compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                      &k_tile[0][0], q_frag[qg], col_tile, score_scale);
            }
            apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            if constexpr (ApplyCausalMask) {
              const bool needs_causal_mask = k_col_start + BK > q_start[qg];
              if (needs_causal_mask) {
                apply_tqk_causal_mask<true>(
                    scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base), col_tile, lane);
              }
            }
          }
          if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
            apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
          }
          score_cache[col_tile] = scores;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            local_max = fmaxf(local_max, scores[elem]);
          }
        }
        const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        const float old_m = m[qg];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
        m[qg] = new_m;
        l[qg] *= alpha;

        float alpha_rows[8];
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
        }

#pragma unroll
        for (int vdt = 0; vdt < ValueTiles; ++vdt) {
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[qg][vdt][elem] *= alpha_rows[elem];
          }
        }

        float local_sum = 0.0f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future =
              ApplyCausalMask && k_col_start >= q_start[qg] + RM;
          const float8_vec scores = score_cache[col_tile];
          float8_vec prob_values;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            float prob = 0.0f;
            if (!fully_future) {
              prob = fast_exp2(scores[elem] - m[qg] + kFp8SoftmaxOffset);
              local_sum += prob;
            }
            prob_values[elem] = prob;
          }
          prob_cache[qg][col_tile] = make_p_fp8_regs_from_tqk_prob_regs(prob_values, lane);
        }
        l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
      }
      }

      if constexpr (!UseStreamedFp8Pv) {
#pragma unroll
      for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
        if constexpr (UseActiveCausalColSkip) {
          if (col_tile >= active_cols_any) {
            continue;
          }
        }
        const int64_t k_col_start = kb_base + col_tile * BK;
        bool q0_fully_future =
            ApplyCausalMask && k_col_start >= q_start[0] + RM;
        bool q1_fully_future =
            ApplyCausalMask && k_col_start >= q_start[1] + RM;
        if constexpr (UseActiveCausalColSkip) {
          q0_fully_future = col_tile >= active_cols0;
          q1_fully_future = col_tile >= active_cols1;
        }
        if (q0_fully_future && q1_fully_future) {
          continue;
        }
#pragma unroll
        for (int vdt = 0; vdt < ValueTiles; ++vdt) {
          const int dt = ValueTileBase + vdt;
          i32x2_vec v_regs;
          if constexpr (UsePrepackedLaneMajorValue) {
            v_regs = make_v_fp8_regs_from_lane_major_global<DTiles>(
                v, v_head_base, kb_base, col_tile, dt, lane);
          } else if constexpr (ValueTransposed) {
            if constexpr (PackedTransposedValue) {
              v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStrideT>(
                  &v_tile_t[0][0], col_tile, dt, lane);
            } else {
              v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStride>(
                  &v_tile[0][0], col_tile, dt, lane);
            }
          } else {
            v_regs = make_v_fp8_regs_from_shared<SharedHeadStride>(
                &v_tile[0][0], col_tile, dt, lane);
          }
          float8_vec acc0;
          float8_vec acc1;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            acc0[elem] = out_frag[0][vdt][elem];
            acc1[elem] = out_frag[1][vdt][elem];
          }
          if (!q0_fully_future) {
            const i32x2_vec p_regs0 = prob_cache[0][col_tile];
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                p_regs0, v_regs, acc0);
          }
          if (!q1_fully_future) {
            const i32x2_vec p_regs1 = prob_cache[1][col_tile];
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                p_regs1, v_regs, acc1);
          }
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[0][vdt][elem] = acc0[elem];
            out_frag[1][vdt][elem] = acc1[elem];
          }
        }
      }
      }
    } else {
#pragma unroll
      for (int qg = 0; qg < QGroups; ++qg) {
        float8_vec score_cache[ColTiles];
        float local_max = -FLT_MAX * 0.5f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future =
              ApplyCausalMask && k_col_start >= q_start[qg] + RM;
          float8_vec scores;
          if (fully_future) {
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              scores[elem] = -FLT_MAX * 0.5f;
            }
          } else {
            float k_scale_local = k_scale_tile;
            if constexpr (!QuantizeKeyValue) {
              if constexpr (!PerThreadQK) {
                const int k_scale_idx = k_scale_col_per_warp(k_col_start);
                k_scale_local = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
              }
            }
            const float score_scale = qs[qg] * k_scale_local;
            if constexpr (QuantizeQuery) {
              scores = compute_tqk_score_regs_raw_kq<DTiles, SharedHeadStride>(
                  &k_tile[0][0], q_regs[qg], lane, col_tile, score_scale);
            } else {
              scores =
                  compute_tqk_score_regs<DTiles, SharedHeadStride, FragK, FragQ, FragScoreT>(
                      &k_tile[0][0], q_frag[qg], col_tile, score_scale);
            }
            apply_per_thread_qk_score_scale<PerThreadQK, false>(scores, q_scale, k_scale, b, hq, hkv, q_start[qg], kb_base,
                col_tile, lane, qo_len, kv_len, qs_stride_b, qs_stride_h,
                ks_stride_b, ks_stride_h, sm_scale);
            if constexpr (ApplyCausalMask) {
              const bool needs_causal_mask = k_col_start + BK > q_start[qg];
              if (needs_causal_mask) {
                apply_tqk_causal_mask<true>(
                    scores, static_cast<int>(q_start[qg]), static_cast<int>(kb_base), col_tile, lane);
              }
            }
          }
          if constexpr (!NoKvTail) if (k_col_start + BK > kv_len) {
            apply_tqk_kv_tail_mask<false>(scores, kv_len, kb_base, col_tile, lane);
          }
          score_cache[col_tile] = scores;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            local_max = fmaxf(local_max, scores[elem]);
          }
        }
        const float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        const float old_m = m[qg];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = l[qg] == 0.0f ? 0.0f : fast_exp2(old_m - new_m);
        m[qg] = new_m;
        l[qg] *= alpha;

        float alpha_rows[8];
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          alpha_rows[elem] = __shfl(alpha, row_base + elem, 32);
        }

#pragma unroll
        for (int vdt = 0; vdt < ValueTiles; ++vdt) {
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            out_frag[qg][vdt][elem] *= alpha_rows[elem];
          }
        }

        float local_sum = 0.0f;
#pragma unroll
        for (int col_tile = 0; col_tile < ColTiles; ++col_tile) {
          const int64_t k_col_start = kb_base + col_tile * BK;
          const bool fully_future =
              ApplyCausalMask && k_col_start >= q_start[qg] + RM;
          if (fully_future) {
            continue;
          }
          const float8_vec scores = score_cache[col_tile];
          float8_vec prob_values;
#pragma unroll
          for (int elem = 0; elem < 8; ++elem) {
            const float prob = fast_exp2(scores[elem] - m[qg] + kFp8SoftmaxOffset);
            local_sum += prob;
            prob_values[elem] = prob;
          }

          const i32x2_vec p_regs = make_p_fp8_regs_from_tqk_prob_regs(prob_values, lane);
#pragma unroll
          for (int vdt = 0; vdt < ValueTiles; ++vdt) {
            const int dt = ValueTileBase + vdt;
            i32x2_vec v_regs;
            if constexpr (UsePrepackedLaneMajorValue) {
              v_regs = make_v_fp8_regs_from_lane_major_global<DTiles>(
                  v, v_head_base, kb_base, col_tile, dt, lane);
            } else if constexpr (ValueTransposed) {
              if constexpr (PackedTransposedValue) {
                v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStrideT>(
                    &v_tile_t[0][0], col_tile, dt, lane);
              } else {
                v_regs = make_v_fp8_regs_from_transposed_shared<SharedValueStride>(
                    &v_tile[0][0], col_tile, dt, lane);
              }
            } else {
              v_regs = make_v_fp8_regs_from_shared<SharedHeadStride>(
                  &v_tile[0][0], col_tile, dt, lane);
            }
            float8_vec acc;
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              acc[elem] = out_frag[qg][vdt][elem];
            }
            const float8_vec pv_acc =
                __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(p_regs, v_regs, acc);
#pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
              out_frag[qg][vdt][elem] = pv_acc[elem];
            }
          }
        }
        l[qg] += local_sum + __shfl_xor(local_sum, 16, 32);
      }
    }
    if constexpr (UsesTileSharedMemory) {
      __syncthreads();
    }
  };

  if constexpr (IsCausal) {
    const int64_t diag_base = (q_base / BC) * BC;
    const int64_t prefix_limit = diag_base < kv_limit ? diag_base : kv_limit;
#pragma unroll 2
    for (int64_t kb_base = 0; kb_base < prefix_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::false_type{});
    }
#pragma unroll 2
    for (int64_t kb_base = prefix_limit; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::true_type{});
    }
  } else {
#pragma unroll 2
    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BC) {
      process_kv_tile(kb_base, std::false_type{});
    }
  }

  float value_scale_tile[ValueTiles];
#pragma unroll
  for (int vdt = 0; vdt < ValueTiles; ++vdt) {
    const int d = (ValueTileBase + vdt) * BK + col;
    value_scale_tile[vdt] = v_scale == nullptr ?
        1.0f : v_scale[(b * num_kv_heads + hkv) * HeadDim + d];
  }

#pragma unroll
  for (int qg = 0; qg < QGroups; ++qg) {
    float l_rows[8];
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const float l_sum = __shfl(l[qg], row_base + elem, 32);
      if constexpr (InvLRowsEpilogue) {
        l_rows[elem] = l_sum == 0.0f ? 0.0f : 1.0f / l_sum;
      } else {
        l_rows[elem] = l_sum;
      }
    }
#pragma unroll
    for (int vdt = 0; vdt < ValueTiles; ++vdt) {
      const int d = (ValueTileBase + vdt) * BK + col;
      const float value_scale = value_scale_tile[vdt];
#pragma unroll
      for (int pair = 0; pair < PackedRows; ++pair) {
        const int elem = pair * 2;
        const int64_t q_idx0 = q_start[qg] + row_base + elem;
        const int64_t q_idx1 = q_idx0 + 1;
        float value0;
        float value1;
        if constexpr (InvLRowsEpilogue) {
          value0 = out_frag[qg][vdt][elem] * l_rows[elem] * value_scale;
          value1 = out_frag[qg][vdt][elem + 1] * l_rows[elem + 1] * value_scale;
        } else {
          const float l_sum0 = l_rows[elem];
          const float l_sum1 = l_rows[elem + 1];
          value0 = l_sum0 == 0.0f ?
              0.0f : (out_frag[qg][vdt][elem] / l_sum0) * value_scale;
          value1 = l_sum1 == 0.0f ?
              0.0f : (out_frag[qg][vdt][elem + 1] / l_sum1) * value_scale;
        }
        store_output_value(output, qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx0, d, o_stride_b, o_stride_n, o_stride_h), value0);
        store_output_value(output, qkv_offset_dispatch<HeadDim, HndContiguous, StaticNhdLayout>(
            tensor_layout, b, hq, q_idx1, d, o_stride_b, o_stride_n, o_stride_h), value1);
      }
    }
  }
}

template <typename T,
          typename OutT,
          bool ToFp8,
          int HeadDim,
          int Threads = 256,
          bool TransposeValue = true,
          bool PrepareQuery = true,
          bool PrepackF16VLane = false,
          bool PrepackF16KLane = false,
          bool PrepackFp8Lane = false,
          bool FullGroupsNoTail = false,
          int StaticQLen = 0,
          int StaticKvLen = 0,
          bool SubtractKeyMean = false>
__global__ void prepare_qkv_hnd_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ key_mean,
    int8_t* __restrict__ query_out,
    int8_t* __restrict__ key_out,
    float* __restrict__ query_scale,
    float* __restrict__ key_scale,
    OutT* __restrict__ value_out,
    const int64_t batch,
    const int64_t q_heads,
    const int64_t kv_heads,
    const int64_t q_len,
    const int64_t kv_len,
    const int q_groups,
    const int k_groups,
    const bool fuse_self_qkv) {
  constexpr int PackElems = 8;
  constexpr int KRows = 64;
  constexpr int QRows = 32;
  constexpr int ValueStride = HeadDim + 16;
  constexpr int LaneColTiles = KRows / 16;
  constexpr int LaneDTiles = HeadDim / 16;
  static_assert((HeadDim % PackElems) == 0, "native preparation packs eight elements");
  static_assert(!PrepackF16VLane || (!ToFp8 && TransposeValue && HeadDim == 64),
                "fp16 lane-major V prepack is specialized for transposed D64 fp16 values");
  static_assert(!PrepackF16KLane || (!ToFp8 && HeadDim == 64),
                "fp16 lane-major K prepack is specialized for D64 fp16/bf16 keys");
  static_assert(!PrepackFp8Lane || (ToFp8 && TransposeValue && HeadDim == 64),
                "fp8 lane-major K/V prepack is specialized for transposed D64 fp8 values");
  static_assert(StaticQLen == 0 || (StaticQLen % (2 * QRows)) == 0,
                "static QKV preparation Q length must cover full two-group Q tasks.");
  static_assert(StaticKvLen == 0 || (StaticKvLen % KRows) == 0,
                "static QKV preparation KV length must cover full K groups.");
  static_assert(!SubtractKeyMean || !PrepackF16KLane,
                "smooth-K preparation does not use lane-major K prepack.");

  __shared__ float shared_amax[2];
  __shared__ float shared_pair_amax[2][32];
  __shared__ OutT value_tile[PrepackFp8Lane ? 1 : KRows][PrepackFp8Lane ? 1 : ValueStride];
  __shared__ int8_t key_tile[PrepackF16KLane ? KRows : 1][PrepackF16KLane ? ValueStride : 1];

  const int task = blockIdx.x;
  const int head = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  constexpr bool StaticFullQ = FullGroupsNoTail && StaticQLen != 0;
  constexpr bool StaticFullKV = FullGroupsNoTail && StaticKvLen != 0;
  constexpr int StaticQGroups = StaticQLen == 0 ? 0 : (StaticQLen / QRows);
  constexpr int StaticQTaskGroups = StaticQLen == 0 ? 0 : ((StaticQGroups + 1) / 2);
  constexpr int StaticKGroups = StaticKvLen == 0 ? 0 : (StaticKvLen / KRows);
  const int effective_q_groups = StaticQLen == 0 ? q_groups : StaticQGroups;
  const int effective_q_task_groups =
      StaticQLen == 0 ? ((q_groups + 1) / 2) : StaticQTaskGroups;
  const int effective_k_groups = StaticKvLen == 0 ? k_groups : StaticKGroups;
  const int64_t effective_q_len = StaticQLen == 0 ? q_len : StaticQLen;
  const int64_t effective_kv_len = StaticKvLen == 0 ? kv_len : StaticKvLen;

  if constexpr (PrepareQuery) {
  if (task < effective_q_task_groups) {
    const int local_group_base = task * 2;
    const int64_t base_row = static_cast<int64_t>(local_group_base) * QRows;
    if (b >= batch || head >= q_heads ||
        (!StaticFullQ && base_row >= effective_q_len)) {
      return;
    }

    constexpr int packs = (QRows * HeadDim) / PackElems;
    const bool has_q_group1 = StaticFullQ || ((local_group_base + 1) < effective_q_groups);
    float local_amax0 = 0.0000001f;
    float local_amax1 = 0.0000001f;
    for (int pack = tid; pack < packs; pack += Threads) {
      const int elem_base = pack * PackElems;
      const int row = elem_base / HeadDim;
      const int d = elem_base - row * HeadDim;
      const int64_t seq0 = base_row + row;
      if (StaticFullQ || seq0 < effective_q_len) {
        const int64_t off =
            ((static_cast<int64_t>(b) * q_heads + head) * effective_q_len + seq0) *
                HeadDim + d;
        const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
        const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < PackElems; ++i) {
          local_amax0 = fmaxf(local_amax0, fabsf(value_to_float(values[i])));
        }
      }
      if (has_q_group1) {
        const int64_t seq1 = base_row + QRows + row;
        if (StaticFullQ || seq1 < effective_q_len) {
          const int64_t off =
              ((static_cast<int64_t>(b) * q_heads + head) * effective_q_len + seq1) *
                  HeadDim + d;
          const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
          const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
          for (int i = 0; i < PackElems; ++i) {
            local_amax1 = fmaxf(local_amax1, fabsf(value_to_float(values[i])));
          }
        }
      }
    }
    const int lane = tid & 31;
    const int wid = tid >> 5;
    local_amax0 = vllm::warpReduceMax(local_amax0);
    local_amax1 = vllm::warpReduceMax(local_amax1);
    if (lane == 0) {
      shared_pair_amax[0][wid] = local_amax0;
      shared_pair_amax[1][wid] = local_amax1;
    }
    __syncthreads();
    const bool warp_lane_active = tid < (blockDim.x / 32);
    local_amax0 = warp_lane_active ? shared_pair_amax[0][lane] : -1e20f;
    local_amax1 = warp_lane_active ? shared_pair_amax[1][lane] : -1e20f;
    local_amax0 = vllm::warpReduceMax(local_amax0);
    local_amax1 = vllm::warpReduceMax(local_amax1);
    if (tid == 0) {
      shared_amax[0] = local_amax0;
      query_scale[(static_cast<int64_t>(b) * q_heads + head) * effective_q_groups +
                  local_group_base] = local_amax0 / 127.0f;
      if (has_q_group1) {
        shared_amax[1] = local_amax1;
        query_scale[(static_cast<int64_t>(b) * q_heads + head) * effective_q_groups +
                    local_group_base + 1] = local_amax1 / 127.0f;
      }
    }
    __syncthreads();
    const float inv_scale0 = 127.0f / shared_amax[0];
    const float inv_scale1 = has_q_group1 ? (127.0f / shared_amax[1]) : 0.0f;

    for (int pack = tid; pack < packs; pack += Threads) {
      const int elem_base = pack * PackElems;
      const int row = elem_base / HeadDim;
      const int d = elem_base - row * HeadDim;
      const int64_t seq0 = base_row + row;
      if (StaticFullQ || seq0 < effective_q_len) {
        const int64_t off =
            ((static_cast<int64_t>(b) * q_heads + head) * effective_q_len + seq0) *
                HeadDim + d;
        const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
        const T* values = reinterpret_cast<const T*>(&raw);
        char4 out0;
        char4 out1;
        if constexpr (SubtractKeyMean) {
          out0.x = float_to_int8_nearby_gfx12(value_to_float(values[0]) * inv_scale0);
          out0.y = float_to_int8_nearby_gfx12(value_to_float(values[1]) * inv_scale0);
          out0.z = float_to_int8_nearby_gfx12(value_to_float(values[2]) * inv_scale0);
          out0.w = float_to_int8_nearby_gfx12(value_to_float(values[3]) * inv_scale0);
          out1.x = float_to_int8_nearby_gfx12(value_to_float(values[4]) * inv_scale0);
          out1.y = float_to_int8_nearby_gfx12(value_to_float(values[5]) * inv_scale0);
          out1.z = float_to_int8_nearby_gfx12(value_to_float(values[6]) * inv_scale0);
          out1.w = float_to_int8_nearby_gfx12(value_to_float(values[7]) * inv_scale0);
        } else {
          out0.x = float_to_int8_rn_gfx12(value_to_float(values[0]) * inv_scale0);
          out0.y = float_to_int8_rn_gfx12(value_to_float(values[1]) * inv_scale0);
          out0.z = float_to_int8_rn_gfx12(value_to_float(values[2]) * inv_scale0);
          out0.w = float_to_int8_rn_gfx12(value_to_float(values[3]) * inv_scale0);
          out1.x = float_to_int8_rn_gfx12(value_to_float(values[4]) * inv_scale0);
          out1.y = float_to_int8_rn_gfx12(value_to_float(values[5]) * inv_scale0);
          out1.z = float_to_int8_rn_gfx12(value_to_float(values[6]) * inv_scale0);
          out1.w = float_to_int8_rn_gfx12(value_to_float(values[7]) * inv_scale0);
        }
        *reinterpret_cast<char4*>(query_out + off) = out0;
        *reinterpret_cast<char4*>(query_out + off + 4) = out1;
      }
      if (has_q_group1) {
        const int64_t seq1 = base_row + QRows + row;
        if (StaticFullQ || seq1 < effective_q_len) {
          const int64_t off =
              ((static_cast<int64_t>(b) * q_heads + head) * effective_q_len + seq1) *
                  HeadDim + d;
          const uint4 raw = *reinterpret_cast<const uint4*>(query + off);
          const T* values = reinterpret_cast<const T*>(&raw);
          char4 out0;
          char4 out1;
          if constexpr (SubtractKeyMean) {
            out0.x = float_to_int8_nearby_gfx12(value_to_float(values[0]) * inv_scale1);
            out0.y = float_to_int8_nearby_gfx12(value_to_float(values[1]) * inv_scale1);
            out0.z = float_to_int8_nearby_gfx12(value_to_float(values[2]) * inv_scale1);
            out0.w = float_to_int8_nearby_gfx12(value_to_float(values[3]) * inv_scale1);
            out1.x = float_to_int8_nearby_gfx12(value_to_float(values[4]) * inv_scale1);
            out1.y = float_to_int8_nearby_gfx12(value_to_float(values[5]) * inv_scale1);
            out1.z = float_to_int8_nearby_gfx12(value_to_float(values[6]) * inv_scale1);
            out1.w = float_to_int8_nearby_gfx12(value_to_float(values[7]) * inv_scale1);
          } else {
            out0.x = float_to_int8_rn_gfx12(value_to_float(values[0]) * inv_scale1);
            out0.y = float_to_int8_rn_gfx12(value_to_float(values[1]) * inv_scale1);
            out0.z = float_to_int8_rn_gfx12(value_to_float(values[2]) * inv_scale1);
            out0.w = float_to_int8_rn_gfx12(value_to_float(values[3]) * inv_scale1);
            out1.x = float_to_int8_rn_gfx12(value_to_float(values[4]) * inv_scale1);
            out1.y = float_to_int8_rn_gfx12(value_to_float(values[5]) * inv_scale1);
            out1.z = float_to_int8_rn_gfx12(value_to_float(values[6]) * inv_scale1);
            out1.w = float_to_int8_rn_gfx12(value_to_float(values[7]) * inv_scale1);
          }
          *reinterpret_cast<char4*>(query_out + off) = out0;
          *reinterpret_cast<char4*>(query_out + off + 4) = out1;
        }
      }
    }
    if (!fuse_self_qkv) {
      return;
    }
  }
  }

  const int local_group =
      PrepareQuery ? (fuse_self_qkv ? task : task - effective_q_task_groups) : task;
  const int64_t base_row = static_cast<int64_t>(local_group) * KRows;
  if (b >= batch || head >= kv_heads || local_group >= effective_k_groups ||
      (!StaticFullKV && base_row >= effective_kv_len)) {
    return;
  }

  constexpr int kv_packs = (KRows * HeadDim) / PackElems;
  float local_amax = 0.0000001f;
  for (int pack = tid; pack < kv_packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (StaticFullKV || seq < effective_kv_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * kv_heads + head) * effective_kv_len + seq) *
              HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(key + off);
      const T* values = reinterpret_cast<const T*>(&raw);
      const T* mean_values = nullptr;
      if constexpr (SubtractKeyMean) {
        mean_values = key_mean + (static_cast<int64_t>(b) * kv_heads + head) * HeadDim + d;
      }
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        float value = value_to_float(values[i]);
        if constexpr (SubtractKeyMean) {
          value -= value_to_float(mean_values[i]);
        }
        local_amax = fmaxf(local_amax, fabsf(value));
      }
    }
  }
  const float block_amax = vllm::blockReduceMax(local_amax);
  if (tid == 0) {
    shared_amax[0] = block_amax;
    key_scale[(static_cast<int64_t>(b) * kv_heads + head) * effective_k_groups +
              local_group] = shared_amax[0] / 127.0f;
  }
  __syncthreads();
  const float inv_scale = 127.0f / shared_amax[0];

  for (int pack = tid; pack < kv_packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (StaticFullKV || seq < effective_kv_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * kv_heads + head) * effective_kv_len + seq) *
              HeadDim + d;
      const uint4 raw_k = *reinterpret_cast<const uint4*>(key + off);
      const uint4 raw_v = *reinterpret_cast<const uint4*>(value + off);
      const T* k_values = reinterpret_cast<const T*>(&raw_k);
      const T* v_values = reinterpret_cast<const T*>(&raw_v);
      const T* mean_values = nullptr;
      if constexpr (SubtractKeyMean) {
        mean_values = key_mean + (static_cast<int64_t>(b) * kv_heads + head) * HeadDim + d;
      }
      char4 out0;
      char4 out1;
      float k0 = value_to_float(k_values[0]);
      float k1 = value_to_float(k_values[1]);
      float k2 = value_to_float(k_values[2]);
      float k3 = value_to_float(k_values[3]);
      float k4 = value_to_float(k_values[4]);
      float k5 = value_to_float(k_values[5]);
      float k6 = value_to_float(k_values[6]);
      float k7 = value_to_float(k_values[7]);
      if constexpr (SubtractKeyMean) {
        k0 -= value_to_float(mean_values[0]);
        k1 -= value_to_float(mean_values[1]);
        k2 -= value_to_float(mean_values[2]);
        k3 -= value_to_float(mean_values[3]);
        k4 -= value_to_float(mean_values[4]);
        k5 -= value_to_float(mean_values[5]);
        k6 -= value_to_float(mean_values[6]);
        k7 -= value_to_float(mean_values[7]);
      }
      if constexpr (SubtractKeyMean) {
        out0.x = float_to_int8_nearby_gfx12(k0 * inv_scale);
        out0.y = float_to_int8_nearby_gfx12(k1 * inv_scale);
        out0.z = float_to_int8_nearby_gfx12(k2 * inv_scale);
        out0.w = float_to_int8_nearby_gfx12(k3 * inv_scale);
        out1.x = float_to_int8_nearby_gfx12(k4 * inv_scale);
        out1.y = float_to_int8_nearby_gfx12(k5 * inv_scale);
        out1.z = float_to_int8_nearby_gfx12(k6 * inv_scale);
        out1.w = float_to_int8_nearby_gfx12(k7 * inv_scale);
      } else {
        out0.x = float_to_int8_rn_gfx12(k0 * inv_scale);
        out0.y = float_to_int8_rn_gfx12(k1 * inv_scale);
        out0.z = float_to_int8_rn_gfx12(k2 * inv_scale);
        out0.w = float_to_int8_rn_gfx12(k3 * inv_scale);
        out1.x = float_to_int8_rn_gfx12(k4 * inv_scale);
        out1.y = float_to_int8_rn_gfx12(k5 * inv_scale);
        out1.z = float_to_int8_rn_gfx12(k6 * inv_scale);
        out1.w = float_to_int8_rn_gfx12(k7 * inv_scale);
      }
      if constexpr (PrepackFp8Lane) {
        const int row_in_group = row & 63;
        const int col_tile = row_in_group >> 4;
        const int col = row_in_group & 15;
        const int d_tile = d >> 4;
        const int lane_out = col | (((d & 8) != 0) ? 16 : 0);
        const int64_t lane_off =
            ((((static_cast<int64_t>(b) * kv_heads + head) * k_groups + local_group) *
              LaneColTiles + col_tile) * LaneDTiles + d_tile) * 32 * 8 +
            static_cast<int64_t>(lane_out) * 8;
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&out0);
        packed.y = *reinterpret_cast<const uint32_t*>(&out1);
        *reinterpret_cast<uint2*>(key_out + lane_off) = packed;
      } else if constexpr (PrepackF16KLane) {
        *reinterpret_cast<char4*>(&key_tile[row][d]) = out0;
        *reinterpret_cast<char4*>(&key_tile[row][d + 4]) = out1;
      } else {
        *reinterpret_cast<char4*>(key_out + off) = out0;
        *reinterpret_cast<char4*>(key_out + off + 4) = out1;
      }
      if constexpr (ToFp8) {
        const uint32_t v_pack0 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[0]), value_to_float(v_values[1]),
            value_to_float(v_values[2]), value_to_float(v_values[3])));
        const uint32_t v_pack1 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[4]), value_to_float(v_values[5]),
            value_to_float(v_values[6]), value_to_float(v_values[7])));
        if constexpr (PrepackFp8Lane) {
          const int row_in_group = row & 63;
          const int col_tile = row_in_group >> 4;
          const int row_local = row_in_group & 15;
          const int lane_hi = row_local >> 3;
          const int gpr = (row_local & 7) >> 2;
          const int byte = row_local & 3;
          uint8_t* value_bytes = reinterpret_cast<uint8_t*>(value_out);
#pragma unroll
          for (int elem = 0; elem < PackElems; ++elem) {
        const int d_elem = d + elem;
        const int d_tile = d_elem >> 4;
        const int lane_local = (d_elem & 15) | (lane_hi << 4);
        const int64_t byte_off =
            (((((static_cast<int64_t>(b) * kv_heads + head) * k_groups +
                    local_group) * LaneColTiles + col_tile) * LaneDTiles + d_tile) * 32 +
                 lane_local) * 8 +
                gpr * 4 + byte;
            const uint32_t packed = elem < 4 ? v_pack0 : v_pack1;
            value_bytes[byte_off] =
                static_cast<uint8_t>((packed >> (8 * (elem & 3))) & 0xff);
          }
        } else if constexpr (TransposeValue) {
          *reinterpret_cast<uint32_t*>(&value_tile[row][d]) = v_pack0;
          *reinterpret_cast<uint32_t*>(&value_tile[row][d + 4]) = v_pack1;
        } else {
          *reinterpret_cast<uint32_t*>(value_out + off) = v_pack0;
          *reinterpret_cast<uint32_t*>(value_out + off + 4) = v_pack1;
        }
      } else {
#pragma unroll
        for (int i = 0; i < PackElems; ++i) {
          if constexpr (std::is_same<T, __half>::value && std::is_same<OutT, __half>::value) {
            value_tile[row][d + i] = v_values[i];
          } else {
            const float v = value_to_float(v_values[i]);
            value_tile[row][d + i] = __float2half_rn(v);
          }
        }
      }
    }
  }
  if constexpr ((!ToFp8 || TransposeValue) && !PrepackFp8Lane) {
    __syncthreads();

    if constexpr (PrepackF16KLane) {
      constexpr int ColTiles64 = KRows / 16;
      constexpr int DTiles = HeadDim / 16;
      constexpr int LaneMajorElems = ColTiles64 * DTiles * 32;
      for (int idx = tid; idx < LaneMajorElems; idx += Threads) {
        const int lane_local = idx & 31;
        const int d_tile = (idx >> 5) % DTiles;
        const int col_tile = idx / (DTiles * 32);
        const int col = lane_local & 15;
        const int k_base = 8 * (lane_local >> 4);
        const int row = col_tile * 16 + pv_k_order_for_acc_row(col);
        const int d = d_tile * 16 + k_base;
        const int64_t out_off =
            ((((static_cast<int64_t>(b) * kv_heads + head) * k_groups + local_group) *
              ColTiles64 * DTiles * 32) + idx) * 8;
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&key_tile[row][d]);
        packed.y = *reinterpret_cast<const uint32_t*>(&key_tile[row][d + 4]);
        *reinterpret_cast<uint2*>(key_out + out_off) = packed;
      }
    }

    if constexpr (PrepackF16VLane) {
      constexpr int ColTiles64 = KRows / 16;
      constexpr int DTiles = HeadDim / 16;
      constexpr int LaneMajorElems = ColTiles64 * DTiles * 32;
      for (int idx = tid; idx < LaneMajorElems; idx += Threads) {
        const int lane_local = idx & 31;
        const int d_tile = (idx >> 5) % DTiles;
        const int col_tile = idx / (DTiles * 32);
        const int d = d_tile * 16 + (lane_local & 15);
        const int high_half = (lane_local >> 4) & 1;
        const int n0 = col_tile * 16 + high_half * 4;
        const int n1 = col_tile * 16 + 8 + high_half * 4;
        uint4 packed;
        __half* vals = reinterpret_cast<__half*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
        const int row = n0 + i;
          vals[i] = (StaticFullKV || (base_row + row) < effective_kv_len) ?
              value_tile[row][d] : __float2half_rn(0.0f);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = n1 + i;
          vals[4 + i] =
              (StaticFullKV || (base_row + row) < effective_kv_len) ?
              value_tile[row][d] : __float2half_rn(0.0f);
        }
        const int64_t out_off =
            ((((static_cast<int64_t>(b) * kv_heads + head) * k_groups + local_group) *
              ColTiles64 * DTiles * 32) + idx) * 8;
        *reinterpret_cast<uint4*>(value_out + out_off) = packed;
      }
    } else if constexpr (ToFp8) {
      constexpr int StoreRows = 16;
      constexpr int RowGroups = KRows / StoreRows;
      for (int linear = tid; linear < HeadDim * RowGroups; linear += Threads) {
        const int d = linear / RowGroups;
        const int row = (linear - d * RowGroups) * StoreRows;
        const int64_t seq = base_row + row;
        if (StaticFullKV || seq + StoreRows - 1 < effective_kv_len) {
          uint4 packed;
          packed.x = static_cast<uint32_t>(value_tile[row + 0][d]) |
              (static_cast<uint32_t>(value_tile[row + 1][d]) << 8) |
              (static_cast<uint32_t>(value_tile[row + 2][d]) << 16) |
              (static_cast<uint32_t>(value_tile[row + 3][d]) << 24);
          packed.y = static_cast<uint32_t>(value_tile[row + 4][d]) |
              (static_cast<uint32_t>(value_tile[row + 5][d]) << 8) |
              (static_cast<uint32_t>(value_tile[row + 6][d]) << 16) |
              (static_cast<uint32_t>(value_tile[row + 7][d]) << 24);
          packed.z = static_cast<uint32_t>(value_tile[row + 8][d]) |
              (static_cast<uint32_t>(value_tile[row + 9][d]) << 8) |
              (static_cast<uint32_t>(value_tile[row + 10][d]) << 16) |
              (static_cast<uint32_t>(value_tile[row + 11][d]) << 24);
          packed.w = static_cast<uint32_t>(value_tile[row + 12][d]) |
              (static_cast<uint32_t>(value_tile[row + 13][d]) << 8) |
              (static_cast<uint32_t>(value_tile[row + 14][d]) << 16) |
              (static_cast<uint32_t>(value_tile[row + 15][d]) << 24);
          const int64_t out_off =
              ((static_cast<int64_t>(b) * kv_heads + head) * HeadDim + d) *
                  effective_kv_len + seq;
          *reinterpret_cast<uint4*>(value_out + out_off) = packed;
        } else {
#pragma unroll
          for (int i = 0; i < StoreRows; ++i) {
            const int64_t tail_seq = seq + i;
            if (tail_seq < effective_kv_len) {
              const int64_t out_off =
                  ((static_cast<int64_t>(b) * kv_heads + head) * HeadDim + d) *
                      effective_kv_len +
                  tail_seq;
              value_out[out_off] = value_tile[row + i][d];
            }
          }
        }
      }
    } else {
      constexpr int StoreRows = 8;
      constexpr int RowGroups = KRows / StoreRows;
      for (int linear = tid; linear < HeadDim * RowGroups; linear += Threads) {
        const int d = linear / RowGroups;
        const int row = (linear - d * RowGroups) * StoreRows;
        const int64_t seq = base_row + row;
        const int64_t out_off =
            ((static_cast<int64_t>(b) * kv_heads + head) * HeadDim + d) *
                effective_kv_len + seq;
        if (StaticFullKV || seq + StoreRows - 1 < effective_kv_len) {
          uint4 packed;
          __half* vals = reinterpret_cast<__half*>(&packed);
#pragma unroll
          for (int i = 0; i < StoreRows; ++i) {
            vals[i] = value_tile[row + i][d];
          }
          *reinterpret_cast<uint4*>(value_out + out_off) = packed;
        } else {
#pragma unroll
          for (int i = 0; i < StoreRows; ++i) {
            const int64_t tail_seq = seq + i;
            if (tail_seq < effective_kv_len) {
              value_out[out_off + i] = value_tile[row + i][d];
            }
          }
        }
      }
    }
  }
}

template <typename T,
          int GroupsPerBlock,
          bool TransposedValueStaging = false,
          bool LaneMajorKV = false,
          int HeadDim = 64,
          bool CacheKeyInShared = false,
          bool LaneMajorKOnly = false,
          bool LaneMajorVOnly = false,
          bool FullGroupsNoTail = false,
          int PrepThreads = 256,
          int StaticKvLen = 0>
__global__ void prepare_kv_hnd_fp8_kernel(
    const T* __restrict__ key,
    const T* __restrict__ value,
    int8_t* __restrict__ key_out,
    float* __restrict__ key_scale,
    uint8_t* __restrict__ value_out,
    const int64_t batch,
    const int64_t kv_heads,
    const int64_t kv_len,
    const int k_groups) {
  constexpr int PackElems = 8;
  constexpr int GroupRows = 64;
  constexpr int KRows = GroupRows * GroupsPerBlock;
  constexpr int Threads = PrepThreads;
  constexpr int ValueStride = HeadDim + 16;
  constexpr int StoreRows = 16;
  constexpr int RowGroups = KRows / StoreRows;
  constexpr int ValueRowGroups4 = KRows / 4;
  constexpr int ValueStride32 = ValueRowGroups4 + 4;
  constexpr int LaneColTiles = GroupRows / 16;
  constexpr int LaneDTiles = HeadDim / 16;
  constexpr int Packs = (KRows * HeadDim) / PackElems;
  constexpr bool LaneMajorK = LaneMajorKV || LaneMajorKOnly;
  constexpr bool LaneMajorV = LaneMajorKV || LaneMajorVOnly;
  static_assert(HeadDim == 64 || HeadDim == 128,
                "fp8 KV preparation supports D64/D128.");
  static_assert(GroupsPerBlock == 1 || GroupsPerBlock == 2 || GroupsPerBlock == 4,
                "fp8 KV preparation supports 1, 2, or 4 scale groups per CTA.");
  static_assert(PrepThreads == 128 || PrepThreads == 256 || PrepThreads == 512,
                "fp8 KV preparation supports 128, 256, or 512 threads.");
  static_assert(!CacheKeyInShared || GroupsPerBlock == 1,
                "cached-key fp8 KV preparation is specialized for one scale group per CTA.");
  static_assert(!LaneMajorKOnly || ((HeadDim == 64 || HeadDim == 128) && !LaneMajorKV),
                "K-only lane-major fp8 preparation is specialized for D64/D128.");
  static_assert(!LaneMajorVOnly ||
                    ((HeadDim == 64 || HeadDim == 128) &&
                     !LaneMajorKV && !TransposedValueStaging),
                "V-only lane-major fp8 preparation is specialized for D64/D128.");
  static_assert(!FullGroupsNoTail || GroupsPerBlock == 1,
                "no-tail fp8 KV preparation is specialized for one scale group per CTA.");
  static_assert(StaticKvLen == 0 || (StaticKvLen % GroupRows) == 0,
                "static fp8 KV preparation length must be a whole scale group.");

  __shared__ float shared_amax[GroupsPerBlock];
  __shared__ float shared_pair_amax[GroupsPerBlock][32];
  __shared__ uint4 key_cache[CacheKeyInShared ? Packs : 1];
  __shared__ uint8_t value_tile[LaneMajorV ? 1 : (TransposedValueStaging ? 1 : KRows)]
                               [LaneMajorV ? 1 : ValueStride];
  __shared__ uint32_t value_tile_t[LaneMajorV ? 1 : (TransposedValueStaging ? HeadDim : 1)]
                                  [LaneMajorV ? 1 : (TransposedValueStaging ? ValueStride32 : 1)];

  const int block_group = blockIdx.x;
  const int head = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int group_base = block_group * GroupsPerBlock;
  constexpr int StaticKGroups = StaticKvLen == 0 ? 0 : (StaticKvLen / GroupRows);
  const int effective_k_groups = StaticKvLen == 0 ? k_groups : StaticKGroups;
  const int64_t effective_kv_len = StaticKvLen == 0 ? kv_len : StaticKvLen;
  const int64_t head_seq_base =
      (static_cast<int64_t>(b) * kv_heads + head) * effective_kv_len;
  const int64_t transposed_value_head_base =
      (static_cast<int64_t>(b) * kv_heads + head) * HeadDim * effective_kv_len;
  const int64_t scale_head_base =
      (static_cast<int64_t>(b) * kv_heads + head) * effective_k_groups;
  const int64_t base_row = static_cast<int64_t>(group_base) * GroupRows;
  if (b >= batch || head >= kv_heads || group_base >= effective_k_groups ||
      base_row >= effective_kv_len) {
    return;
  }

  float local_amax[GroupsPerBlock];
#pragma unroll
  for (int group = 0; group < GroupsPerBlock; ++group) {
    local_amax[group] = 0.0000001f;
  }
  for (int pack = tid; pack < Packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (FullGroupsNoTail || seq < effective_kv_len) {
      const int64_t off = (head_seq_base + seq) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(key + off);
      if constexpr (CacheKeyInShared) {
        key_cache[pack] = raw;
      }
      const T* values = reinterpret_cast<const T*>(&raw);
      float pack_amax = 0.0000001f;
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        pack_amax = fmaxf(pack_amax, fabsf(value_to_float(values[i])));
      }
      const int group = row >> 6;
      local_amax[group] = fmaxf(local_amax[group], pack_amax);
    }
  }

#pragma unroll
  for (int group = 0; group < GroupsPerBlock; ++group) {
    local_amax[group] = vllm::warpReduceMax(local_amax[group]);
    if (lane == 0) {
      shared_pair_amax[group][wid] = local_amax[group];
    }
  }
  __syncthreads();
  const bool warp_lane_active = tid < (blockDim.x / 32);
#pragma unroll
  for (int group = 0; group < GroupsPerBlock; ++group) {
    float group_amax = warp_lane_active ? shared_pair_amax[group][lane] : -1e20f;
    group_amax = vllm::warpReduceMax(group_amax);
    if (tid == 0) {
      shared_amax[group] = group_amax;
      if (FullGroupsNoTail || group_base + group < effective_k_groups) {
        key_scale[scale_head_base + group_base + group] = group_amax / 127.0f;
      }
    }
  }
  __syncthreads();
  float inv_scales[GroupsPerBlock];
#pragma unroll
  for (int group = 0; group < GroupsPerBlock; ++group) {
    inv_scales[group] =
        (FullGroupsNoTail || (group_base + group) < effective_k_groups) ?
        (127.0f / shared_amax[group]) : 0.0f;
  }

  for (int pack = tid; pack < Packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (FullGroupsNoTail || seq < effective_kv_len) {
      const int64_t off = (head_seq_base + seq) * HeadDim + d;
      const uint4 raw_k = CacheKeyInShared ? key_cache[pack] :
          *reinterpret_cast<const uint4*>(key + off);
      const T* k_values = reinterpret_cast<const T*>(&raw_k);
      const float inv_scale = inv_scales[row >> 6];
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_rn_gfx12(value_to_float(k_values[0]) * inv_scale);
      out0.y = float_to_int8_rn_gfx12(value_to_float(k_values[1]) * inv_scale);
      out0.z = float_to_int8_rn_gfx12(value_to_float(k_values[2]) * inv_scale);
      out0.w = float_to_int8_rn_gfx12(value_to_float(k_values[3]) * inv_scale);
      out1.x = float_to_int8_rn_gfx12(value_to_float(k_values[4]) * inv_scale);
      out1.y = float_to_int8_rn_gfx12(value_to_float(k_values[5]) * inv_scale);
      out1.z = float_to_int8_rn_gfx12(value_to_float(k_values[6]) * inv_scale);
      out1.w = float_to_int8_rn_gfx12(value_to_float(k_values[7]) * inv_scale);
      if constexpr (LaneMajorK) {
        const int group = row >> 6;
        const int row_in_group = row & 63;
        const int col_tile = row_in_group >> 4;
        const int col = row_in_group & 15;
        const int d_tile = d >> 4;
        const int lane_out = col | (((d & 8) != 0) ? 16 : 0);
        const int64_t lane_off =
            ((((static_cast<int64_t>(b) * kv_heads + head) * effective_k_groups +
               (group_base + group)) * LaneColTiles + col_tile) * LaneDTiles + d_tile) *
                32 * 8 +
            static_cast<int64_t>(lane_out) * 8;
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&out0);
        packed.y = *reinterpret_cast<const uint32_t*>(&out1);
        *reinterpret_cast<uint2*>(key_out + lane_off) = packed;
      } else {
        *reinterpret_cast<char4*>(key_out + off) = out0;
        *reinterpret_cast<char4*>(key_out + off + 4) = out1;
      }

      if constexpr (!TransposedValueStaging && !LaneMajorV) {
        const uint4 raw_v = *reinterpret_cast<const uint4*>(value + off);
        const T* v_values = reinterpret_cast<const T*>(&raw_v);
        const uint32_t v_pack0 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[0]), value_to_float(v_values[1]),
            value_to_float(v_values[2]), value_to_float(v_values[3])));
        const uint32_t v_pack1 = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(
            value_to_float(v_values[4]), value_to_float(v_values[5]),
            value_to_float(v_values[6]), value_to_float(v_values[7])));
        *reinterpret_cast<uint32_t*>(&value_tile[row][d]) = v_pack0;
        *reinterpret_cast<uint32_t*>(&value_tile[row][d + 4]) = v_pack1;
      }
    }
  }

  if constexpr (LaneMajorV) {
    constexpr int LaneMajorValueRegs =
        GroupsPerBlock * LaneColTiles * LaneDTiles * 32;
    for (int idx = tid; idx < LaneMajorValueRegs; idx += Threads) {
      const int lane_out = idx & 31;
      const int tile = idx >> 5;
      const int d_tile = tile % LaneDTiles;
      const int col_tile = (tile / LaneDTiles) % LaneColTiles;
      const int group = tile / (LaneDTiles * LaneColTiles);
      if constexpr (!FullGroupsNoTail) {
        if ((group_base + group) >= effective_k_groups) {
          continue;
        }
      }
      const int row_base =
          group * GroupRows + col_tile * 16 + ((lane_out >> 4) << 3);
      const int d = d_tile * 16 + (lane_out & 15);
      float values[8];
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        const int64_t seq = base_row + row_base + elem;
        if (FullGroupsNoTail || seq < effective_kv_len) {
          const int64_t off = (head_seq_base + seq) * HeadDim + d;
          values[elem] = value_to_float(value[off]);
        } else {
          values[elem] = 0.0f;
        }
      }
      const uint32_t packed0 = static_cast<uint32_t>(
          pack_f32x4_to_ocp_fp8(values[0], values[1], values[2], values[3]));
      const uint32_t packed1 = static_cast<uint32_t>(
          pack_f32x4_to_ocp_fp8(values[4], values[5], values[6], values[7]));
      const int64_t out_off =
          (((((static_cast<int64_t>(b) * kv_heads + head) * effective_k_groups +
              (group_base + group)) * LaneColTiles + col_tile) * LaneDTiles + d_tile) *
               32 +
           lane_out) *
          8;
      uint2 packed;
      packed.x = packed0;
      packed.y = packed1;
      *reinterpret_cast<uint2*>(value_out + out_off) = packed;
    }
  } else if constexpr (TransposedValueStaging) {
    for (int linear = tid; linear < HeadDim * ValueRowGroups4; linear += Threads) {
      const int d = linear / ValueRowGroups4;
      const int row4 = linear - d * ValueRowGroups4;
      const int row = row4 * 4;
      const int64_t seq = base_row + row;
      float v0 = 0.0f;
      float v1 = 0.0f;
      float v2 = 0.0f;
      float v3 = 0.0f;
      if (FullGroupsNoTail || seq < effective_kv_len) {
        const int64_t value_base = (head_seq_base + seq) * HeadDim + d;
        v0 = value_to_float(value[value_base + 0 * HeadDim]);
        if (FullGroupsNoTail || seq + 1 < effective_kv_len) {
          v1 = value_to_float(value[value_base + 1 * HeadDim]);
        }
        if (FullGroupsNoTail || seq + 2 < effective_kv_len) {
          v2 = value_to_float(value[value_base + 2 * HeadDim]);
        }
        if (FullGroupsNoTail || seq + 3 < effective_kv_len) {
          v3 = value_to_float(value[value_base + 3 * HeadDim]);
        }
      }
      value_tile_t[d][row4] = static_cast<uint32_t>(pack_f32x4_to_ocp_fp8(v0, v1, v2, v3));
    }
    __syncthreads();

    for (int linear = tid; linear < HeadDim * RowGroups; linear += Threads) {
      const int d = linear / RowGroups;
      const int row = (linear - d * RowGroups) * StoreRows;
      const int64_t seq = base_row + row;
      const int row4 = row >> 2;
      if constexpr (FullGroupsNoTail) {
        uint4 packed;
        packed.x = value_tile_t[d][row4 + 0];
        packed.y = value_tile_t[d][row4 + 1];
        packed.z = value_tile_t[d][row4 + 2];
        packed.w = value_tile_t[d][row4 + 3];
        const int64_t out_off = transposed_value_head_base +
            static_cast<int64_t>(d) * effective_kv_len + seq;
        *reinterpret_cast<uint4*>(value_out + out_off) = packed;
      } else if (seq + StoreRows - 1 < effective_kv_len) {
        uint4 packed;
        packed.x = value_tile_t[d][row4 + 0];
        packed.y = value_tile_t[d][row4 + 1];
        packed.z = value_tile_t[d][row4 + 2];
        packed.w = value_tile_t[d][row4 + 3];
        const int64_t out_off = transposed_value_head_base +
            static_cast<int64_t>(d) * effective_kv_len + seq;
        *reinterpret_cast<uint4*>(value_out + out_off) = packed;
      } else {
#pragma unroll
        for (int i = 0; i < StoreRows; ++i) {
          const int64_t tail_seq = seq + i;
          if (tail_seq < effective_kv_len) {
            const uint32_t packed = value_tile_t[d][row4 + (i >> 2)];
            const int64_t out_off = transposed_value_head_base +
                static_cast<int64_t>(d) * effective_kv_len + tail_seq;
            value_out[out_off] = static_cast<uint8_t>((packed >> (8 * (i & 3))) & 0xffu);
          }
        }
      }
    }
  } else {
    __syncthreads();

    for (int linear = tid; linear < HeadDim * RowGroups; linear += Threads) {
      const int d = linear / RowGroups;
      const int row = (linear - d * RowGroups) * StoreRows;
      const int64_t seq = base_row + row;
      if constexpr (FullGroupsNoTail) {
        uint4 packed;
        packed.x = static_cast<uint32_t>(value_tile[row + 0][d]) |
            (static_cast<uint32_t>(value_tile[row + 1][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 2][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 3][d]) << 24);
        packed.y = static_cast<uint32_t>(value_tile[row + 4][d]) |
            (static_cast<uint32_t>(value_tile[row + 5][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 6][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 7][d]) << 24);
        packed.z = static_cast<uint32_t>(value_tile[row + 8][d]) |
            (static_cast<uint32_t>(value_tile[row + 9][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 10][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 11][d]) << 24);
        packed.w = static_cast<uint32_t>(value_tile[row + 12][d]) |
            (static_cast<uint32_t>(value_tile[row + 13][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 14][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 15][d]) << 24);
        const int64_t out_off = transposed_value_head_base +
            static_cast<int64_t>(d) * effective_kv_len + seq;
        *reinterpret_cast<uint4*>(value_out + out_off) = packed;
      } else if (seq + StoreRows - 1 < effective_kv_len) {
        uint4 packed;
        packed.x = static_cast<uint32_t>(value_tile[row + 0][d]) |
            (static_cast<uint32_t>(value_tile[row + 1][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 2][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 3][d]) << 24);
        packed.y = static_cast<uint32_t>(value_tile[row + 4][d]) |
            (static_cast<uint32_t>(value_tile[row + 5][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 6][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 7][d]) << 24);
        packed.z = static_cast<uint32_t>(value_tile[row + 8][d]) |
            (static_cast<uint32_t>(value_tile[row + 9][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 10][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 11][d]) << 24);
        packed.w = static_cast<uint32_t>(value_tile[row + 12][d]) |
            (static_cast<uint32_t>(value_tile[row + 13][d]) << 8) |
            (static_cast<uint32_t>(value_tile[row + 14][d]) << 16) |
            (static_cast<uint32_t>(value_tile[row + 15][d]) << 24);
        const int64_t out_off = transposed_value_head_base +
            static_cast<int64_t>(d) * effective_kv_len + seq;
        *reinterpret_cast<uint4*>(value_out + out_off) = packed;
      } else {
#pragma unroll
        for (int i = 0; i < StoreRows; ++i) {
          const int64_t tail_seq = seq + i;
          if (tail_seq < effective_kv_len) {
            const int64_t out_off = transposed_value_head_base +
                static_cast<int64_t>(d) * effective_kv_len + tail_seq;
            value_out[out_off] = value_tile[row + i][d];
          }
        }
      }
    }
  }
}

} // namespace

static int select_fp8_d64_block_rows_gfx12(
    const int64_t q_len,
    const bool is_causal,
    const bool value_transposed_hnd) {
  if (is_causal) {
    if (q_len <= 64) {
      return 64;
    }
    return 128;
  }
  if (q_len <= 64) {
    return 64;
  }
  if ((q_len % 256) == 0 && (q_len >= 2048 || value_transposed_hnd)) {
    return 256;
  }
  return 128;
}

#if SAGEATTN_GFX12_BUILD_AUX

torch::Tensor transpose_value_fp8_hnd_gfx12(torch::Tensor value) {
  return transpose_value_hnd_gfx12<uint8_t, true>(value);
}

torch::Tensor transpose_value_fp8_scaled_hnd_gfx12(
    torch::Tensor value,
    torch::Tensor value_scale) {
  TORCH_CHECK(value.is_cuda() && value_scale.is_cuda(),
              "gfx12 scaled value transpose expects CUDA/HIP tensors");
  TORCH_CHECK(value.dim() == 4, "gfx12 scaled value transpose expects [B, H, S, D]");
  TORCH_CHECK(value.is_contiguous(), "gfx12 scaled value transpose expects contiguous HND input");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16 || value.scalar_type() == torch::kBFloat16,
              "gfx12 scaled value transpose supports fp16/bf16 input");
  TORCH_CHECK(value_scale.scalar_type() == torch::kFloat32,
              "gfx12 value scale must be fp32");
  TORCH_CHECK(value_scale.dim() == 3 && value_scale.is_contiguous(),
              "gfx12 value scale expects contiguous [B, H, D]");
  TORCH_CHECK(value_scale.size(0) == value.size(0) &&
                  value_scale.size(1) == value.size(1) &&
                  value_scale.size(2) == value.size(3),
              "gfx12 value scale shape must match [B, H, D]");

  const int64_t batch = value.size(0);
  const int64_t heads = value.size(1);
  const int64_t seq_len = value.size(2);
  const int64_t head_dim = value.size(3);
  torch::Tensor output =
      torch::empty({batch, heads, head_dim, seq_len}, value.options().dtype(torch::kUInt8));

  dim3 block(256);
  dim3 grid((seq_len + 127) / 128, (head_dim + 15) / 16, batch * heads);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == torch::kFloat16) {
    transpose_value_fp8_scaled_hnd_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        value_scale.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        batch * heads, seq_len, head_dim);
  } else {
    transpose_value_fp8_scaled_hnd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
        value_scale.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        batch * heads, seq_len, head_dim);
  }
  hip_kernel_launch_check();
  return output;
}

std::vector<torch::Tensor> fp8_value_nhd_short_gfx12(
    torch::Tensor value,
    double scale_max) {
  TORCH_CHECK(value.is_cuda(), "gfx12 short NHD value prep expects a CUDA/HIP tensor");
  TORCH_CHECK(value.dim() == 4, "gfx12 short NHD value prep expects [B, S, H, D]");
  TORCH_CHECK(value.is_contiguous(), "gfx12 short NHD value prep expects contiguous NHD input");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16 || value.scalar_type() == torch::kBFloat16,
              "gfx12 short NHD value prep supports fp16/bf16 input");

  const int64_t batch = value.size(0);
  const int64_t seq_len = value.size(1);
  const int64_t heads = value.size(2);
  const int64_t head_dim = value.size(3);
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "gfx12 short NHD fp8 value prep currently supports head_dim 64 or 128");
  TORCH_CHECK(seq_len == 512 || seq_len == 1024,
              "gfx12 short NHD fp8 value prep currently supports sequence length 512 or 1024");

  torch::Tensor output =
      torch::empty({batch, heads, head_dim, seq_len}, value.options().dtype(torch::kUInt8));
  torch::Tensor value_scale =
      torch::empty({batch, heads, head_dim}, value.options().dtype(torch::kFloat32));

  dim3 block(256);
  dim3 grid((head_dim + 15) / 16, heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == torch::kFloat16) {
    fp8_value_nhd_short_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        value_scale.data_ptr<float>(),
        seq_len, heads, head_dim, static_cast<float>(scale_max));
  } else {
    fp8_value_nhd_short_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
        output.data_ptr<uint8_t>(),
        value_scale.data_ptr<float>(),
        seq_len, heads, head_dim, static_cast<float>(scale_max));
  }
  hip_kernel_launch_check();
  return {output, value_scale};
}

torch::Tensor mean_nhd_gfx12(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "gfx12 NHD mean expects a CUDA/HIP tensor");
  TORCH_CHECK(input.dim() == 4, "gfx12 NHD mean expects [B, S, H, D]");
  TORCH_CHECK(input.is_contiguous(), "gfx12 NHD mean expects contiguous NHD input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "gfx12 NHD mean supports fp16/bf16 input");

  const int64_t batch = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t head_dim = input.size(3);
  torch::Tensor mean = torch::empty({batch, heads, head_dim}, input.options());

  const bool use_short_mean =
      (head_dim == 64 || head_dim == 128) && (seq_len == 512 || seq_len == 1024);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (use_short_mean) {
    dim3 block(1024);
    dim3 grid((head_dim + 31) / 32, heads, batch);
    if (input.scalar_type() == torch::kFloat16) {
      mean_nhd_short_kernel<__half, 32, 32><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(mean.data_ptr<at::Half>()),
          seq_len, heads, head_dim);
    } else {
      mean_nhd_short_kernel<__hip_bfloat16, 32, 32><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(input.data_ptr<at::BFloat16>()),
          reinterpret_cast<__hip_bfloat16*>(mean.data_ptr<at::BFloat16>()),
          seq_len, heads, head_dim);
    }
  } else {
    dim3 block(256);
    dim3 grid((head_dim + 15) / 16, heads, batch);
    if (input.scalar_type() == torch::kFloat16) {
      mean_nhd_kernel<__half><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(mean.data_ptr<at::Half>()),
          seq_len, heads, head_dim);
    } else {
      mean_nhd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(input.data_ptr<at::BFloat16>()),
          reinterpret_cast<__hip_bfloat16*>(mean.data_ptr<at::BFloat16>()),
          seq_len, heads, head_dim);
    }
  }
  hip_kernel_launch_check();
  return mean;
}

torch::Tensor mean_nhd_d64_seq32_gfx12(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "gfx12 D64 seq32 NHD mean expects a CUDA/HIP tensor");
  TORCH_CHECK(input.dim() == 4, "gfx12 D64 seq32 NHD mean expects [B, S, H, D]");
  TORCH_CHECK(input.is_contiguous(), "gfx12 D64 seq32 NHD mean expects contiguous NHD input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              "gfx12 D64 seq32 NHD mean supports fp16 input");

  const int64_t batch = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t head_dim = input.size(3);
  TORCH_CHECK(head_dim == 64,
              "gfx12 D64 seq32 NHD mean expects head_dim 64");
  TORCH_CHECK(seq_len == 2048 || seq_len == 4096 || seq_len == 8192,
              "gfx12 D64 seq32 NHD mean expects sequence length 2048/4096/8192");

  torch::Tensor mean = torch::empty({batch, heads, head_dim}, input.options());
  dim3 block(1024);
  dim3 grid((head_dim + 31) / 32, heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  mean_nhd_short_kernel<__half, 32, 32><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(mean.data_ptr<at::Half>()),
      seq_len, heads, head_dim);
  hip_kernel_launch_check();
  return mean;
}

torch::Tensor mean_hnd_gfx12(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "gfx12 HND mean expects a CUDA/HIP tensor");
  TORCH_CHECK(input.dim() == 4, "gfx12 HND mean expects [B, H, S, D]");
  TORCH_CHECK(input.is_contiguous(), "gfx12 HND mean expects contiguous HND input");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "gfx12 HND mean supports fp16/bf16 input");

  const int64_t batch = input.size(0);
  const int64_t heads = input.size(1);
  const int64_t seq_len = input.size(2);
  const int64_t head_dim = input.size(3);
  torch::Tensor mean = torch::empty({batch, heads, head_dim}, input.options());

  dim3 block(256);
  dim3 grid((head_dim + 15) / 16, heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (input.scalar_type() == torch::kFloat16) {
    mean_hnd_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(mean.data_ptr<at::Half>()),
        seq_len, heads, head_dim);
  } else {
    mean_hnd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(mean.data_ptr<at::BFloat16>()),
        seq_len, heads, head_dim);
  }
  hip_kernel_launch_check();
  return mean;
}

std::vector<torch::Tensor> mean_and_fp8_value_nhd_short_gfx12(
    torch::Tensor key,
    torch::Tensor value,
    double scale_max) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda(),
              "gfx12 short NHD mean/value prep expects CUDA/HIP tensors");
  TORCH_CHECK(key.dim() == 4 && value.dim() == 4,
              "gfx12 short NHD mean/value prep expects [B, S, H, D]");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
              "gfx12 short NHD mean/value prep expects contiguous NHD tensors");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "gfx12 short NHD mean/value prep expects matching key/value dtypes");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16 || key.scalar_type() == torch::kBFloat16,
              "gfx12 short NHD mean/value prep supports fp16/bf16 input");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "gfx12 short NHD mean/value prep expects matching key/value shapes");

  const int64_t batch = value.size(0);
  const int64_t seq_len = value.size(1);
  const int64_t heads = value.size(2);
  const int64_t head_dim = value.size(3);
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "gfx12 short NHD mean/value prep currently supports head_dim 64 or 128");
  TORCH_CHECK(seq_len == 512 || seq_len == 1024 || seq_len == 2048 ||
                  seq_len == 4096 || seq_len == 8192,
              "gfx12 NHD mean/value prep currently supports sequence length 512/1024/2048/4096/8192");

  torch::Tensor key_mean = torch::empty({batch, heads, head_dim}, key.options());
  torch::Tensor output =
      torch::empty({batch, heads, head_dim, seq_len}, value.options().dtype(torch::kUInt8));
  torch::Tensor value_scale =
      torch::empty({batch, heads, head_dim}, value.options().dtype(torch::kFloat32));

  const int seq_lanes = head_dim == 64 ? 32 : 16;
  dim3 block(32 * seq_lanes);
  dim3 grid((head_dim + 31) / 32, heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

#define SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT(T_, LANES_) \
  mean_and_fp8_value_nhd_short_kernel<T_, LANES_><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const T_*>(key.data_ptr()), \
      reinterpret_cast<const T_*>(value.data_ptr()), \
      reinterpret_cast<T_*>(key_mean.data_ptr()), \
      output.data_ptr<uint8_t>(), \
      value_scale.data_ptr<float>(), \
      seq_len, heads, head_dim, static_cast<float>(scale_max))

  if (value.scalar_type() == torch::kFloat16) {
    if (head_dim == 64) {
      SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT(__half, 32);
    } else {
      SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT(__half, 16);
    }
  } else {
    if (head_dim == 64) {
      SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT(__hip_bfloat16, 32);
    } else {
      SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT(__hip_bfloat16, 16);
    }
  }
#undef SAGEATTN_LAUNCH_MEAN_FP8_VALUE_SHORT
  hip_kernel_launch_check();
  return {key_mean, output, value_scale};
}

torch::Tensor transpose_value_f16_hnd_gfx12(torch::Tensor value) {
  return transpose_value_hnd_gfx12<__half, false>(value);
}

#endif // SAGEATTN_GFX12_BUILD_AUX

#if SAGEATTN_GFX12_BUILD_PREPARE

template <typename OutT, bool ToFp8>
std::vector<torch::Tensor> prepare_qkv_hnd_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "gfx12 QKV preparation expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "gfx12 QKV preparation expects [B, H, S, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "gfx12 QKV preparation expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
              "gfx12 QKV preparation expects matching input dtypes");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "gfx12 QKV preparation supports fp16/bf16 input");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
              "Q/K/V batch size mismatch");
  TORCH_CHECK(query.size(3) == key.size(3) && query.size(3) == value.size(3),
              "Q/K/V head_dim mismatch");
  TORCH_CHECK(key.size(1) == value.size(1) && key.size(2) == value.size(2),
              "K/V shape mismatch");

  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "gfx12 QKV preparation supports head_dim 16, 64, or 128");
  TORCH_CHECK((q_len % 64) == 0 && (kv_len % 64) == 0,
              "gfx12 QKV preparation requires sequence lengths divisible by 64");

  const int q_groups = static_cast<int>((q_len + 31) / 32);
  const int q_task_groups = (q_groups + 1) / 2;
  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  const bool fuse_self_qkv =
      q_heads == kv_heads && q_len == kv_len && q_task_groups == k_groups;
  torch::Tensor query_out = torch::empty_like(query, query.options().dtype(torch::kInt8));
  torch::Tensor key_out = torch::empty_like(key, key.options().dtype(torch::kInt8));
  torch::Tensor query_scale =
      torch::empty({batch, q_heads, q_groups}, query.options().dtype(torch::kFloat32));
  torch::Tensor key_scale =
      torch::empty({batch, kv_heads, k_groups}, key.options().dtype(torch::kFloat32));
  auto value_options = value.options().dtype(ToFp8 ? torch::kUInt8 : torch::kFloat16);
  torch::Tensor value_out = torch::empty({batch, kv_heads, head_dim, kv_len}, value_options);

  constexpr int D64PrepThreads = 256;
  const dim3 block(head_dim == 64 ? D64PrepThreads : 256);
  const dim3 grid(fuse_self_qkv ? k_groups : (q_task_groups + k_groups),
                  std::max(q_heads, kv_heads),
                  batch);
  const bool use_qkv_static_1024 =
      ToFp8 && head_dim == 128 && q_len == 1024 && kv_len == 1024 &&
      fuse_self_qkv;
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 16, 256><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr,
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         reinterpret_cast<OutT*>(value_out.data_ptr()),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 64, D64PrepThreads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr,
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         reinterpret_cast<OutT*>(value_out.data_ptr()),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else {
      if (use_qkv_static_1024) {
        prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 128, 256,
                                                   true, true, false, false, false,
                                                   true, 1024, 1024><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                           nullptr,
                           query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                           query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                           reinterpret_cast<OutT*>(value_out.data_ptr()),
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           true);
      } else {
        prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 128><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                           nullptr,
                           query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                           query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                           reinterpret_cast<OutT*>(value_out.data_ptr()),
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           fuse_self_qkv);
      }
    }
  } else {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 16, 256><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr,
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         reinterpret_cast<OutT*>(value_out.data_ptr()),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 64, D64PrepThreads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr,
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         reinterpret_cast<OutT*>(value_out.data_ptr()),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else {
      if (use_qkv_static_1024) {
        prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 128, 256,
                                                   true, true, false, false, false,
                                                   true, 1024, 1024><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                           nullptr,
                           query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                           query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                           reinterpret_cast<OutT*>(value_out.data_ptr()),
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           true);
      } else {
        prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 128><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                           nullptr,
                           query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                           query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                           reinterpret_cast<OutT*>(value_out.data_ptr()),
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           fuse_self_qkv);
      }
    }
  }
  hip_kernel_launch_check();
  return {query_out, query_scale, key_out, key_scale, value_out};
}

std::vector<torch::Tensor> prepare_qkv_hnd_smooth_f16_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && key_mean.is_cuda(),
              "smooth gfx12 QKV preparation expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "smooth gfx12 QKV preparation expects [B, H, S, D]");
  TORCH_CHECK(key_mean.dim() == 3,
              "smooth gfx12 QKV preparation expects key_mean [B, H, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
                  key_mean.is_contiguous(),
              "smooth gfx12 QKV preparation expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == key_mean.scalar_type(),
              "smooth gfx12 QKV preparation expects matching input dtypes");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "smooth gfx12 QKV preparation supports fp16/bf16 input");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
                  query.size(0) == key_mean.size(0),
              "Q/K/V batch size mismatch");
  TORCH_CHECK(query.size(3) == key.size(3) && query.size(3) == value.size(3) &&
                  query.size(3) == key_mean.size(2),
              "Q/K/V head_dim mismatch");
  TORCH_CHECK(key.size(1) == value.size(1) && key.size(2) == value.size(2) &&
                  key.size(1) == key_mean.size(1),
              "K/V shape mismatch");

  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "smooth gfx12 QKV preparation supports head_dim 64 or 128");
  TORCH_CHECK((q_len % 64) == 0 && (kv_len % 64) == 0,
              "smooth gfx12 QKV preparation requires sequence lengths divisible by 64");

  const int q_groups = static_cast<int>((q_len + 31) / 32);
  const int q_task_groups = (q_groups + 1) / 2;
  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  const bool fuse_self_qkv =
      q_heads == kv_heads && q_len == kv_len && q_task_groups == k_groups;
  torch::Tensor query_out = torch::empty_like(query, query.options().dtype(torch::kInt8));
  torch::Tensor key_out = torch::empty_like(key, key.options().dtype(torch::kInt8));
  torch::Tensor query_scale =
      torch::empty({batch, q_heads, q_groups}, query.options().dtype(torch::kFloat32));
  torch::Tensor key_scale =
      torch::empty({batch, kv_heads, k_groups}, key.options().dtype(torch::kFloat32));
  torch::Tensor value_out =
      torch::empty({batch, kv_heads, head_dim, kv_len}, value.options().dtype(torch::kFloat16));

  constexpr int D64PrepThreads = 256;
  const dim3 block(head_dim == 64 ? D64PrepThreads : 256);
  const dim3 grid(fuse_self_qkv ? k_groups : (q_task_groups + k_groups),
                  std::max(q_heads, kv_heads),
                  batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__half, __half, false, 64, D64PrepThreads,
                             true, true, false, false, false, false, 0, 0, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(key_mean.data_ptr<at::Half>()),
              query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
              query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
              reinterpret_cast<__half*>(value_out.data_ptr<at::Half>()),
              batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
              fuse_self_qkv);
    } else {
      prepare_qkv_hnd_kernel<__half, __half, false, 128, 256,
                             true, true, false, false, false, false, 0, 0, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(key_mean.data_ptr<at::Half>()),
              query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
              query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
              reinterpret_cast<__half*>(value_out.data_ptr<at::Half>()),
              batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
              fuse_self_qkv);
    }
  } else {
    if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, __half, false, 64, D64PrepThreads,
                             true, true, false, false, false, false, 0, 0, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(key_mean.data_ptr<at::BFloat16>()),
              query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
              query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
              reinterpret_cast<__half*>(value_out.data_ptr<at::Half>()),
              batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
              fuse_self_qkv);
    } else {
      prepare_qkv_hnd_kernel<__hip_bfloat16, __half, false, 128, 256,
                             true, true, false, false, false, false, 0, 0, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
              reinterpret_cast<const __hip_bfloat16*>(key_mean.data_ptr<at::BFloat16>()),
              query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
              query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
              reinterpret_cast<__half*>(value_out.data_ptr<at::Half>()),
              batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
              fuse_self_qkv);
    }
  }
  hip_kernel_launch_check();
  return {query_out, query_scale, key_out, key_scale, value_out};
}

template <typename OutT,
          bool ToFp8,
          bool TransposeValue = true,
          bool PrepackF16KLane = false,
          bool PrepackFp8Lane = false>
std::vector<torch::Tensor> prepare_qkv_hnd_packed_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "packed gfx12 QKV preparation expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "packed gfx12 QKV preparation expects [B, H, S, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "packed gfx12 QKV preparation expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
              "packed gfx12 QKV preparation expects matching input dtypes");

  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "packed gfx12 QKV preparation supports head_dim 16, 64, or 128");
  TORCH_CHECK(!PrepackF16KLane || (!ToFp8 && head_dim == 64),
              "fp16 lane-major K prepack requires D64 fp16/bf16 keys");
  TORCH_CHECK(!PrepackFp8Lane || (ToFp8 && TransposeValue && head_dim == 64),
              "fp8 lane-major K/V prepack requires transposed D64 fp8 values");
  TORCH_CHECK(!PrepackFp8Lane || ((key.size(2) % 64) == 0),
              "fp8 lane-major K/V prepack requires kv_len divisible by 64");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch,
              "Q/K/V batch size mismatch");
  TORCH_CHECK(key.size(2) == kv_len && value.size(2) == kv_len && value.size(1) == kv_heads,
              "K/V shape mismatch");
  TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
              "Q/K/V head_dim mismatch");

  const int q_groups = static_cast<int>((q_len + 31) / 32);
  const int q_task_groups = (q_groups + 1) / 2;
  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  const bool fuse_self_qkv =
      q_heads == kv_heads && q_len == kv_len && q_task_groups == k_groups;

  const int64_t q_numel = query.numel();
  const int64_t k_numel = key.numel();
  const int64_t v_numel = batch * kv_heads * head_dim * kv_len;
  const int64_t q_scale_numel = batch * q_heads * q_groups;
  const int64_t k_scale_numel = batch * kv_heads * k_groups;
  const bool pack_scales_with_bytes = ToFp8;
  const int64_t value_byte_offset = q_numel + k_numel;
  const int64_t scale_byte_offset = value_byte_offset + (ToFp8 ? v_numel : 0);
  torch::Tensor byte_workspace =
      torch::empty({scale_byte_offset +
                    (pack_scales_with_bytes ? (q_scale_numel + k_scale_numel) * 4 : 0)},
                   query.options().dtype(torch::kUInt8));
  uint8_t* byte_ptr = byte_workspace.data_ptr<uint8_t>();
  int8_t* query_ptr = reinterpret_cast<int8_t*>(byte_ptr);
  int8_t* key_ptr = reinterpret_cast<int8_t*>(byte_ptr + q_numel);
  torch::Tensor value_out;
  OutT* value_ptr = nullptr;
  if constexpr (ToFp8) {
    value_ptr = reinterpret_cast<OutT*>(byte_ptr + value_byte_offset);
    if constexpr (TransposeValue) {
      value_out = torch::from_blob(
          value_ptr, {batch, kv_heads, head_dim, kv_len}, value.options().dtype(torch::kUInt8));
    } else {
      value_out = torch::from_blob(
          value_ptr, {batch, kv_heads, kv_len, head_dim}, value.options().dtype(torch::kUInt8));
    }
  } else {
    value_out = torch::empty({batch, kv_heads, head_dim, kv_len},
                             value.options().dtype(torch::kFloat16));
    value_ptr = reinterpret_cast<OutT*>(value_out.data_ptr());
  }

  torch::Tensor scale_workspace;
  float* scale_ptr = nullptr;
  if (pack_scales_with_bytes) {
    scale_ptr = reinterpret_cast<float*>(byte_ptr + scale_byte_offset);
  } else {
    scale_workspace =
        torch::empty({q_scale_numel + k_scale_numel}, query.options().dtype(torch::kFloat32));
    scale_ptr = scale_workspace.data_ptr<float>();
  }

  torch::Tensor query_out = torch::from_blob(
      query_ptr, {batch, q_heads, q_len, head_dim}, query.options().dtype(torch::kInt8));
  torch::Tensor key_out = torch::from_blob(
      key_ptr, {batch, kv_heads, kv_len, head_dim}, key.options().dtype(torch::kInt8));
  torch::Tensor query_scale = torch::from_blob(
      scale_ptr, {batch, q_heads, q_groups}, query.options().dtype(torch::kFloat32));
  torch::Tensor key_scale = torch::from_blob(
      scale_ptr + q_scale_numel, {batch, kv_heads, k_groups}, key.options().dtype(torch::kFloat32));

  constexpr int D64PrepThreads = 256;
  const dim3 block(head_dim == 64 ? D64PrepThreads : 256);
  const dim3 grid(fuse_self_qkv ? k_groups : (q_task_groups + k_groups),
                  std::max(q_heads, kv_heads),
                  batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 16, 256, TransposeValue, true><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 64, D64PrepThreads, TransposeValue, true, false, PrepackF16KLane, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 128, 256, TransposeValue, true, false, false, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    }
  } else {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 16, 256, TransposeValue, true><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else if (head_dim == 64) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 64, D64PrepThreads, TransposeValue, true, false, PrepackF16KLane, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    } else {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 128, 256, TransposeValue, true, false, false, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr,
                         query_ptr, key_ptr, scale_ptr, scale_ptr + q_scale_numel,
                         value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         fuse_self_qkv);
    }
  }
  hip_kernel_launch_check();
  if (pack_scales_with_bytes) {
    return {query_out, query_scale, key_out, key_scale, value_out, byte_workspace};
  }
  return {query_out, query_scale, key_out, key_scale, value_out, byte_workspace, scale_workspace};
}

template <typename OutT,
          bool ToFp8,
          bool TransposeValue = true,
          bool PrepackF16VLane = false,
          bool PrepackF16KLane = false,
          bool PrepackFp8Lane = false,
          bool PrepackFp8VLane = false,
          bool PrepackFp8KLane = false>
std::vector<torch::Tensor> prepare_kv_hnd_packed_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "packed gfx12 KV preparation expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "packed gfx12 KV preparation expects [B, H, S, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "packed gfx12 KV preparation expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
              "packed gfx12 KV preparation expects matching input dtypes");

  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "packed gfx12 KV preparation supports head_dim 16, 64, or 128");
  TORCH_CHECK(!PrepackF16VLane || (!ToFp8 && TransposeValue && head_dim == 64),
              "fp16 lane-major V prepack requires transposed D64 fp16 values");
  TORCH_CHECK(!PrepackF16KLane || (!ToFp8 && head_dim == 64),
              "fp16 lane-major K prepack requires D64 fp16/bf16 keys");
  TORCH_CHECK(!PrepackFp8Lane || (ToFp8 && TransposeValue && head_dim == 64),
              "fp8 lane-major K/V prepack requires transposed D64 fp8 values");
  TORCH_CHECK(!PrepackFp8Lane || ((key.size(2) % 64) == 0),
              "fp8 lane-major K/V prepack requires kv_len divisible by 64");
  TORCH_CHECK(!PrepackFp8VLane ||
                  (ToFp8 && TransposeValue && (head_dim == 64 || head_dim == 128)),
              "fp8 lane-major V prepack requires transposed D64/D128 fp8 values");
  TORCH_CHECK(!PrepackFp8VLane || ((key.size(2) % 64) == 0),
              "fp8 lane-major V prepack requires kv_len divisible by 64");
  TORCH_CHECK(!PrepackFp8KLane ||
                  (ToFp8 && TransposeValue && (head_dim == 64 || head_dim == 128)),
              "fp8 lane-major K prepack requires transposed D64/D128 fp8 values");
  TORCH_CHECK(!PrepackFp8KLane || ((key.size(2) % 64) == 0),
              "fp8 lane-major K prepack requires kv_len divisible by 64");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch,
              "Q/K/V batch size mismatch");
  TORCH_CHECK(key.size(2) == kv_len && value.size(2) == kv_len && value.size(1) == kv_heads,
              "K/V shape mismatch");
  TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
              "Q/K/V head_dim mismatch");

  const int q_groups = static_cast<int>((q_len + 31) / 32);
  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  const int64_t k_numel = key.numel();
  const int64_t v_numel = batch * kv_heads * head_dim * kv_len;
  const int64_t k_scale_numel = batch * kv_heads * k_groups;
  const int64_t value_byte_offset = k_numel;
  const int64_t scale_byte_offset = value_byte_offset + (ToFp8 ? v_numel : 0);
  torch::Tensor byte_workspace =
      torch::empty({scale_byte_offset + (ToFp8 ? k_scale_numel * 4 : 0)},
                   query.options().dtype(torch::kUInt8));
  uint8_t* byte_ptr = byte_workspace.data_ptr<uint8_t>();
  int8_t* key_ptr = reinterpret_cast<int8_t*>(byte_ptr);
  OutT* value_ptr = nullptr;
  torch::Tensor value_out;
  if constexpr (ToFp8) {
    value_ptr = reinterpret_cast<OutT*>(byte_ptr + value_byte_offset);
    if constexpr (TransposeValue) {
      value_out = torch::from_blob(
          value_ptr, {batch, kv_heads, head_dim, kv_len}, value.options().dtype(torch::kUInt8));
    } else {
      value_out = torch::from_blob(
          value_ptr, {batch, kv_heads, kv_len, head_dim}, value.options().dtype(torch::kUInt8));
    }
  } else {
    if constexpr (PrepackF16VLane) {
      value_out = torch::empty({batch, kv_heads, k_groups, 4, 4, 32, 8},
                               value.options().dtype(torch::kFloat16));
    } else {
      value_out = torch::empty({batch, kv_heads, head_dim, kv_len},
                               value.options().dtype(torch::kFloat16));
    }
    value_ptr = reinterpret_cast<OutT*>(value_out.data_ptr());
  }
  torch::Tensor scale_workspace;
  float* key_scale_ptr = nullptr;
  if constexpr (ToFp8) {
    key_scale_ptr = reinterpret_cast<float*>(byte_ptr + scale_byte_offset);
  } else {
    scale_workspace = torch::empty({k_scale_numel}, query.options().dtype(torch::kFloat32));
    key_scale_ptr = scale_workspace.data_ptr<float>();
  }
  torch::Tensor key_out = torch::from_blob(
      key_ptr, {batch, kv_heads, kv_len, head_dim}, key.options().dtype(torch::kInt8));
  torch::Tensor key_scale = torch::from_blob(
      key_scale_ptr, {batch, kv_heads, k_groups}, key.options().dtype(torch::kFloat32));

  constexpr int D64PrepThreads = 256;
  const dim3 block(head_dim == 64 ? D64PrepThreads : 256);
  const dim3 grid(k_groups, kv_heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_kv1 = q_len <= 4096;
  const bool use_kv_static_1024 =
      ToFp8 && TransposeValue && use_kv1 &&
      q_len == 1024 && kv_len == 1024 &&
      !PrepackFp8VLane && !PrepackFp8KLane;
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 16, 256, TransposeValue, false><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                         nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         false);
    } else if (head_dim == 64) {
      if constexpr (ToFp8 && TransposeValue) {
        if (use_kv1) {
          if (use_kv_static_1024) {
            prepare_kv_hnd_fp8_kernel<__half, 1, false, PrepackFp8Lane, 64, false, false, false, true, 256, 1024><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                               reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          } else {
            prepare_kv_hnd_fp8_kernel<__half, 1, false, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                               reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          }
        } else {
          const dim3 grid_kv((k_groups + 1) / 2, kv_heads, batch);
          prepare_kv_hnd_fp8_kernel<__half, 2, false, PrepackFp8Lane><<<grid_kv, block, 0, stream>>>(
                             reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                             reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                             key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                             batch, kv_heads, kv_len, k_groups);
        }
      } else {
        prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 64, D64PrepThreads, TransposeValue, false, PrepackF16VLane, PrepackF16KLane><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                           nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           false);
      }
    } else {
      if constexpr (ToFp8 && TransposeValue) {
        if (use_kv1) {
          if (use_kv_static_1024) {
            prepare_kv_hnd_fp8_kernel<__half, 1, false, false, 128, false, false, false, true, 256, 1024><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                               reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          } else {
            prepare_kv_hnd_fp8_kernel<__half, 1, false, false, 128><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                               reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          }
        } else {
          const dim3 grid_kv((k_groups + 1) / 2, kv_heads, batch);
          prepare_kv_hnd_fp8_kernel<__half, 2, false, false, 128><<<grid_kv, block, 0, stream>>>(
                             reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                             reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                             key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                             batch, kv_heads, kv_len, k_groups);
        }
      } else {
        prepare_qkv_hnd_kernel<__half, OutT, ToFp8, 128, 256, TransposeValue, false><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                           reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                           nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           false);
      }
    }
  } else {
    if (head_dim == 16) {
      prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 16, 256, TransposeValue, false><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                         nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                         false);
    } else if (head_dim == 64) {
      if constexpr (ToFp8 && TransposeValue) {
        if (use_kv1) {
          if (use_kv_static_1024) {
            prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 1, false, PrepackFp8Lane, 64, false, false, false, true, 256, 1024><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                               reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          } else {
            prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 1, false, PrepackFp8Lane><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                               reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          }
        } else {
          const dim3 grid_kv((k_groups + 1) / 2, kv_heads, batch);
          prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 2, false, PrepackFp8Lane><<<grid_kv, block, 0, stream>>>(
                             reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                             reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                             key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                             batch, kv_heads, kv_len, k_groups);
        }
      } else {
        prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 64, D64PrepThreads, TransposeValue, false, PrepackF16VLane, PrepackF16KLane><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                           nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           false);
      }
    } else {
      if constexpr (ToFp8 && TransposeValue) {
        if (use_kv1) {
          if (use_kv_static_1024) {
            prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 1, false, false, 128, false, false, false, true, 256, 1024><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                               reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          } else {
            prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 1, false, false, 128><<<grid, block, 0, stream>>>(
                               reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                               reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                               key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                               batch, kv_heads, kv_len, k_groups);
          }
        } else {
          const dim3 grid_kv((k_groups + 1) / 2, kv_heads, batch);
          prepare_kv_hnd_fp8_kernel<__hip_bfloat16, 2, false, false, 128><<<grid_kv, block, 0, stream>>>(
                             reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                             reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                             key_ptr, key_scale_ptr, reinterpret_cast<uint8_t*>(value_ptr),
                             batch, kv_heads, kv_len, k_groups);
        }
      } else {
        prepare_qkv_hnd_kernel<__hip_bfloat16, OutT, ToFp8, 128, 256, TransposeValue, false><<<grid, block, 0, stream>>>(
                           reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                           reinterpret_cast<const __hip_bfloat16*>(value.data_ptr<at::BFloat16>()),
                           nullptr, nullptr, key_ptr, nullptr, key_scale_ptr, value_ptr,
                           batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups,
                           false);
      }
    }
  }
  hip_kernel_launch_check();
  if constexpr (ToFp8) {
    return {key_out, key_scale, value_out, byte_workspace};
  }
  return {key_out, key_scale, value_out, byte_workspace, scale_workspace};
}

template <typename T, int HeadDim, int Threads = 256>
__global__ void prepare_k_hnd_kernel(
    const T* __restrict__ key,
    int8_t* __restrict__ key_out,
    float* __restrict__ key_scale,
    const int64_t batch,
    const int64_t kv_heads,
    const int64_t kv_len,
    const int k_groups) {
  constexpr int PackElems = 8;
  constexpr int KRows = 64;
  const int local_group = blockIdx.x;
  const int head = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int64_t base_row = static_cast<int64_t>(local_group) * KRows;
  if (b >= batch || head >= kv_heads || local_group >= k_groups || base_row >= kv_len) {
    return;
  }

  __shared__ float shared_amax;
  constexpr int kv_packs = (KRows * HeadDim) / PackElems;
  float local_amax = 0.0000001f;
  for (int pack = tid; pack < kv_packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (seq < kv_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * kv_heads + head) * kv_len + seq) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(key + off);
      const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int i = 0; i < PackElems; ++i) {
        local_amax = fmaxf(local_amax, fabsf(value_to_float(values[i])));
      }
    }
  }

  const float block_amax = vllm::blockReduceMax(local_amax);
  if (tid == 0) {
    shared_amax = block_amax;
    key_scale[(static_cast<int64_t>(b) * kv_heads + head) * k_groups + local_group] =
        block_amax / 127.0f;
  }
  __syncthreads();
  const float inv_scale = 127.0f / shared_amax;

  for (int pack = tid; pack < kv_packs; pack += Threads) {
    const int elem_base = pack * PackElems;
    const int row = elem_base / HeadDim;
    const int d = elem_base - row * HeadDim;
    const int64_t seq = base_row + row;
    if (seq < kv_len) {
      const int64_t off =
          ((static_cast<int64_t>(b) * kv_heads + head) * kv_len + seq) * HeadDim + d;
      const uint4 raw = *reinterpret_cast<const uint4*>(key + off);
      const T* values = reinterpret_cast<const T*>(&raw);
      char4 out0;
      char4 out1;
      out0.x = float_to_int8_rn_gfx12(value_to_float(values[0]) * inv_scale);
      out0.y = float_to_int8_rn_gfx12(value_to_float(values[1]) * inv_scale);
      out0.z = float_to_int8_rn_gfx12(value_to_float(values[2]) * inv_scale);
      out0.w = float_to_int8_rn_gfx12(value_to_float(values[3]) * inv_scale);
      out1.x = float_to_int8_rn_gfx12(value_to_float(values[4]) * inv_scale);
      out1.y = float_to_int8_rn_gfx12(value_to_float(values[5]) * inv_scale);
      out1.z = float_to_int8_rn_gfx12(value_to_float(values[6]) * inv_scale);
      out1.w = float_to_int8_rn_gfx12(value_to_float(values[7]) * inv_scale);
      *reinterpret_cast<char4*>(key_out + off) = out0;
      *reinterpret_cast<char4*>(key_out + off + 4) = out1;
    }
  }
}

std::vector<torch::Tensor> prepare_k_hnd_packed_gfx12(torch::Tensor key) {
  TORCH_CHECK(key.is_cuda(), "packed gfx12 K preparation expects a CUDA/HIP tensor");
  TORCH_CHECK(key.dim() == 4, "packed gfx12 K preparation expects [B, H, S, D]");
  TORCH_CHECK(key.is_contiguous(), "packed gfx12 K preparation expects contiguous HND tensors");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16 || key.scalar_type() == torch::kBFloat16,
              "packed gfx12 K preparation supports fp16/bf16 input");
  const int64_t batch = key.size(0);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = key.size(3);
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "packed gfx12 K preparation supports head_dim 64 or 128");
  TORCH_CHECK((kv_len % 64) == 0,
              "packed gfx12 K preparation requires sequence length divisible by 64");

  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  const int64_t k_numel = key.numel();
  torch::Tensor byte_workspace = torch::empty({k_numel}, key.options().dtype(torch::kUInt8));
  torch::Tensor scale_workspace =
      torch::empty({batch * kv_heads * k_groups}, key.options().dtype(torch::kFloat32));
  int8_t* key_ptr = reinterpret_cast<int8_t*>(byte_workspace.data_ptr<uint8_t>());
  torch::Tensor key_out = torch::from_blob(
      key_ptr, {batch, kv_heads, kv_len, head_dim}, key.options().dtype(torch::kInt8));
  torch::Tensor key_scale = torch::from_blob(
      scale_workspace.data_ptr<float>(), {batch, kv_heads, k_groups},
      key.options().dtype(torch::kFloat32));

  constexpr int Threads = 256;
  const dim3 block(Threads);
  const dim3 grid(k_groups, kv_heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (key.scalar_type() == torch::kFloat16) {
    if (head_dim == 64) {
      prepare_k_hnd_kernel<__half, 64, Threads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         key_ptr, scale_workspace.data_ptr<float>(),
                         batch, kv_heads, kv_len, k_groups);
    } else {
      prepare_k_hnd_kernel<__half, 128, Threads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         key_ptr, scale_workspace.data_ptr<float>(),
                         batch, kv_heads, kv_len, k_groups);
    }
  } else {
    if (head_dim == 64) {
      prepare_k_hnd_kernel<__hip_bfloat16, 64, Threads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         key_ptr, scale_workspace.data_ptr<float>(),
                         batch, kv_heads, kv_len, k_groups);
    } else {
      prepare_k_hnd_kernel<__hip_bfloat16, 128, Threads><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         key_ptr, scale_workspace.data_ptr<float>(),
                         batch, kv_heads, kv_len, k_groups);
    }
  }
  hip_kernel_launch_check();
  return {key_out, key_scale, byte_workspace, scale_workspace};
}

#endif // SAGEATTN_GFX12_BUILD_PREPARE

#if SAGEATTN_GFX12_BUILD_AUX

__global__ void convert_f16_to_bf16_kernel(
    const __half* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    const int64_t numel) {
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (idx + 1 < numel) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(input + idx);
    const __hip_bfloat162 b2 = __float22bfloat162_rn(__half22float2(h2));
    *reinterpret_cast<__hip_bfloat162*>(output + idx) = b2;
  } else if (idx < numel) {
    output[idx] = __float2bfloat16(__half2float(input[idx]));
  }
}

torch::Tensor convert_f16_to_bf16_gfx12(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "gfx12 f16 to bf16 conversion expects a CUDA/HIP tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              "gfx12 f16 to bf16 conversion expects fp16 input");
  TORCH_CHECK(input.is_contiguous(),
              "gfx12 f16 to bf16 conversion expects contiguous input");
  torch::Tensor output = torch::empty(input.sizes(), input.options().dtype(torch::kBFloat16));
  const int64_t numel = input.numel();
  const int threads = 256;
  const dim3 block(threads);
  const dim3 grid((numel + threads * 2 - 1) / (threads * 2));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  convert_f16_to_bf16_kernel<<<grid, block, 0, stream>>>(
                     reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
                     reinterpret_cast<__hip_bfloat16*>(output.data_ptr<at::BFloat16>()),
                     numel);
  hip_kernel_launch_check();
  return output;
}

std::vector<torch::Tensor> quant_q_nhd_per_warp_gfx12(torch::Tensor query) {
  TORCH_CHECK(query.is_cuda(), "gfx12 NHD Q quantization expects a CUDA/HIP tensor");
  TORCH_CHECK(query.dim() == 4, "gfx12 NHD Q quantization expects [B, S, H, D]");
  TORCH_CHECK(query.is_contiguous(), "gfx12 NHD Q quantization expects contiguous NHD input");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "gfx12 NHD Q quantization supports fp16/bf16 input");
  const int64_t batch = query.size(0);
  const int64_t q_len = query.size(1);
  const int64_t q_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "gfx12 NHD Q quantization supports head_dim 16, 64, or 128");
  const int q_scale_groups = static_cast<int>(((q_len + 127) / 128) * 4);
  torch::Tensor query_out = torch::empty_like(query, query.options().dtype(torch::kInt8));
  torch::Tensor query_scale =
      torch::empty({batch, q_heads, q_scale_groups}, query.options().dtype(torch::kFloat32));

  const dim3 block(256);
  const dim3 grid((q_scale_groups + 1) / 2, q_heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 16) {
      quant_q_nhd_per_warp_kernel<__half, 16><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    } else if (head_dim == 64) {
      quant_q_nhd_per_warp_kernel<__half, 64><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    } else {
      quant_q_nhd_per_warp_kernel<__half, 128><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    }
  } else {
    if (head_dim == 16) {
      quant_q_nhd_per_warp_kernel<__hip_bfloat16, 16><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    } else if (head_dim == 64) {
      quant_q_nhd_per_warp_kernel<__hip_bfloat16, 64><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    } else {
      quant_q_nhd_per_warp_kernel<__hip_bfloat16, 128><<<grid, block, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
          query_out.data_ptr<int8_t>(), query_scale.data_ptr<float>(),
          batch, q_len, q_heads, q_scale_groups);
    }
  }
  hip_kernel_launch_check();
  return {query_out, query_scale};
}

#endif // SAGEATTN_GFX12_BUILD_AUX

#if SAGEATTN_GFX12_BUILD_PREPARE

std::vector<torch::Tensor> quant_qk_int8_hnd_gfx12(torch::Tensor query, torch::Tensor key) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(), "gfx12 Q/K quantization expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "gfx12 Q/K quantization expects [B, H, S, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(),
              "gfx12 Q/K quantization expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(),
              "gfx12 Q/K quantization expects matching Q/K dtypes");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "gfx12 Q/K quantization supports fp16/bf16 input");
  TORCH_CHECK(query.size(0) == key.size(0), "Q/K batch size mismatch");
  TORCH_CHECK(query.size(3) == key.size(3), "Q/K head_dim mismatch");
  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t kv_len = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "gfx12 native Q/K quantization supports head_dim 16, 64, or 128");
  TORCH_CHECK((q_len % 64) == 0 && (kv_len % 64) == 0,
              "gfx12 native Q/K quantization requires sequence lengths divisible by 64");

  const int q_groups = static_cast<int>((q_len + 31) / 32);
  const int k_groups = static_cast<int>((kv_len + 63) / 64);
  torch::Tensor query_out = torch::empty_like(query, query.options().dtype(torch::kInt8));
  torch::Tensor key_out = torch::empty_like(key, key.options().dtype(torch::kInt8));
  torch::Tensor query_scale = torch::empty({batch, q_heads, q_groups}, query.options().dtype(torch::kFloat32));
  torch::Tensor key_scale = torch::empty({batch, kv_heads, k_groups}, key.options().dtype(torch::kFloat32));

  const dim3 block(256);
  const dim3 grid(q_groups + k_groups, std::max(q_heads, kv_heads), batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query.scalar_type() == torch::kFloat16) {
    if (head_dim == 16) {
      quant_qk_int8_hnd_kernel<__half, 16><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    } else if (head_dim == 64) {
      quant_qk_int8_hnd_kernel<__half, 64><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    } else {
      quant_qk_int8_hnd_kernel<__half, 128><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                         reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    }
  } else {
    if (head_dim == 16) {
      quant_qk_int8_hnd_kernel<__hip_bfloat16, 16><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    } else if (head_dim == 64) {
      quant_qk_int8_hnd_kernel<__hip_bfloat16, 64><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    } else {
      quant_qk_int8_hnd_kernel<__hip_bfloat16, 128><<<grid, block, 0, stream>>>(
                         reinterpret_cast<const __hip_bfloat16*>(query.data_ptr<at::BFloat16>()),
                         reinterpret_cast<const __hip_bfloat16*>(key.data_ptr<at::BFloat16>()),
                         query_out.data_ptr<int8_t>(), key_out.data_ptr<int8_t>(),
                         query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                         batch, q_heads, kv_heads, q_len, kv_len, q_groups, k_groups);
    }
  }
  hip_kernel_launch_check();
  return {query_out, query_scale, key_out, key_scale};
}

torch::Tensor qk_int8_sv_f16_d64_native_attn_gfx12_dispatch(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    torch::Tensor value_scale,
    int value_transposed_hnd_hint,
    int pv_accum_mode);

torch::Tensor qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int is_causal,
    int value_is_fp8,
    int use_raw_f16_value,
    float sm_scale,
    int64_t valid_kv_len,
    int pv_accum_mode) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "native gfx12 prepare+attention expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "native gfx12 prepare+attention expects HND [B, H, S, D] tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "native gfx12 prepare+attention expects contiguous HND tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
              "native gfx12 prepare+attention expects matching input dtypes");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
              "native gfx12 prepare+attention supports fp16/bf16 input");

  const int64_t head_dim = query.size(3);
  TORCH_CHECK(value_is_fp8 || head_dim == 16 || head_dim == 64 || head_dim == 128,
              "native gfx12 fp16 value prepare+attention supports head_dim 16, 64, or 128");
  TORCH_CHECK(value_is_fp8 || !use_raw_f16_value || query.scalar_type() == torch::kFloat16,
              "raw fp16 value path requires fp16 input");
  const int64_t batch = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t q_len = query.size(2);
  const int64_t kv_heads = key.size(1);
  const int64_t padded_kv_len = key.size(2);
  const int64_t kv_len = valid_kv_len > 0 ? valid_kv_len : padded_kv_len;
  TORCH_CHECK(kv_len > 0 && kv_len <= padded_kv_len,
              "valid_kv_len must be in (0, padded_kv_len]");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch,
              "Q/K/V batch size mismatch");
  TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim,
              "Q/K/V head_dim mismatch");
  TORCH_CHECK(value.size(2) == padded_kv_len && value.size(1) == kv_heads,
              "K/V shape mismatch");
  TORCH_CHECK((q_heads % kv_heads) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK((q_len % 64) == 0 && (padded_kv_len % 64) == 0,
              "native gfx12 prepare+attention requires sequence lengths divisible by 64");
  TORCH_CHECK(!is_causal || q_len == padded_kv_len,
              "native gfx12 causal prepare+attention requires q_len == kv_len");
  TORCH_CHECK(pv_accum_mode >= -1 && pv_accum_mode <= 1,
              "invalid gfx12 fp16 PV accumulation mode");

  const auto output_dtype =
      (value_is_fp8 && query.scalar_type() == torch::kBFloat16) ? torch::kBFloat16 : torch::kFloat16;
  torch::Tensor output;
  if (!value_is_fp8) {
    output = torch::empty_like(query, query.options().dtype(output_dtype));
  }
  const bool force_fp32_pv_accum = !value_is_fp8 && pv_accum_mode == 0;
  const bool prefer_prepared_f16_causal =
      !force_fp32_pv_accum && !value_is_fp8 && head_dim == 64 && is_causal &&
      query.scalar_type() == torch::kFloat16 && q_len >= 4096;
  const bool auto_f16_fused_q =
      !force_fp32_pv_accum && !value_is_fp8 && (head_dim == 16 || head_dim == 64) &&
      query.scalar_type() == torch::kFloat16 &&
      (is_causal || q_len >= 2048 || (head_dim == 64 && q_len >= 1024)) &&
      q_len <= 8192 &&
      !prefer_prepared_f16_causal;
  const bool auto_f16_raw_qk =
      !force_fp32_pv_accum && !value_is_fp8 && is_causal && head_dim == 16 &&
      query.scalar_type() == torch::kFloat16 && q_len <= 2048;
  if (!value_is_fp8 && is_causal && (head_dim == 16 || head_dim == 64) &&
      query.scalar_type() == torch::kFloat16 &&
      auto_f16_raw_qk) {
    int block_rows = q_len <= 64 ? 64 : 128;
    TORCH_CHECK((q_len % block_rows) == 0,
                "native raw-QK fp16 path requires q_len to be a multiple of block rows");
    const dim3 block(block_rows);
    const dim3 grid((q_len + block_rows - 1) / block_rows, q_heads, batch);
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool use_f16_pv_accum =
        auto_f16_raw_qk && q_len <= 1024;
    constexpr bool use_f16_tvload = false;
#define SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL(BR_, TVLOAD_, PAD_, F16ACC_) \
    if (head_dim == 16) { \
      qk_int8_sv_f16_d64_native_2q_kernel<64, true, BR_, false, PAD_, true, TVLOAD_, F16ACC_, __half, true, __half, true, false, false, false, false, 16><<<grid, block, 0, stream>>>( \
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), \
          reinterpret_cast<const __half*>(key.data_ptr<at::Half>()), \
          reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
          nullptr, nullptr, \
          batch, q_len, kv_len, q_heads, kv_heads, \
          query.stride(0), query.stride(2), query.stride(1), \
          key.stride(0), key.stride(2), key.stride(1), \
          value.stride(0), value.stride(2), value.stride(1), \
          output.stride(0), output.stride(2), output.stride(1), \
          0, 0, 0, 0, \
          kHND, sm_scale); \
    } else { \
      qk_int8_sv_f16_d64_native_2q_kernel<64, true, BR_, false, PAD_, true, TVLOAD_, F16ACC_, __half, true, __half, true><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), \
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        nullptr, nullptr, \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        key.stride(0), key.stride(2), key.stride(1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, 0, 0, \
        kHND, sm_scale); \
    }
#define SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(BR_) \
    if (use_f16_tvload) { \
      if (use_f16_pv_accum) { \
        SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL(BR_, true, 4, true); \
      } else { \
        SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL(BR_, true, 4, false); \
      } \
    } else if (use_f16_pv_accum) { \
      SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL(BR_, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD, true); \
    } else { \
      SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL(BR_, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD, false); \
    }
    if (block_rows == 64) {
      SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(64);
    } else if (block_rows == 256) {
      SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(256);
    } else if (block_rows == 512) {
      SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(512);
    } else if (block_rows == 1024) {
      SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(1024);
    } else {
      SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL(128);
    }
#undef SAGEATTN_DISPATCH_RAW_QK_F16_CAUSAL
#undef SAGEATTN_LAUNCH_RAW_QK_F16_CAUSAL
    hip_kernel_launch_check();
    return output;
  }
  if (value_is_fp8) {
    int block_rows = head_dim == 64 ?
        select_fp8_d64_block_rows_gfx12(q_len, is_causal, true) :
        (q_len <= 64 ? 64 : 128);
    int block_cols = 64;
    if (head_dim == 64 && !is_causal && q_len == 1024) {
      block_rows = 128;
    }
    if (head_dim == 16 && is_causal && q_len <= 1024) {
      block_rows = 64;
    }
    const bool transpose_fp8_value = true;
    const bool use_fused_q =
        (head_dim == 16 || head_dim == 64 || head_dim == 128) && transpose_fp8_value &&
        query.scalar_type() == torch::kFloat16 && output_dtype == torch::kFloat16 &&
         ((head_dim == 16 && block_cols == 64 && is_causal && q_len == 1024) ||
          (head_dim == 64 && block_cols == 64 && is_causal &&
            q_len >= 1024) ||
          (head_dim == 128 && block_cols <= 64 && is_causal && q_len >= 1024) ||
          (head_dim == 128 && block_cols <= 64 && !is_causal && q_len <= 512));
    const bool use_fp8_kvlane =
        use_fused_q && head_dim == 64 && block_cols == 64 && transpose_fp8_value &&
        true;
    const bool auto_fp8_streamcols4 =
        use_fused_q && head_dim == 64 && is_causal && block_cols == 64;
    const bool use_fp8_streamcols4 =
        use_fused_q && block_cols == 64 && auto_fp8_streamcols4;
    const dim3 block(block_rows);
    const dim3 grid((q_len + block_rows - 1) / block_rows, q_heads, batch);
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
    std::vector<torch::Tensor> prepared;
    int8_t* fused_key_ptr = nullptr;
    uint8_t* fused_value_ptr = nullptr;
    float* fused_k_scale_ptr = nullptr;
    int64_t fused_k_stride_b = 0;
    int64_t fused_k_stride_n = 0;
    int64_t fused_k_stride_h = 0;
    int64_t fused_v_stride_b = 0;
    int64_t fused_v_stride_n = 0;
    int64_t fused_v_stride_h = 0;
    int64_t fused_ks_stride_b = 0;
    int64_t fused_ks_stride_h = 0;
    if (head_dim == 16 && !use_fused_q && transpose_fp8_value) {
      prepared = prepare_qkv_hnd_gfx12<uint8_t, true>(query, key, value);
    } else if (use_fused_q) {
      if (use_fp8_kvlane) {
        prepared = prepare_kv_hnd_packed_gfx12<uint8_t, true, true, false, false, true>(
            query, key, value);
      } else {
        prepared = prepare_kv_hnd_packed_gfx12<uint8_t, true, true>(query, key, value);
      }
    } else if (head_dim == 128 && is_causal && q_len == 1024 && transpose_fp8_value) {
      prepared = prepare_qkv_hnd_gfx12<uint8_t, true>(query, key, value);
    } else if (transpose_fp8_value) {
      prepared = prepare_qkv_hnd_packed_gfx12<uint8_t, true, true>(query, key, value);
    } else {
      prepared = prepare_qkv_hnd_packed_gfx12<uint8_t, true, false>(query, key, value);
    }
    output = torch::empty_like(query, query.options().dtype(output_dtype));
    if (use_fused_q) {
      fused_key_ptr = prepared[0].data_ptr<int8_t>();
      fused_value_ptr = prepared[2].data_ptr<uint8_t>();
      fused_k_scale_ptr = prepared[1].data_ptr<float>();
      fused_k_stride_b = prepared[0].stride(0);
      fused_k_stride_n = prepared[0].stride(2);
      fused_k_stride_h = prepared[0].stride(1);
      fused_v_stride_b = prepared[2].stride(0);
      fused_v_stride_n = prepared[2].stride(2);
      fused_v_stride_h = prepared[2].stride(1);
      fused_ks_stride_b = prepared[1].stride(0);
      fused_ks_stride_h = prepared[1].stride(1);
    }
#define SAGEATTN_LAUNCH_PREPARED_FP8_EX(BC_, HD_, BR_, VT_, CAUSAL_, OUT_T_, LOWP_) \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, 0, ((HD_) / 16), true, BR_, VT_, CAUSAL_, OUT_T_, int8_t, false, int8_t, uint8_t, false, false, 0, false, false, 2, LOWP_><<<grid, block, 0, stream>>>( \
        prepared[0].data_ptr<int8_t>(), prepared[2].data_ptr<int8_t>(), \
        prepared[4].data_ptr<uint8_t>(), \
        reinterpret_cast<OUT_T_*>(output.data_ptr()), \
        prepared[1].data_ptr<float>(), prepared[3].data_ptr<float>(), nullptr, \
        batch, q_len, kv_len, q_heads, kv_heads, \
        prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
        prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
        prepared[4].stride(0), prepared[4].stride(2), prepared[4].stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        prepared[1].stride(0), prepared[1].stride(1), \
        prepared[3].stride(0), prepared[3].stride(1), \
        kHND, sm_scale)
#define SAGEATTN_LAUNCH_PREPARED_FP8(BC_, HD_, BR_, VT_, CAUSAL_, OUT_T_) \
    SAGEATTN_LAUNCH_PREPARED_FP8_EX(BC_, HD_, BR_, VT_, CAUSAL_, OUT_T_, false)

#define SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SLICE(BC_, HD_, BR_, CAUSAL_, KVLANE_, SC_, KLANE_, VLANE_, VBASE_, VTILES_) \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, VBASE_, VTILES_, true, BR_, true, CAUSAL_, __half, __half, true, int8_t, uint8_t, false, KVLANE_, SC_, KLANE_, VLANE_><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), fused_key_ptr, \
        fused_value_ptr, \
        reinterpret_cast<__half*>(output.data_ptr()), \
        nullptr, fused_k_scale_ptr, nullptr, \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        fused_k_stride_b, fused_k_stride_n, fused_k_stride_h, \
        fused_v_stride_b, fused_v_stride_n, fused_v_stride_h, \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        fused_ks_stride_b, fused_ks_stride_h, \
        kHND, sm_scale)
#define SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SC(BC_, HD_, BR_, CAUSAL_, KVLANE_, SC_) \
    SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SLICE(BC_, HD_, BR_, CAUSAL_, KVLANE_, SC_, false, false, 0, ((HD_) / 16))
#define SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL(BC_, HD_, BR_, CAUSAL_, KVLANE_) \
    if constexpr ((BC_) == 64) { \
      if (use_fp8_streamcols4) { \
        SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SC(BC_, HD_, BR_, CAUSAL_, KVLANE_, 4); \
      } else { \
        SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SC(BC_, HD_, BR_, CAUSAL_, KVLANE_, 0); \
      } \
    } else { \
      SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SC(BC_, HD_, BR_, CAUSAL_, KVLANE_, 0); \
    }
#define SAGEATTN_LAUNCH_FUSED_Q_FP8(BC_, HD_, BR_, CAUSAL_) \
    if (use_fp8_kvlane) { \
      SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL(BC_, HD_, BR_, CAUSAL_, true); \
    } else { \
      SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL(BC_, HD_, BR_, CAUSAL_, false); \
    }

#define SAGEATTN_DISPATCH_PREPARED_FP8_VT_BC(BC_, VT_, OUT_T_) \
    if (head_dim == 16) { \
      if (block_rows == 64) { \
        if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 64, VT_, true, OUT_T_); } \
        else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 64, VT_, false, OUT_T_); } \
      } else if (block_rows == 256) { \
        if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 256, VT_, true, OUT_T_); } \
        else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 256, VT_, false, OUT_T_); } \
      } else { \
        if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 128, VT_, true, OUT_T_); } \
        else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 16, 128, VT_, false, OUT_T_); } \
      } \
    } else if (head_dim == 128) { \
      if (block_rows == 64) { \
        if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 128, 64, true, true, OUT_T_); } \
        else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 128, 64, true, false, OUT_T_); } \
      } else { \
        if (is_causal) { \
          if constexpr ((BC_) == 64) { \
            if (q_len == 1024) { \
              SAGEATTN_LAUNCH_PREPARED_FP8_EX(BC_, 128, 128, true, true, OUT_T_, true); \
            } else { \
              SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 128, 128, true, true, OUT_T_); \
            } \
          } else { \
            SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 128, 128, true, true, OUT_T_); \
          } \
        } \
        else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 128, 128, true, false, OUT_T_); } \
      } \
    } else if (block_rows == 256) { \
      if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 256, VT_, true, OUT_T_); } \
      else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 256, VT_, false, OUT_T_); } \
    } else if (block_rows == 64) { \
      if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 64, VT_, true, OUT_T_); } \
      else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 64, VT_, false, OUT_T_); } \
    } else { \
      if (is_causal) { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 128, VT_, true, OUT_T_); } \
      else { SAGEATTN_LAUNCH_PREPARED_FP8(BC_, 64, 128, VT_, false, OUT_T_); } \
    }

#define SAGEATTN_DISPATCH_PREPARED_FP8_TV(OUT_T_) \
    SAGEATTN_DISPATCH_PREPARED_FP8_VT_BC(64, true, OUT_T_)

    if (use_fused_q) {
#define SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD_CAUSAL(BC_, HD_, CAUSAL_) \
      if (block_rows == 64) { SAGEATTN_LAUNCH_FUSED_Q_FP8(BC_, HD_, 64, CAUSAL_); } \
      else if (block_rows == 256) { SAGEATTN_LAUNCH_FUSED_Q_FP8(BC_, HD_, 256, CAUSAL_); } \
      else { SAGEATTN_LAUNCH_FUSED_Q_FP8(BC_, HD_, 128, CAUSAL_); }
#define SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD(BC_, HD_) \
      if (is_causal) { SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD_CAUSAL(BC_, HD_, true); } \
      else { SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD_CAUSAL(BC_, HD_, false); }
      if (head_dim == 16) {
        SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD(64, 16);
      } else if (head_dim == 128) {
        SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD(64, 128);
      } else {
        SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD(64, 64);
      }
#undef SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD
#undef SAGEATTN_DISPATCH_FUSED_Q_FP8_BC_HD_CAUSAL
    } else if (output.scalar_type() == torch::kBFloat16) {
      SAGEATTN_DISPATCH_PREPARED_FP8_TV(__hip_bfloat16);
    } else {
      SAGEATTN_DISPATCH_PREPARED_FP8_TV(__half);
    }
#undef SAGEATTN_DISPATCH_PREPARED_FP8_TV
#undef SAGEATTN_DISPATCH_PREPARED_FP8_VT_BC
#undef SAGEATTN_LAUNCH_FUSED_Q_FP8
#undef SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL
#undef SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SC
#undef SAGEATTN_LAUNCH_FUSED_Q_FP8_IMPL_SLICE
#undef SAGEATTN_LAUNCH_PREPARED_FP8
#undef SAGEATTN_LAUNCH_PREPARED_FP8_EX
  } else if (auto_f16_fused_q) {
    const bool auto_f16_1q_short =
        head_dim == 64 && is_causal && q_len == 1024;
    const bool use_f16_fused_q_1q_tv =
        head_dim == 64 && auto_f16_1q_short;
    const bool use_f16_fused_q_1q =
        head_dim == 64 && auto_f16_1q_short;
    const bool use_f16_raw_value =
        head_dim == 64 &&
        (use_raw_f16_value ||
         (use_f16_fused_q_1q && !use_f16_fused_q_1q_tv));
    int block_rows = q_len <= 64 ? 64 : 128;
    if ((q_len % 256) == 0) {
      if ((!is_causal && q_len >= 1024) || q_len <= 512 ||
          q_len >= 8192) {
        block_rows = 256;
      }
    }
    if (head_dim == 64 && is_causal && !use_f16_fused_q_1q &&
        q_len >= 2048 && (q_len % 256) == 0) {
      block_rows = 256;
    }
    int block_cols = 64;
    if (use_f16_fused_q_1q && block_rows != 128) {
      block_rows = 64;
    }
    const dim3 block(use_f16_fused_q_1q ? (block_rows / 16) * 32 : block_rows);
    const bool use_f16_flat_q_schedule =
        head_dim == 64 && is_causal && !use_f16_fused_q_1q &&
        q_len >= 2048;
    const int64_t q_blocks = (q_len + block_rows - 1) / block_rows;
    const dim3 grid(q_blocks, q_heads, batch);
    const dim3 grid_f16_flat(q_blocks * q_heads * batch);
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr bool use_f16_pv_accum = true;
    const bool use_f16_pv_ordered_qk =
        use_f16_pv_accum && !use_f16_raw_value &&
        q_len >= 1024;
    const bool auto_f16_lane_qk =
        head_dim == 64 && is_causal && (q_len == 2048 || q_len == 4096);
    const bool use_f16_vlane =
        (((q_len >= 1024 && q_len <= 2048) || q_len >= 8192) ||
         auto_f16_lane_qk);
    const bool use_f16_streamk =
        head_dim == 64 && is_causal && q_len == 2048 && block_rows == 256;
    const bool use_f16_klane =
        !use_f16_raw_value && block_cols == 64 &&
        use_f16_pv_ordered_qk &&
        auto_f16_lane_qk;
    std::vector<torch::Tensor> prepared = use_f16_raw_value ?
        prepare_k_hnd_packed_gfx12(key) :
        prepare_kv_hnd_packed_gfx12<__half, false>(query, key, value);
#define SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_) \
    if (head_dim == 16) { \
      qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, true, false, F16ACC_, __half, true, int8_t, false, PVORDER_, VLANE_, STREAM_, KLANE_, 16><<<grid, block, 0, stream>>>( \
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
          reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
          nullptr, prepared[1].data_ptr<float>(), \
          batch, q_len, kv_len, q_heads, kv_heads, \
          query.stride(0), query.stride(2), query.stride(1), \
          prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
          prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
          output.stride(0), output.stride(2), output.stride(1), \
          0, 0, \
          prepared[1].stride(0), prepared[1].stride(1), \
          kHND, sm_scale); \
    } else { \
      if (use_f16_flat_q_schedule) { \
        qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, true, false, F16ACC_, __half, true, int8_t, false, PVORDER_, VLANE_, STREAM_, KLANE_, 64, true><<<grid_f16_flat, block, 0, stream>>>( \
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
          reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
          nullptr, prepared[1].data_ptr<float>(), \
          batch, q_len, kv_len, q_heads, kv_heads, \
          query.stride(0), query.stride(2), query.stride(1), \
          prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
          prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
          output.stride(0), output.stride(2), output.stride(1), \
          0, 0, \
          prepared[1].stride(0), prepared[1].stride(1), \
          kHND, sm_scale); \
      } else { \
        qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, true, false, F16ACC_, __half, true, int8_t, false, PVORDER_, VLANE_, STREAM_, KLANE_><<<grid, block, 0, stream>>>( \
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
          reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
          nullptr, prepared[1].data_ptr<float>(), \
          batch, q_len, kv_len, q_heads, kv_heads, \
          query.stride(0), query.stride(2), query.stride(1), \
          prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
          prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
          output.stride(0), output.stride(2), output.stride(1), \
          0, 0, \
          prepared[1].stride(0), prepared[1].stride(1), \
          kHND, sm_scale); \
      } \
    }
#define SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(BR_, PAD_, F16ACC_, PVORDER_, VLANE_, KLANE_) \
    if (head_dim == 16) { \
      qk_int8_sv_f16_d64_native_2q_kernel<64, true, BR_, true, PAD_, false, false, F16ACC_, __half, true, int8_t, false, PVORDER_, false, false, false, 16><<<grid, block, 0, stream>>>( \
          reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
          reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
          nullptr, prepared[1].data_ptr<float>(), \
          batch, q_len, kv_len, q_heads, kv_heads, \
          query.stride(0), query.stride(2), query.stride(1), \
          prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
          prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
          output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        prepared[1].stride(0), prepared[1].stride(1), \
        kHND, sm_scale); \
    } else { \
      qk_int8_sv_f16_d64_native_2q_kernel<64, true, BR_, true, PAD_, false, false, F16ACC_, __half, true, int8_t, false, PVORDER_, VLANE_, false, KLANE_><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        nullptr, prepared[1].data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
        prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        prepared[1].stride(0), prepared[1].stride(1), \
        kHND, sm_scale); \
    }
#define SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL(BC_, BR_, PAD_, F16ACC_, VLANE_, STREAM_) \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, false, PAD_, true, true, F16ACC_, __half, true, int8_t, false, false, VLANE_, STREAM_><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        nullptr, prepared[1].data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        prepared[1].stride(0), prepared[1].stride(1), \
        kHND, sm_scale)
#define SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_RAWV_CAUSAL(BR_, F16ACC_, SPLIT_) \
    qk_int8_sv_f16_d64_native_kernel<64, BR_, true, false, 4, true, true, F16ACC_, true, __half, true, SPLIT_><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        nullptr, prepared[1].data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        prepared[1].stride(0), prepared[1].stride(1), \
        kHND, sm_scale)
#define SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_TV_CAUSAL(BR_, F16ACC_) \
    qk_int8_sv_f16_d64_native_kernel<64, BR_, true, true, 4, true, false, F16ACC_, true, __half, true><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()), prepared[0].data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(prepared[2].data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        nullptr, prepared[1].data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(2), query.stride(1), \
        prepared[0].stride(0), prepared[0].stride(2), prepared[0].stride(1), \
        prepared[2].stride(0), prepared[2].stride(2), prepared[2].stride(1), \
        output.stride(0), output.stride(2), output.stride(1), \
        0, 0, \
        prepared[1].stride(0), prepared[1].stride(1), \
        kHND, sm_scale)
#define SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(BR_, PAD_) \
    if (use_f16_raw_value) { \
      if (use_f16_vlane && use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL(64, BR_, PAD_, true, true, true); } \
      else if (use_f16_vlane) { SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL(64, BR_, PAD_, true, true, false); } \
      else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL(64, BR_, PAD_, true, false, true); } \
      else { SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL(64, BR_, PAD_, true, false, false); } \
    } else if (use_f16_pv_ordered_qk) { \
      if (use_f16_klane && use_f16_vlane && use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, true, true, true); } \
      else if (use_f16_klane && use_f16_vlane) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, true, false, true); } \
      else if (use_f16_klane && use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, false, true, true); } \
      else if (use_f16_klane) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, false, false, true); } \
      else if (use_f16_vlane && use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, true, true, false); } \
      else if (use_f16_vlane) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, true, false, false); } \
      else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, false, true, false); } \
      else { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, true, false, false, false); } \
    } else { \
      if (use_f16_vlane && use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, false, true, true, false); } \
      else if (use_f16_vlane) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, false, true, false, false); } \
      else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, false, false, true, false); } \
      else { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL(64, BR_, PAD_, true, false, false, false, false); } \
    }
    if (!is_causal) {
      TORCH_CHECK(block_cols == 64,
                  "native fp16 fused-Q non-causal path currently supports BC64");
      TORCH_CHECK(!use_f16_raw_value,
                  "native fp16 fused-Q non-causal path requires transposed prepared values");
      if (use_f16_pv_ordered_qk) {
        if (block_rows == 64) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(64, 4, true, true, false, false); }
        else if (block_rows == 256) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(256, 4, true, true, false, false); }
        else { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(128, 4, true, true, false, false); }
      } else {
        if (block_rows == 64) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(64, 4, true, false, false, false); }
        else if (block_rows == 256) { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(256, 4, true, false, false, false); }
        else { SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL(128, 4, true, false, false, false); }
      }
    } else if (use_f16_fused_q_1q) {
      if (use_f16_fused_q_1q_tv && block_rows == 128) {
        SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_TV_CAUSAL(128, true);
      } else if (use_f16_fused_q_1q_tv) {
        SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_TV_CAUSAL(64, true);
      } else if (block_rows == 128) {
        SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_RAWV_CAUSAL(128, true, false);
      } else {
        SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_RAWV_CAUSAL(64, true, false);
      }
    } else if (block_rows == 64) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(64, 4);
    } else if (block_rows == 256) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(256, 4);
    } else if (block_rows == 512) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(512, 4);
    } else if (block_rows == 1024) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(1024, 4);
    } else if (q_len >= 8192) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(128, 8);
    } else if (q_len >= 1024) {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(128, 4);
    } else {
      SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL(128, 16);
    }
#undef SAGEATTN_DISPATCH_F16_FUSED_Q_TV_CAUSAL
#undef SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_TV_CAUSAL
#undef SAGEATTN_LAUNCH_F16_FUSED_Q_1Q_RAWV_CAUSAL
#undef SAGEATTN_LAUNCH_F16_FUSED_Q_RAWV_CAUSAL
#undef SAGEATTN_LAUNCH_F16_FUSED_Q_TV_NONCAUSAL
#undef SAGEATTN_LAUNCH_F16_FUSED_Q_TV_CAUSAL
    hip_kernel_launch_check();
  } else if (use_raw_f16_value) {
    std::vector<torch::Tensor> prepared = quant_qk_int8_hnd_gfx12(query, key);
    qk_int8_sv_f16_d64_native_attn_gfx12_dispatch(
        prepared[0], prepared[2], value, output, prepared[1], prepared[3],
        kHND, is_causal, sm_scale, kv_len, torch::Tensor(), 0,
        pv_accum_mode);
  } else {
    const bool use_f16_separate_prepared =
        is_causal && head_dim == 64 && q_len == 4096 &&
        query.scalar_type() == torch::kFloat16;
    std::vector<torch::Tensor> prepared =
        use_f16_separate_prepared ?
             prepare_qkv_hnd_gfx12<__half, false>(query, key, value) :
             prepare_qkv_hnd_packed_gfx12<__half, false>(query, key, value);
    qk_int8_sv_f16_d64_native_attn_gfx12_dispatch(
        prepared[0], prepared[2], prepared[4], output, prepared[1], prepared[3],
        kHND, is_causal, sm_scale, kv_len, torch::Tensor(), 1,
        pv_accum_mode);
  }
  return output;
}

#endif // SAGEATTN_GFX12_BUILD_PREPARE

#if SAGEATTN_GFX12_BUILD_ATTN_F16 || SAGEATTN_GFX12_BUILD_ATTN_FP8

template <bool PerThreadQK = false>
static torch::Tensor qk_int8_sv_f16_d64_native_attn_gfx12_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    torch::Tensor value_scale,
    int value_transposed_hnd_hint,
    int pv_accum_mode) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && output.is_cuda(),
              "native gfx12 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(query.scalar_type() == torch::kInt8, "query must be int8");
  TORCH_CHECK(key.scalar_type() == torch::kInt8, "key must be int8");
  const bool value_is_fp8 = value.scalar_type() == torch::kByte;
#if SAGEATTN_GFX12_BUILD_ATTN_F16 && !SAGEATTN_GFX12_BUILD_ATTN_FP8
  TORCH_CHECK(!value_is_fp8, "native gfx12 fp16 attention TU expects fp16 values");
#endif
#if SAGEATTN_GFX12_BUILD_ATTN_FP8 && !SAGEATTN_GFX12_BUILD_ATTN_F16
  TORCH_CHECK(value_is_fp8, "native gfx12 fp8 attention TU expects fp8 values");
#endif
  TORCH_CHECK(value.scalar_type() == torch::kFloat16 || value_is_fp8,
              "value must be fp16 or raw OCP e4m3 fp8 bytes");
  const bool output_is_bf16 = output.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 || (value_is_fp8 && output_is_bf16),
              "output must be fp16, or bf16 for the fp8 value path");
  TORCH_CHECK(query_scale.scalar_type() == torch::kFloat32, "query_scale must be fp32");
  TORCH_CHECK(key_scale.scalar_type() == torch::kFloat32, "key_scale must be fp32");
  TORCH_CHECK(tensor_layout == kHND || tensor_layout == kNHD, "invalid tensor_layout");
  const int64_t head_dim = query.size(-1);
  const bool value_maybe_transposed_hnd =
      tensor_layout == kHND && value.dim() == 4 && value.size(2) == head_dim &&
      (value_is_fp8 || value_transposed_hnd_hint > 0 ||
       (value_transposed_hnd_hint < 0 && (value_is_fp8 || head_dim != 128)));
  TORCH_CHECK(key.size(-1) == head_dim &&
                  (value.size(-1) == head_dim || value_maybe_transposed_hnd),
              "query, key, and value must have matching head_dim");
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128,
              "native gfx12 path supports D16/D64/D128");
  TORCH_CHECK(value_is_fp8 || head_dim != 128 || value_maybe_transposed_hnd,
              "native gfx12 fp16 D128 path requires transposed HND values");

  const int64_t batch = query.size(0);
  const int64_t q_heads = tensor_layout == kNHD ? query.size(2) : query.size(1);
  const int64_t q_len = tensor_layout == kNHD ? query.size(1) : query.size(2);
  const int64_t kv_heads = tensor_layout == kNHD ? key.size(2) : key.size(1);
  const int64_t padded_kv_len = tensor_layout == kNHD ? key.size(1) : key.size(2);
  const int64_t kv_len = valid_kv_len > 0 ? valid_kv_len : padded_kv_len;
  TORCH_CHECK(kv_len > 0 && kv_len <= padded_kv_len,
              "valid_kv_len must be in (0, padded_kv_len]");
  const bool value_transposed_hnd = value_maybe_transposed_hnd && value.size(3) >= padded_kv_len;
  TORCH_CHECK(!value_maybe_transposed_hnd || value_transposed_hnd,
              "transposed HND value must have shape [B, H, D, padded_kv_len]");
  TORCH_CHECK(!value_transposed_hnd || value.is_contiguous(),
              "transposed HND value must be contiguous");
  TORCH_CHECK((q_len % 64) == 0 && (padded_kv_len % 64) == 0,
              "native gfx12 path requires q_len and kv_len multiples of 64");
  TORCH_CHECK(!is_causal || q_len == padded_kv_len,
              "native gfx12 causal path currently requires q_len == kv_len");
  TORCH_CHECK((q_heads % kv_heads) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(pv_accum_mode >= -1 && pv_accum_mode <= 1,
              "invalid gfx12 fp16 PV accumulation mode");
  TORCH_CHECK(query_scale.stride(-1) == 1 && key_scale.stride(-1) == 1,
              "scale tensors must have contiguous scale columns");
  const int64_t per_warp_q_groups = ((q_len + 127) / 128) * 4;
  const int64_t per_thread_q_groups_warp32 = ((q_len + 127) / 128) * 32;
  const int64_t per_thread_q_groups_warp16 = ((q_len + 127) / 128) * 64;
  const int64_t per_warp_k_groups = (padded_kv_len + 63) / 64;
  const int64_t per_thread_k_groups = ((padded_kv_len + 63) / 64) * 4;
  const bool use_per_thread_qk =
      query_scale.size(2) == per_thread_q_groups_warp32 ||
      query_scale.size(2) == per_thread_q_groups_warp16 ||
      key_scale.size(2) == per_thread_k_groups;
  TORCH_CHECK((query_scale.size(2) == per_warp_q_groups &&
               key_scale.size(2) == per_warp_k_groups) ||
              ((query_scale.size(2) == per_thread_q_groups_warp32 ||
                query_scale.size(2) == per_thread_q_groups_warp16) &&
               key_scale.size(2) == per_thread_k_groups),
              "gfx12 query/key scale shapes must both be per-warp or both be per-thread");
  if constexpr (!PerThreadQK) {
    if (use_per_thread_qk) {
      return qk_int8_sv_f16_d64_native_attn_gfx12_impl<true>(
          query, key, value, output, query_scale, key_scale, tensor_layout,
          is_causal, sm_scale, valid_kv_len, value_scale,
          value_transposed_hnd_hint, pv_accum_mode);
    }
  }
  const bool has_value_scale = value_scale.defined() && value_scale.numel() > 0;
  TORCH_CHECK(!has_value_scale || value_is_fp8,
              "value_scale is only valid for the fp8 value path");
  if (has_value_scale) {
    TORCH_CHECK(value_scale.is_cuda(), "value_scale must be a CUDA/HIP tensor");
    TORCH_CHECK(value_scale.scalar_type() == torch::kFloat32,
                "value_scale must be fp32");
    TORCH_CHECK(value_scale.dim() == 3 && value_scale.is_contiguous(),
                "value_scale must be contiguous [B, H_kv, D]");
    TORCH_CHECK(value_scale.size(0) == batch &&
                    value_scale.size(1) == kv_heads &&
                    value_scale.size(2) == head_dim,
                "value_scale shape must match [B, H_kv, D]");
  }
  const float* value_scale_ptr = has_value_scale ? value_scale.data_ptr<float>() : nullptr;
  const bool hnd_contiguous = tensor_layout == kHND &&
      query.is_contiguous() && key.is_contiguous() &&
      value.is_contiguous() && output.is_contiguous();

  int block_cols = 64;
  bool use_2q = !value_is_fp8;
  bool use_f16_causal_1q = false;
  bool use_fp8_2q = value_is_fp8;
  if (!value_is_fp8 && !value_transposed_hnd &&
      head_dim == 64 && q_len <= 1024) {
    use_2q = false;
  }
  if (is_causal) {
    if (value_is_fp8) {
      use_2q = false;
      use_f16_causal_1q = false;
      use_fp8_2q = true;
    } else {
      use_fp8_2q = false;
      use_f16_causal_1q = false;
      use_2q = true;
    }
  }
  TORCH_CHECK(!(value_transposed_hnd && !value_is_fp8 && !use_2q && !use_f16_causal_1q),
              "transposed fp16 value path currently requires tqk1/tqk2/auto mode");
  int block_rows = q_len <= 64 ? 64 : 128;
  if (use_2q) {
    block_rows = q_len <= 64 ? 64 : 128;
  } else if (use_fp8_2q && block_rows != 32 && block_rows != 64 &&
             block_rows != 256 && block_rows != 512) {
    block_rows = q_len <= 64 ? 64 : 128;
  }
  if (is_causal && head_dim == 16 && (use_2q || use_fp8_2q)) {
    block_rows = (use_fp8_2q && q_len <= 1024) ? 64 : 128;
  }
  if (use_f16_causal_1q) {
    block_cols = 64;
    block_rows = 64;
  }
  if (!is_causal && use_fp8_2q && head_dim == 64 &&
      (q_len >= 2048 || value_transposed_hnd) && (q_len % 256) == 0) {
    block_rows = select_fp8_d64_block_rows_gfx12(q_len, is_causal, value_transposed_hnd);
  }
  if (is_causal && use_fp8_2q && head_dim == 64 &&
      (q_len % 256) == 0) {
    block_rows = select_fp8_d64_block_rows_gfx12(q_len, is_causal, value_transposed_hnd);
  }
  if (is_causal && use_fp8_2q && head_dim == 128 &&
      value_transposed_hnd && q_len >= 2048 && q_len <= 4096) {
    block_cols = 32;
  }
  if (!is_causal && use_2q && value_transposed_hnd &&
      q_len >= 2048 && (q_len % 256) == 0) {
    block_rows = 256;
  }
  if (is_causal && use_2q && head_dim == 64 && value_transposed_hnd &&
      q_len >= 4096 && (q_len % 256) == 0) {
    block_rows = 256;
  }
  if constexpr (PerThreadQK) {
    block_cols = 64;
    block_rows = q_len <= 64 ? 64 : 128;
    use_2q = !value_is_fp8;
    use_fp8_2q = value_is_fp8;
    use_f16_causal_1q = false;
  }
  TORCH_CHECK(!(use_fp8_2q && block_rows == 64 && block_cols == 128),
              "native fp8 2q BR64 is currently specialized for BC32/BC64");
  TORCH_CHECK(!(use_fp8_2q && block_rows == 256 && block_cols != 64),
              "native fp8 2q BR256 is currently specialized for BC64");
  TORCH_CHECK(!(use_2q && value_transposed_hnd && block_cols != 64),
              "native fp16 transposed value 2q path currently supports BC64");
  TORCH_CHECK(!(use_2q && block_rows != 32 && block_rows != 64 &&
                    block_rows != 128 && block_rows != 256 && block_rows != 512 &&
                    block_rows != 1024),
              "native fp16 2q path currently supports BR32/BR64/BR128/BR256/BR512/BR1024");
  TORCH_CHECK(!(use_2q && !value_transposed_hnd && block_rows != 64 && block_rows != 128),
              "native fp16 non-transposed 2q path currently supports BR64/BR128");
  TORCH_CHECK(!(use_f16_causal_1q &&
                ((block_rows != 64 && block_rows != 128) || block_cols != 64)),
              "native fp16 single-q causal path currently supports BR64/BR128/BC64");
  TORCH_CHECK((q_len % block_rows) == 0,
              "native gfx12 path requires q_len to be a multiple of the selected block rows");

  const bool use_f16_flat_q_schedule =
      head_dim == 64 && !value_is_fp8 && is_causal && use_2q &&
      value_transposed_hnd && q_len >= 2048;
  const int64_t q_blocks = (q_len + block_rows - 1) / block_rows;
  dim3 block((use_2q || use_fp8_2q) ? block_rows : (block_rows == 128 ? 256 : 128));
  dim3 grid(q_blocks, q_heads, batch);
  dim3 grid_f16_flat(q_blocks * q_heads * batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_f16_tvload =
      !value_is_fp8 && is_causal && hnd_contiguous && !value_transposed_hnd &&
      q_len >= 1024;
  const bool use_f16_pv_accum =
      !value_is_fp8 && pv_accum_mode != 0 &&
      (pv_accum_mode == 1 ||
       (is_causal && value_transposed_hnd && block_cols == 64 && q_len >= 1024));
  const bool use_f16_pv_ordered_qk =
      use_f16_pv_accum && q_len >= 4096;
  const bool use_f16_vlane =
      head_dim == 64 && !value_is_fp8 && is_causal && value_transposed_hnd &&
      q_len == 4096 && block_rows == 256;
  const bool use_f16_streamk =
      head_dim == 64 && !value_is_fp8 && is_causal && value_transposed_hnd &&
      q_len == 4096 && block_rows == 256;
  const bool use_f16_d128_short_stream =
      head_dim == 128 && !value_is_fp8 && is_causal && block_cols == 64 &&
      block_rows == 128 && q_len <= 1024;
  if constexpr (PerThreadQK) {
    TORCH_CHECK(value_transposed_hnd,
                "gfx12 per-thread QK path expects transposed HND values");
#define SAGEATTN_LAUNCH_PERTHREAD_FP8_OUT(HD_, BR_, CAUSAL_, OUT_T_) \
    qk_int8_sv_f8_native_2q_kernel<64, HD_, 0, ((HD_) / 16), true, BR_, true, CAUSAL_, OUT_T_, int8_t, false, int8_t, uint8_t, false, false, 0, false, false, 2, false, true><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        value.data_ptr<uint8_t>(), \
        reinterpret_cast<OUT_T_*>(output.data_ptr()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr, \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, true)
#define SAGEATTN_LAUNCH_PERTHREAD_FP8(HD_, BR_, CAUSAL_) \
    if (output_is_bf16) { \
      SAGEATTN_LAUNCH_PERTHREAD_FP8_OUT(HD_, BR_, CAUSAL_, __hip_bfloat16); \
    } else { \
      SAGEATTN_LAUNCH_PERTHREAD_FP8_OUT(HD_, BR_, CAUSAL_, __half); \
    }
#define SAGEATTN_LAUNCH_PERTHREAD_F16(HD_, BR_, CAUSAL_) \
    qk_int8_sv_f16_d64_native_2q_kernel<64, true, BR_, true, 4, CAUSAL_, false, false, int8_t, false, int8_t, false, false, false, false, false, HD_, false, true><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, true)
#if SAGEATTN_GFX12_BUILD_ATTN_F16 && SAGEATTN_GFX12_BUILD_ATTN_FP8
#define SAGEATTN_DISPATCH_PERTHREAD_HEADS(BR_, CAUSAL_) \
    if (value_is_fp8) { \
      if (head_dim == 16) { SAGEATTN_LAUNCH_PERTHREAD_FP8(16, BR_, CAUSAL_); } \
      else if (head_dim == 64) { SAGEATTN_LAUNCH_PERTHREAD_FP8(64, BR_, CAUSAL_); } \
      else { SAGEATTN_LAUNCH_PERTHREAD_FP8(128, BR_, CAUSAL_); } \
    } else { \
      if (head_dim == 16) { SAGEATTN_LAUNCH_PERTHREAD_F16(16, BR_, CAUSAL_); } \
      else if (head_dim == 64) { SAGEATTN_LAUNCH_PERTHREAD_F16(64, BR_, CAUSAL_); } \
      else { SAGEATTN_LAUNCH_PERTHREAD_F16(128, BR_, CAUSAL_); } \
    }
#elif SAGEATTN_GFX12_BUILD_ATTN_FP8
#define SAGEATTN_DISPATCH_PERTHREAD_HEADS(BR_, CAUSAL_) \
    if (head_dim == 16) { SAGEATTN_LAUNCH_PERTHREAD_FP8(16, BR_, CAUSAL_); } \
    else if (head_dim == 64) { SAGEATTN_LAUNCH_PERTHREAD_FP8(64, BR_, CAUSAL_); } \
    else { SAGEATTN_LAUNCH_PERTHREAD_FP8(128, BR_, CAUSAL_); }
#else
#define SAGEATTN_DISPATCH_PERTHREAD_HEADS(BR_, CAUSAL_) \
    if (head_dim == 16) { SAGEATTN_LAUNCH_PERTHREAD_F16(16, BR_, CAUSAL_); } \
    else if (head_dim == 64) { SAGEATTN_LAUNCH_PERTHREAD_F16(64, BR_, CAUSAL_); } \
    else { SAGEATTN_LAUNCH_PERTHREAD_F16(128, BR_, CAUSAL_); }
#endif
    if (block_rows == 64) {
      if (is_causal) { SAGEATTN_DISPATCH_PERTHREAD_HEADS(64, true); }
      else { SAGEATTN_DISPATCH_PERTHREAD_HEADS(64, false); }
    } else {
      if (is_causal) { SAGEATTN_DISPATCH_PERTHREAD_HEADS(128, true); }
      else { SAGEATTN_DISPATCH_PERTHREAD_HEADS(128, false); }
    }
#undef SAGEATTN_DISPATCH_PERTHREAD_HEADS
#undef SAGEATTN_LAUNCH_PERTHREAD_F16
#undef SAGEATTN_LAUNCH_PERTHREAD_FP8
#undef SAGEATTN_LAUNCH_PERTHREAD_FP8_OUT
    hip_kernel_launch_check();
    return output;
  }
#define SAGEATTN_LAUNCH_FP8_2Q_OUT(BC_, HD_, HND_, BR_, OUT_T_) \
  if (is_causal) { \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, 0, ((HD_) / 16), HND_, BR_, false, true, OUT_T_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      value.data_ptr<uint8_t>(), \
      reinterpret_cast<OUT_T_*>(output.data_ptr()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr, \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk); \
  } else { \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, 0, ((HD_) / 16), HND_, BR_, false, false, OUT_T_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      value.data_ptr<uint8_t>(), \
      reinterpret_cast<OUT_T_*>(output.data_ptr()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr, \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk); \
  }
#define SAGEATTN_LAUNCH_FP8_2Q(BC_, HD_, HND_, BR_) \
  if (output_is_bf16) { \
    SAGEATTN_LAUNCH_FP8_2Q_OUT(BC_, HD_, HND_, BR_, __hip_bfloat16); \
  } else { \
    SAGEATTN_LAUNCH_FP8_2Q_OUT(BC_, HD_, HND_, BR_, __half); \
  }
#define SAGEATTN_LAUNCH_FP8_2Q_TV_OUT(BC_, HD_, BR_, OUT_T_) \
  if (is_causal) { \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, 0, ((HD_) / 16), true, BR_, true, true, OUT_T_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      value.data_ptr<uint8_t>(), \
      reinterpret_cast<OUT_T_*>(output.data_ptr()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr, \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(2), value.stride(1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk); \
  } else { \
    qk_int8_sv_f8_native_2q_kernel<BC_, HD_, 0, ((HD_) / 16), true, BR_, true, false, OUT_T_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      value.data_ptr<uint8_t>(), \
      reinterpret_cast<OUT_T_*>(output.data_ptr()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr, \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(2), value.stride(1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk); \
  }
#define SAGEATTN_LAUNCH_FP8_2Q_TV(BC_, HD_, BR_) \
  if (output_is_bf16) { \
    SAGEATTN_LAUNCH_FP8_2Q_TV_OUT(BC_, HD_, BR_, __hip_bfloat16); \
  } else { \
    SAGEATTN_LAUNCH_FP8_2Q_TV_OUT(BC_, HD_, BR_, __half); \
  }
#define SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID_HD(HD_, BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_, GRID_, FLAT_) \
  qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, true, false, F16ACC_, int8_t, false, int8_t, false, PVORDER_, VLANE_, STREAM_, KLANE_, HD_, FLAT_><<<GRID_, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(2), value.stride(1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk)
#define SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID(BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_, GRID_, FLAT_) \
  SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID_HD(64, BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_, GRID_, FLAT_)
#define SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_) \
  if (use_f16_flat_q_schedule) { \
    SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID(BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_, grid_f16_flat, true); \
  } else { \
    SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID(BC_, BR_, PAD_, F16ACC_, PVORDER_, VLANE_, STREAM_, KLANE_, grid, false); \
  }
#define SAGEATTN_LAUNCH_F16_D16_2Q_TV(BC_, BR_, PAD_, CAUSAL_, F16ACC_) \
  qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, CAUSAL_, false, F16ACC_, int8_t, false, int8_t, false, false, false, false, false, 16><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(2), value.stride(1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk)
#define SAGEATTN_LAUNCH_F16_D128_2Q_TV(BC_, BR_, PAD_) \
  if (is_causal) { \
    if (use_f16_d128_short_stream) { \
      SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID_HD(128, BC_, BR_, PAD_, false, false, false, true, false, grid, false); \
    } else { \
      SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL_GRID_HD(128, BC_, BR_, PAD_, false, false, false, false, false, grid, false); \
    } \
  } else { \
    if (use_f16_d128_short_stream) { \
      qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, false, false, false, int8_t, false, int8_t, false, false, false, true, false, 128><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
    } else { \
      qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_, false, false, false, int8_t, false, int8_t, false, false, false, false, false, 128><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
    } \
  }
#define SAGEATTN_LAUNCH_F16_2Q_TV(BC_, BR_, PAD_) \
  if (is_causal) { \
    if (use_f16_pv_accum) { \
      if (use_f16_pv_ordered_qk) { \
        if (use_f16_vlane && (BC_) == 64 && use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, true, true, true, false); } \
        else if (use_f16_vlane && (BC_) == 64) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, true, true, false, false); } \
        else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, true, false, true, false); } \
        else { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, true, false, false, false); } \
      } else { \
        if (use_f16_vlane && (BC_) == 64 && use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, false, true, true, false); } \
        else if (use_f16_vlane && (BC_) == 64) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, false, true, false, false); } \
        else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, false, false, true, false); } \
        else { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, true, false, false, false, false); } \
      } \
    } else { \
      if (use_f16_vlane && (BC_) == 64 && use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, false, false, true, true, false); } \
      else if (use_f16_vlane && (BC_) == 64) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, false, false, true, false, false); } \
      else if (use_f16_streamk) { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, false, false, false, true, false); } \
      else { SAGEATTN_LAUNCH_F16_2Q_TV_CAUSAL(BC_, BR_, PAD_, false, false, false, false, false); } \
    } \
  } else { \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, true, PAD_><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(2), value.stride(1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
  }
#define SAGEATTN_LAUNCH_F16_2Q(BC_, HND_, BR_) \
  if (is_causal) { \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, HND_, BR_, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD, true><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
  } else { \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, HND_, BR_><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
  }
#define SAGEATTN_LAUNCH_F16_2Q_TVLOAD_CAUSAL(BC_, BR_, PAD_, F16ACC_) \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, false, PAD_, true, true, F16ACC_><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk)
#define SAGEATTN_LAUNCH_F16_2Q_TVLOAD(BC_, BR_, PAD_) \
  if (is_causal) { \
    SAGEATTN_LAUNCH_F16_2Q_TVLOAD_CAUSAL(BC_, BR_, PAD_, false); \
  } else { \
    qk_int8_sv_f16_d64_native_2q_kernel<BC_, true, BR_, false, PAD_, false, true><<<grid, block, 0, stream>>>( \
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
        batch, q_len, kv_len, q_heads, kv_heads, \
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
        query_scale.stride(0), query_scale.stride(1), \
        key_scale.stride(0), key_scale.stride(1), \
        tensor_layout, sm_scale, use_per_thread_qk); \
  }
#define SAGEATTN_LAUNCH_F16_1Q(BC_, BR_) \
  qk_int8_sv_f16_d64_native_kernel<BC_, BR_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk)
#define SAGEATTN_LAUNCH_F16_1Q_CAUSAL(BR_, TRANSPOSED_, TVLOAD_, PAD_, F16ACC_) \
  qk_int8_sv_f16_d64_native_kernel<64, BR_, true, TRANSPOSED_, PAD_, true, TVLOAD_, F16ACC_><<<grid, block, 0, stream>>>( \
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), \
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), (TRANSPOSED_ ? value.stride(2) : value.stride(tensor_layout == kNHD ? 1 : 2)), \
      (TRANSPOSED_ ? value.stride(1) : value.stride(tensor_layout == kNHD ? 2 : 1)), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      query_scale.stride(0), query_scale.stride(1), \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, use_per_thread_qk)
#if SAGEATTN_GFX12_BUILD_ATTN_F16
  if (use_f16_causal_1q) {
    TORCH_CHECK(hnd_contiguous, "fp16 single-q causal path requires contiguous HND tensors");
    const bool use_f16_1q_pv_accum = use_f16_pv_accum;
#define SAGEATTN_DISPATCH_F16_1Q_CAUSAL(BR_, TRANSPOSED_, TVLOAD_, PAD_) \
    if (use_f16_1q_pv_accum) { \
      SAGEATTN_LAUNCH_F16_1Q_CAUSAL(BR_, TRANSPOSED_, TVLOAD_, PAD_, true); \
    } else { \
      SAGEATTN_LAUNCH_F16_1Q_CAUSAL(BR_, TRANSPOSED_, TVLOAD_, PAD_, false); \
    }
    if (value_transposed_hnd) {
      if (q_len >= 4096) {
        if (block_rows == 128) { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(128, true, false, 8); }
        else { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(64, true, false, 8); }
      } else if (q_len >= 1024) {
        if (block_rows == 128) { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(128, true, false, 4); }
        else { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(64, true, false, 4); }
      } else {
        if (block_rows == 128) { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(128, true, false, 16); }
        else { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(64, true, false, 16); }
      }
    } else if (use_f16_tvload) {
      if (block_rows == 128) { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(128, false, true, 4); }
      else { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(64, false, true, 4); }
    } else {
      if (block_rows == 128) { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(128, false, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD); }
      else { SAGEATTN_DISPATCH_F16_1Q_CAUSAL(64, false, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD); }
    }
#undef SAGEATTN_DISPATCH_F16_1Q_CAUSAL
  }
#endif // SAGEATTN_GFX12_BUILD_ATTN_F16
#if SAGEATTN_GFX12_BUILD_ATTN_FP8
#if SAGEATTN_GFX12_BUILD_ATTN_F16
  else if (use_fp8_2q && value_transposed_hnd) {
#else
  if (use_fp8_2q && value_transposed_hnd) {
#endif
    TORCH_CHECK(hnd_contiguous, "transposed fp8 value path requires contiguous HND Q/K/O");
    TORCH_CHECK(block_cols == 32 || block_cols == 64,
                "transposed fp8 value path currently supports BC32/BC64");
    TORCH_CHECK(!(block_rows == 256 && block_cols != 64),
                "transposed fp8 value BR256 path currently supports BC64");
    if (head_dim == 16) {
      if (block_cols == 32) {
        if (block_rows == 32) {
          SAGEATTN_LAUNCH_FP8_2Q_TV(32, 16, 32);
        } else if (block_rows == 64) {
          SAGEATTN_LAUNCH_FP8_2Q_TV(32, 16, 64);
        } else {
          SAGEATTN_LAUNCH_FP8_2Q_TV(32, 16, 128);
        }
      } else if (block_rows == 32) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 16, 32);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 16, 64);
      } else if (block_rows == 256) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 16, 256);
      } else if (block_rows == 512) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 16, 512);
      } else {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 16, 128);
      }
    } else if (block_rows == 512 && block_cols == 32 && head_dim == 128) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(32, 128, 512);
    } else if (block_rows == 512 && block_cols == 32) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(32, 64, 512);
    } else if (block_rows == 512 && head_dim == 128) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(64, 128, 512);
    } else if (block_rows == 512) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(64, 64, 512);
    } else if (block_rows == 256 && head_dim == 128) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(64, 128, 256);
    } else if (block_rows == 256) {
      SAGEATTN_LAUNCH_FP8_2Q_TV(64, 64, 256);
    } else if (block_cols == 32 && head_dim == 128) {
      if (block_rows == 32) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 128, 32);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 128, 64);
      } else {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 128, 128);
      }
    } else if (block_cols == 32) {
      if (block_rows == 32) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 64, 32);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 64, 64);
      } else {
        SAGEATTN_LAUNCH_FP8_2Q_TV(32, 64, 128);
      }
    } else if (head_dim == 128) {
      if (block_rows == 32) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 128, 32);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 128, 64);
      } else {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 128, 128);
      }
    } else {
      if (block_rows == 32) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 64, 32);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 64, 64);
      } else {
        SAGEATTN_LAUNCH_FP8_2Q_TV(64, 64, 128);
      }
    }
  } else if (use_fp8_2q && block_rows == 64 && block_cols == 32 && head_dim == 128) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(32, 128, true, 64);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(32, 128, false, 64);
    }
  } else if (use_fp8_2q && block_rows == 64 && block_cols == 32) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(32, 64, true, 64);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(32, 64, false, 64);
    }
  } else if (use_fp8_2q && block_rows == 64 && head_dim == 128) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, true, 64);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, false, 64);
    }
  } else if (use_fp8_2q && block_rows == 64) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, true, 64);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, false, 64);
    }
  } else if (use_fp8_2q && block_rows == 256 && head_dim == 128) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, true, 256);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, false, 256);
    }
  } else if (use_fp8_2q && block_rows == 256) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, true, 256);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, false, 256);
    }
  } else if (use_fp8_2q && block_cols == 32 && head_dim == 128) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(32, 128, true, 128);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(32, 128, false, 128);
    }
  } else if (use_fp8_2q && block_cols == 32) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(32, 64, true, 128);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(32, 64, false, 128);
    }
  } else if (use_fp8_2q && block_cols == 128 && head_dim == 128) {
    qk_int8_sv_f8_native_2q_kernel<128, 128, 0, 8><<<grid, block, 0, stream>>>(
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(),
        value.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr,
        batch, q_len, kv_len, q_heads, kv_heads,
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1),
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1),
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1),
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1),
        query_scale.stride(0), query_scale.stride(1),
        key_scale.stride(0), key_scale.stride(1),
        tensor_layout, sm_scale, use_per_thread_qk);
  } else if (use_fp8_2q && block_cols == 128) {
    qk_int8_sv_f8_native_2q_kernel<128, 64, 0, 4><<<grid, block, 0, stream>>>(
        query.data_ptr<int8_t>(), key.data_ptr<int8_t>(),
        value.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale_ptr,
        batch, q_len, kv_len, q_heads, kv_heads,
        query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1),
        key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1),
        value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1),
        output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1),
        query_scale.stride(0), query_scale.stride(1),
        key_scale.stride(0), key_scale.stride(1),
        tensor_layout, sm_scale, use_per_thread_qk);
  } else if (use_fp8_2q && head_dim == 128) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, true, 128);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 128, false, 128);
    }
  } else if (use_fp8_2q) {
    if (hnd_contiguous) {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, true, 128);
    } else {
      SAGEATTN_LAUNCH_FP8_2Q(64, 64, false, 128);
    }
  }
#if SAGEATTN_GFX12_BUILD_ATTN_FP8 && !SAGEATTN_GFX12_BUILD_ATTN_F16
  else {
    TORCH_CHECK(false, "native gfx12 fp8 attention dispatch could not select a kernel");
  }
#endif
#endif // SAGEATTN_GFX12_BUILD_ATTN_FP8
#if SAGEATTN_GFX12_BUILD_ATTN_F16
  else if (use_2q && value_transposed_hnd) {
    TORCH_CHECK(hnd_contiguous, "transposed fp16 value path requires contiguous HND Q/K/O");
    if (head_dim == 128) {
      if (block_rows == 32) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 32, 4);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 64, 4);
      } else if (block_rows == 256) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 256, 4);
      } else if (block_rows == 512) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 512, 4);
      } else if (block_rows == 1024) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 1024, 4);
      } else if (q_len >= 8192) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 128, 8);
      } else if (q_len >= 1024) {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 128, 4);
      } else {
        SAGEATTN_LAUNCH_F16_D128_2Q_TV(64, 128, 16);
      }
    } else if (head_dim == 16) {
      if (is_causal) {
        if (block_rows == 32) {
          SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 32, 4, true, true);
        } else if (block_rows == 64) {
          SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 64, 4, true, true);
        } else if (block_rows == 256) {
          SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 256, 4, true, true);
        } else {
          SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 128, 4, true, true);
        }
      } else if (block_rows == 32) {
        SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 32, 4, false, true);
      } else if (block_rows == 64) {
        SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 64, 4, false, true);
      } else if (block_rows == 256) {
        SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 256, 4, false, true);
      } else {
        SAGEATTN_LAUNCH_F16_D16_2Q_TV(64, 128, 4, false, true);
      }
    } else if (block_rows == 32) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 32, 4);
    } else if (block_rows == 64) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 64, 4);
    } else if (block_rows == 256) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 256, 4);
    } else if (block_rows == 512) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 512, 4);
    } else if (block_rows == 1024) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 1024, 4);
    } else if (q_len >= 8192) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 128, 8);
    } else if (q_len >= 1024) {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 128, 4);
    } else {
      SAGEATTN_LAUNCH_F16_2Q_TV(64, 128, 16);
    }
  } else if (use_2q && block_cols == 128) {
    if (hnd_contiguous) {
      if (use_f16_tvload) {
        if (block_rows == 64) {
          SAGEATTN_LAUNCH_F16_2Q_TVLOAD(128, 64, 16);
        } else {
          SAGEATTN_LAUNCH_F16_2Q_TVLOAD(128, 128, 16);
        }
      } else {
        if (block_rows == 64) {
          SAGEATTN_LAUNCH_F16_2Q(128, true, 64);
        } else {
          SAGEATTN_LAUNCH_F16_2Q(128, true, 128);
        }
      }
    } else {
      if (block_rows == 64) {
        SAGEATTN_LAUNCH_F16_2Q(128, false, 64);
      } else {
        SAGEATTN_LAUNCH_F16_2Q(128, false, 128);
      }
    }
  } else if (use_2q) {
    if (hnd_contiguous) {
      if (use_f16_tvload) {
        if (block_rows == 64) {
          if (is_causal && use_f16_pv_accum) {
            SAGEATTN_LAUNCH_F16_2Q_TVLOAD_CAUSAL(64, 64, 4, true);
          } else {
            SAGEATTN_LAUNCH_F16_2Q_TVLOAD(64, 64, 4);
          }
        } else {
          if (is_causal && use_f16_pv_accum) {
            SAGEATTN_LAUNCH_F16_2Q_TVLOAD_CAUSAL(64, 128, 4, true);
          } else {
            SAGEATTN_LAUNCH_F16_2Q_TVLOAD(64, 128, 4);
          }
        }
      } else {
        if (block_rows == 64) {
          SAGEATTN_LAUNCH_F16_2Q(64, true, 64);
        } else {
          SAGEATTN_LAUNCH_F16_2Q(64, true, 128);
        }
      }
    } else {
      if (block_rows == 64) {
        SAGEATTN_LAUNCH_F16_2Q(64, false, 64);
      } else {
        SAGEATTN_LAUNCH_F16_2Q(64, false, 128);
      }
    }
  } else if (block_cols == 128 && block_rows == 128) {
    SAGEATTN_LAUNCH_F16_1Q(128, 128);
  } else if (block_cols == 128) {
    SAGEATTN_LAUNCH_F16_1Q(128, 64);
  } else if (block_rows == 128) {
    SAGEATTN_LAUNCH_F16_1Q(64, 128);
  } else {
    SAGEATTN_LAUNCH_F16_1Q(64, 64);
  }
#endif // SAGEATTN_GFX12_BUILD_ATTN_F16
  hip_kernel_launch_check();
  return output;
}

#if SAGEATTN_GFX12_BUILD_ATTN_F16

torch::Tensor qk_int8_sv_f16_d64_native_attn_gfx12_dispatch(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    torch::Tensor value_scale,
    int value_transposed_hnd_hint,
    int pv_accum_mode) {
  return qk_int8_sv_f16_d64_native_attn_gfx12_impl(
      query, key, value, output, query_scale, key_scale, tensor_layout,
      is_causal, sm_scale, valid_kv_len, value_scale,
      value_transposed_hnd_hint, pv_accum_mode);
}

#endif // SAGEATTN_GFX12_BUILD_ATTN_F16

#endif // SAGEATTN_GFX12_BUILD_ATTN_F16 || SAGEATTN_GFX12_BUILD_ATTN_FP8

#if SAGEATTN_GFX12_BUILD_RAWQ_FP8

static torch::Tensor qk_rawq_int8_sv_f8_native_attn_gfx12_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    int value_transposed_hnd_hint,
    int key_hnd_layout) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && output.is_cuda(),
              "raw-Q gfx12 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 && output.dim() == 4,
              "raw-Q gfx12 attention expects 4D Q/K/V/O tensors");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 ||
                  query.scalar_type() == torch::kBFloat16,
              "raw-Q gfx12 attention supports fp16/bf16 query");
  TORCH_CHECK(key.scalar_type() == torch::kInt8,
              "raw-Q gfx12 attention expects pre-quantized int8 key");
  TORCH_CHECK(value.scalar_type() == torch::kByte,
              "raw-Q gfx12 attention expects raw OCP e4m3 fp8 value bytes");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 ||
                  output.scalar_type() == torch::kBFloat16,
              "raw-Q gfx12 attention output must be fp16 or bf16");
  TORCH_CHECK(key_scale.scalar_type() == torch::kFloat32,
              "raw-Q gfx12 attention key_scale must be fp32");
  TORCH_CHECK(tensor_layout == kHND || tensor_layout == kNHD, "invalid tensor_layout");
  TORCH_CHECK(key_hnd_layout == 0 || key_hnd_layout == 1,
              "key_hnd_layout must be 0 or 1");
  TORCH_CHECK(tensor_layout == kNHD || key_hnd_layout == 0,
              "key_hnd_layout is only needed for NHD query/output tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous() && output.is_contiguous(),
              "raw-Q gfx12 attention expects contiguous tensors");

  const int64_t head_dim = query.size(-1);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128 || head_dim == 256,
              "raw-Q gfx12 fp8 path supports head_dim 16, 64, 128, or 256");
  const int64_t batch = query.size(0);
  const int64_t q_heads = tensor_layout == kNHD ? query.size(2) : query.size(1);
  const int64_t q_len = tensor_layout == kNHD ? query.size(1) : query.size(2);
  const int64_t out_q_len = tensor_layout == kNHD ? output.size(1) : output.size(2);
  const bool key_hnd_contiguous = tensor_layout == kHND || key_hnd_layout != 0;
  const int64_t kv_heads = key_hnd_contiguous ? key.size(1) :
      (tensor_layout == kNHD ? key.size(2) : key.size(1));
  const int64_t padded_kv_len = key_hnd_contiguous ? key.size(2) :
      (tensor_layout == kNHD ? key.size(1) : key.size(2));
  const int64_t kv_len = valid_kv_len > 0 ? valid_kv_len : padded_kv_len;
  TORCH_CHECK(kv_len > 0 && kv_len <= padded_kv_len,
              "valid_kv_len must be in (0, padded_kv_len]");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch &&
                  output.size(0) == batch,
              "raw-Q gfx12 batch size mismatch");
  TORCH_CHECK(value_transposed_hnd_hint >= -1 && value_transposed_hnd_hint <= 1,
              "value_transposed_hnd must be -1, 0, or 1");
  const bool value_shape_transposed_hnd =
      value.size(1) == kv_heads && value.size(2) == head_dim &&
      value.size(3) >= padded_kv_len;
  const bool value_shape_normal =
      (tensor_layout == kNHD &&
       value.size(1) == padded_kv_len && value.size(2) == kv_heads &&
       value.size(3) == head_dim) ||
      (tensor_layout == kHND &&
       value.size(1) == kv_heads && value.size(2) == padded_kv_len &&
       value.size(3) == head_dim);
  const bool value_layout_ambiguous =
      value_shape_transposed_hnd && value_shape_normal;
  TORCH_CHECK(value_transposed_hnd_hint <= 0 || value_shape_transposed_hnd,
              "value_transposed_hnd=1 requires value shape [B, H, D, padded_kv_len]");
  TORCH_CHECK(value_transposed_hnd_hint != 0 || value_shape_normal,
              "value_transposed_hnd=0 requires normal value layout");
  TORCH_CHECK(value_transposed_hnd_hint >= 0 || !value_layout_ambiguous,
              "raw-Q gfx12 value layout is ambiguous; pass value_transposed_hnd=0 "
              "for normal layout or 1 for transposed HND [B, H, D, padded_kv_len]");
  const bool value_transposed_hnd =
      value_transposed_hnd_hint > 0 ||
      (value_transposed_hnd_hint < 0 && value_shape_transposed_hnd);
  TORCH_CHECK(key.size(0) == batch && key.size(-1) == head_dim &&
                  output.size(-1) == head_dim &&
                  (value_transposed_hnd || value_shape_normal),
              "raw-Q gfx12 Q/K/V/O head_dim mismatch");
  const bool key_shape_matches =
      key_hnd_contiguous
          ? (key.size(1) == kv_heads && key.size(2) == padded_kv_len)
          : (tensor_layout == kNHD
                 ? (key.size(1) == padded_kv_len && key.size(2) == kv_heads)
                 : (key.size(1) == kv_heads && key.size(2) == padded_kv_len));
  TORCH_CHECK(key_shape_matches, "raw-Q gfx12 key shape mismatch");
  TORCH_CHECK((tensor_layout == kNHD &&
                   ((value_transposed_hnd && output.size(1) >= q_len &&
                     output.size(2) == q_heads) ||
                    (!value_transposed_hnd && value.size(1) == padded_kv_len &&
                     output.size(1) >= q_len && value.size(2) == kv_heads &&
                     output.size(2) == q_heads))) ||
                  (tensor_layout == kHND &&
                   ((value_transposed_hnd && output.size(2) >= q_len &&
                     output.size(1) == q_heads) ||
                    (!value_transposed_hnd && value.size(2) == padded_kv_len &&
                     output.size(2) >= q_len && value.size(1) == kv_heads &&
                     output.size(1) == q_heads))),
              "raw-Q gfx12 Q/K/V/O shape mismatch");
  TORCH_CHECK((q_heads % kv_heads) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK((padded_kv_len % 64) == 0,
              "raw-Q gfx12 attention requires padded kv_len divisible by 64");
  TORCH_CHECK(!is_causal || (q_len % 64) == 0,
              "raw-Q gfx12 causal attention requires q_len divisible by 64");
  TORCH_CHECK(!is_causal || q_len == padded_kv_len,
              "raw-Q gfx12 causal attention requires q_len == padded_kv_len");
  TORCH_CHECK(key_scale.stride(-1) == 1, "key_scale must have contiguous scale columns");
  const bool has_value_scale = value_scale.defined() && value_scale.numel() > 0;
  if (has_value_scale) {
    TORCH_CHECK(value_scale.is_cuda(), "value_scale must be a CUDA/HIP tensor");
    TORCH_CHECK(value_scale.scalar_type() == torch::kFloat32,
                "value_scale must be fp32");
    TORCH_CHECK(value_scale.dim() == 3 && value_scale.is_contiguous(),
                "value_scale must be contiguous [B, H_kv, D]");
    TORCH_CHECK(value_scale.size(0) == batch &&
                    value_scale.size(1) == kv_heads &&
                    value_scale.size(2) == head_dim,
                "value_scale shape must match [B, H_kv, D]");
  }
  const float* value_scale_ptr = has_value_scale ? value_scale.data_ptr<float>() : nullptr;

  int block_rows = head_dim == 64 ?
      select_fp8_d64_block_rows_gfx12(q_len, is_causal, value_transposed_hnd) :
      (q_len <= 64 ? 64 : 128);
  if (head_dim == 64 && !is_causal && value_transposed_hnd) {
    if (q_len == 1024) {
      block_rows = 256;
    }
  }
  if (head_dim == 16 && is_causal && q_len <= 1024) {
    block_rows = 64;
  }
  const int64_t q_blocks = (q_len + block_rows - 1) / block_rows;
  TORCH_CHECK(out_q_len >= q_blocks * block_rows,
              "raw-Q gfx12 attention output must cover the padded query tail");

  const bool use_bc32 =
      !is_causal && value_transposed_hnd && tensor_layout == kNHD &&
      !key_hnd_contiguous && head_dim == 128 && q_len == 1024;
  const bool hnd_contiguous = tensor_layout == kHND;
  const dim3 block(block_rows);
  const dim3 grid(q_blocks, q_heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
#define SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(BC_, HD_, VBASE_, VTILES_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, INVL_) \
  qk_int8_sv_f8_native_2q_kernel<BC_, HD_, VBASE_, VTILES_, HND_, BR_, VT_, CAUSAL_, OUT_T_, QUERY_T_, true, int8_t, uint8_t, false, false, (VT_ ? -1 : 0), false, false, 2, false, false, KEY_HND_, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, INVL_><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const QUERY_T_*>(query.data_ptr<QUERY_AT_T_>()), \
      key.data_ptr<int8_t>(), value.data_ptr<uint8_t>(), \
      reinterpret_cast<OUT_T_*>(output.data_ptr<OUT_AT_T_>()), \
      nullptr, key_scale.data_ptr<float>(), value_scale_ptr, \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(key_hnd_contiguous ? 2 : (tensor_layout == kNHD ? 1 : 2)), key.stride(key_hnd_contiguous ? 1 : (tensor_layout == kNHD ? 2 : 1)), \
      value.stride(0), (VT_ ? value.stride(2) : value.stride(tensor_layout == kNHD ? 1 : 2)), (VT_ ? value.stride(1) : value.stride(tensor_layout == kNHD ? 2 : 1)), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      0, 0, \
      key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale)
#define SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, INVL_) \
  SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(BC_, HD_, 0, ((HD_) / 16), HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, INVL_)
#define SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_) \
  do { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(BC_, 256, 0, 8, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, false, false, false, false, false); \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(BC_, 256, 8, 8, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, false, false, false, false, false); \
  } while (false)
#define SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256_STATIC_TWO() \
  do { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(32, 256, 0, 8, false, false, 128, true, true, __half, __half, at::Half, at::Half, true, true, true, true, true); \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE(32, 256, 8, 8, false, false, 128, true, true, __half, __half, at::Half, at::Half, true, true, true, true, true); \
  } while (false)
#define SAGEATTN_LAUNCH_RAWQ_FP8_TYPED(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_) \
  SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, OUT_T_, QUERY_AT_T_, OUT_AT_T_, false, false, false, false, false)
#define SAGEATTN_DISPATCH_RAWQ_FP8_OUT_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, QUERY_AT_T_) \
  if (output.scalar_type() == torch::kBFloat16) { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, __hip_bfloat16, QUERY_AT_T_, at::BFloat16); \
  } else { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, __half, QUERY_AT_T_, at::Half); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_QUERY_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_) \
  if (query.scalar_type() == torch::kBFloat16) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_OUT_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, __hip_bfloat16, at::BFloat16); \
  } else { \
    SAGEATTN_DISPATCH_RAWQ_FP8_OUT_D256(BC_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, __half, at::Half); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_BR_D256(BC_, HND_, KEY_HND_, VT_, CAUSAL_) \
  if (block_rows == 64) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY_D256(BC_, HND_, KEY_HND_, 64, VT_, CAUSAL_); \
  } else if (block_rows == 256) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY_D256(BC_, HND_, KEY_HND_, 256, VT_, CAUSAL_); \
  } else { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY_D256(BC_, HND_, KEY_HND_, 128, VT_, CAUSAL_); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_OUT(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, QUERY_AT_T_) \
  if (output.scalar_type() == torch::kBFloat16) { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, __hip_bfloat16, QUERY_AT_T_, at::BFloat16); \
  } else { \
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, QUERY_T_, __half, QUERY_AT_T_, at::Half); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_QUERY(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_) \
  if (query.scalar_type() == torch::kBFloat16) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_OUT(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, __hip_bfloat16, at::BFloat16); \
  } else { \
    SAGEATTN_DISPATCH_RAWQ_FP8_OUT(BC_, HD_, HND_, KEY_HND_, BR_, VT_, CAUSAL_, __half, at::Half); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_BR(BC_, HD_, HND_, KEY_HND_, VT_, CAUSAL_) \
  if (block_rows == 64) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY(BC_, HD_, HND_, KEY_HND_, 64, VT_, CAUSAL_); \
  } else if (block_rows == 256) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY(BC_, HD_, HND_, KEY_HND_, 256, VT_, CAUSAL_); \
  } else { \
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY(BC_, HD_, HND_, KEY_HND_, 128, VT_, CAUSAL_); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, HND_, KEY_HND_, VT_, CAUSAL_) \
  if (head_dim == 16) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_BR(BC_, 16, HND_, KEY_HND_, VT_, CAUSAL_); \
  } else if (head_dim == 64) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_BR(BC_, 64, HND_, KEY_HND_, VT_, CAUSAL_); \
  } else if (head_dim == 128) { \
    SAGEATTN_DISPATCH_RAWQ_FP8_BR(BC_, 128, HND_, KEY_HND_, VT_, CAUSAL_); \
  } else { \
    SAGEATTN_DISPATCH_RAWQ_FP8_BR_D256(BC_, HND_, KEY_HND_, VT_, CAUSAL_); \
  }
#define SAGEATTN_DISPATCH_RAWQ_FP8_LAYOUT(BC_) \
  if (hnd_contiguous) { \
    if (is_causal) { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, true, true, true, true); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, true, true, false, true); } \
    } else { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, true, true, true, false); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, true, true, false, false); } \
    } \
  } else if (key_hnd_contiguous) { \
    if (is_causal) { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, true, true, true); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, true, false, true); } \
    } else { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, true, true, false); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, true, false, false); } \
    } \
  } else { \
    if (is_causal) { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, false, true, true); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, false, false, true); } \
    } else { \
      if (value_transposed_hnd) { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, false, true, false); } \
      else { SAGEATTN_DISPATCH_RAWQ_FP8_HD(BC_, false, false, false, false); } \
    } \
  }

  const bool use_static_short_nhd =
      !is_causal && value_transposed_hnd && tensor_layout == kNHD &&
      !key_hnd_contiguous && q_heads == kv_heads && q_len == kv_len &&
      q_len == 512 && head_dim == 128 &&
      query.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16;
  const bool use_static_causal_nhd =
      is_causal && value_transposed_hnd && tensor_layout == kNHD &&
      !key_hnd_contiguous && q_heads == kv_heads && q_len == kv_len &&
      block_rows == 128 &&
      (q_len % block_rows) == 0 &&
      (head_dim == 128 || (head_dim == 64 && q_len >= 1024)) &&
      query.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16;
  const bool use_bc32_causal_short_nhd =
      use_static_causal_nhd && head_dim == 128 && q_len <= 1024;
  const bool use_d256_static_causal_nhd =
      is_causal && value_transposed_hnd && tensor_layout == kNHD &&
      !key_hnd_contiguous && q_heads == kv_heads && q_len == kv_len &&
      block_rows == 128 &&
      (q_len % block_rows) == 0 &&
      head_dim == 256 &&
      query.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16;

  if (use_static_short_nhd) {
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(64, 128, false, false, 128, true, false,
                                      __half, __half, at::Half, at::Half,
                                      true, true, true, true, false);
  } else if (use_static_causal_nhd && head_dim == 64) {
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(64, 64, false, false, 128, true, true,
                                      __half, __half, at::Half, at::Half,
                                      true, true, true, true, false);
  } else if (use_d256_static_causal_nhd) {
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256_STATIC_TWO();
  } else if (use_bc32_causal_short_nhd) {
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(32, 128, false, false, 128, true, true,
                                      __half, __half, at::Half, at::Half,
                                      true, true, true, true, true);
  } else if (use_static_causal_nhd) {
    SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX(64, 128, false, false, 128, true, true,
                                      __half, __half, at::Half, at::Half,
                                      true, true, true, true, false);
  } else if (use_bc32) {
    SAGEATTN_DISPATCH_RAWQ_FP8_QUERY(32, 128, false, false, 128, true, false);
  } else {
    SAGEATTN_DISPATCH_RAWQ_FP8_LAYOUT(64);
  }

#undef SAGEATTN_DISPATCH_RAWQ_FP8_LAYOUT
#undef SAGEATTN_DISPATCH_RAWQ_FP8_HD
#undef SAGEATTN_DISPATCH_RAWQ_FP8_BR
#undef SAGEATTN_DISPATCH_RAWQ_FP8_QUERY
#undef SAGEATTN_DISPATCH_RAWQ_FP8_OUT
#undef SAGEATTN_DISPATCH_RAWQ_FP8_BR_D256
#undef SAGEATTN_DISPATCH_RAWQ_FP8_QUERY_D256
#undef SAGEATTN_DISPATCH_RAWQ_FP8_OUT_D256
#undef SAGEATTN_LAUNCH_RAWQ_FP8_TYPED
#undef SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256
#undef SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_D256_STATIC_TWO
#undef SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_EX
#undef SAGEATTN_LAUNCH_RAWQ_FP8_TYPED_SLICE
  hip_kernel_launch_check();
  return output;
}

torch::Tensor qk_rawq_int8_sv_f8_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    int value_transposed_hnd,
    int key_hnd_layout) {
  return qk_rawq_int8_sv_f8_native_attn_gfx12_impl(
      query, key, value, output, key_scale, torch::Tensor(),
      tensor_layout, is_causal, sm_scale, valid_kv_len, value_transposed_hnd,
      key_hnd_layout);
}

torch::Tensor qk_rawq_int8_sv_f8_scaled_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    int value_transposed_hnd,
    int key_hnd_layout) {
  return qk_rawq_int8_sv_f8_native_attn_gfx12_impl(
      query, key, value, output, key_scale, value_scale,
      tensor_layout, is_causal, sm_scale, valid_kv_len, value_transposed_hnd,
      key_hnd_layout);
}

std::vector<torch::Tensor> mean_and_fp8_value_nhd_short_gfx12(
    torch::Tensor key,
    torch::Tensor value,
    double scale_max);

torch::Tensor sage_fp8_nhd_short_mha_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int is_causal,
    float sm_scale,
    double scale_max) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "gfx12 short fp8 wrapper expects CUDA/HIP tensors");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "gfx12 short fp8 wrapper expects [B, S, H, D]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "gfx12 short fp8 wrapper expects contiguous NHD tensors");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 &&
                  key.scalar_type() == torch::kFloat16 &&
                  value.scalar_type() == torch::kFloat16,
              "gfx12 short fp8 wrapper currently supports fp16 inputs");
  TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
              "gfx12 short fp8 wrapper expects matching Q/K/V shapes");
  const int64_t batch = query.size(0);
  const int64_t seq_len = query.size(1);
  const int64_t heads = query.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK((seq_len == 512 || seq_len == 1024 || seq_len == 2048 ||
                   seq_len == 4096 || seq_len == 8192) &&
                  (head_dim == 64 || head_dim == 128),
              "gfx12 fp8 wrapper supports S512/S1024/S2048/S4096/S8192 and D64/D128");

  std::vector<torch::Tensor> prep =
      mean_and_fp8_value_nhd_short_gfx12(key, value, scale_max);
  torch::Tensor key_mean = prep[0];
  torch::Tensor value_native = prep[1];
  torch::Tensor value_scale = prep[2];
  torch::Tensor key_int8 = torch::empty_like(key, key.options().dtype(torch::kInt8));
  torch::Tensor key_scale =
      torch::empty({batch, heads, (seq_len + 63) / 64},
                   key.options().dtype(torch::kFloat32));

  const dim3 grid((seq_len + 63) / 64, heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (head_dim == 64) {
    constexpr int NumPack = 1;
    dim3 block(64 * (64 / 8) / NumPack);
    quant_k_nhd_fuse_sub_mean_short_kernel<__half, 64, NumPack><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_mean.data_ptr<at::Half>()),
        key_int8.data_ptr<int8_t>(),
        key_scale.data_ptr<float>(),
        seq_len, heads);
  } else {
    constexpr int NumPack = 2;
    dim3 block(64 * (128 / 8) / NumPack);
    quant_k_nhd_fuse_sub_mean_short_kernel<__half, 128, NumPack><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_mean.data_ptr<at::Half>()),
        key_int8.data_ptr<int8_t>(),
        key_scale.data_ptr<float>(),
        seq_len, heads);
  }
  hip_kernel_launch_check();

  torch::Tensor output = torch::empty_like(query);
  qk_rawq_int8_sv_f8_scaled_native_attn_gfx12(
      query, key_int8, value_native, output, key_scale, value_scale,
      kNHD, is_causal, sm_scale, seq_len, 1, 0);
  return output;
}

#endif // SAGEATTN_GFX12_BUILD_RAWQ_FP8

#if SAGEATTN_GFX12_BUILD_ATTN_FP8

torch::Tensor qk_int8_sv_f8_scaled_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len) {
  return qk_int8_sv_f16_d64_native_attn_gfx12_impl(
      query, key, value, output, query_scale, key_scale, tensor_layout,
      is_causal, sm_scale, valid_kv_len, value_scale, 1, -1);
}

#endif // SAGEATTN_GFX12_BUILD_ATTN_FP8

#if SAGEATTN_GFX12_BUILD_ATTN_F16

torch::Tensor qk_int8_sv_f16_d64_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    int value_transposed_hnd,
    int pv_accum_mode) {
  return qk_int8_sv_f16_d64_native_attn_gfx12_impl(
      query, key, value, output, query_scale, key_scale, tensor_layout,
      is_causal, sm_scale, valid_kv_len, torch::Tensor(), value_transposed_hnd,
      pv_accum_mode);
}

torch::Tensor qk_rawq_int8_sv_f16_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len,
    int pv_accum_mode) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && output.is_cuda(),
              "raw-Q fp16 gfx12 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 ||
                  query.scalar_type() == torch::kBFloat16,
              "raw-Q fp16 gfx12 query must be fp16 or bf16");
  TORCH_CHECK(key.scalar_type() == torch::kInt8, "raw-Q fp16 gfx12 key must be int8");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16, "raw-Q fp16 gfx12 value must be fp16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "raw-Q fp16 gfx12 output must be fp16");
  TORCH_CHECK(key_scale.scalar_type() == torch::kFloat32,
              "raw-Q fp16 gfx12 key_scale must be fp32");
  TORCH_CHECK(tensor_layout == kHND || tensor_layout == kNHD, "invalid tensor_layout");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 && output.dim() == 4,
              "raw-Q fp16 gfx12 attention expects 4D tensors");
  const int64_t head_dim = query.size(-1);
  TORCH_CHECK(head_dim == 16 || head_dim == 64 || head_dim == 128 || head_dim == 256,
              "raw-Q fp16 gfx12 supports D16/D64/D128/D256");
  TORCH_CHECK(key.size(-1) == head_dim && value.size(-1) == head_dim &&
                  output.size(-1) == head_dim,
              "raw-Q fp16 gfx12 tensors must have matching head_dim");
  const int64_t batch = query.size(0);
  const int64_t q_heads = tensor_layout == kNHD ? query.size(2) : query.size(1);
  const int64_t q_len = tensor_layout == kNHD ? query.size(1) : query.size(2);
  const int64_t kv_heads = tensor_layout == kNHD ? key.size(2) : key.size(1);
  const int64_t padded_kv_len = tensor_layout == kNHD ? key.size(1) : key.size(2);
  const int64_t kv_len = valid_kv_len > 0 ? valid_kv_len : padded_kv_len;
  TORCH_CHECK(kv_len > 0 && kv_len <= padded_kv_len,
              "valid_kv_len must be in (0, padded_kv_len]");
  TORCH_CHECK((padded_kv_len % 64) == 0,
              "raw-Q fp16 gfx12 requires kv_len to be a multiple of 64");
  TORCH_CHECK(!is_causal || q_len == padded_kv_len,
              "raw-Q fp16 gfx12 causal path requires q_len == kv_len");
  TORCH_CHECK((q_heads % kv_heads) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(key_scale.dim() == 3 && key_scale.stride(-1) == 1,
              "raw-Q fp16 gfx12 key_scale must be [B, H_kv, ceil(K/64)]");
  TORCH_CHECK(key_scale.size(0) == batch && key_scale.size(1) == kv_heads &&
                  key_scale.size(2) == (padded_kv_len + 63) / 64,
              "raw-Q fp16 gfx12 key_scale shape mismatch");
  TORCH_CHECK(pv_accum_mode >= -1 && pv_accum_mode <= 1,
              "invalid gfx12 fp16 PV accumulation mode");

  const bool hnd_contiguous = tensor_layout == kHND &&
      query.is_contiguous() && key.is_contiguous() &&
      value.is_contiguous() && output.is_contiguous();
  int block_rows = q_len <= 64 ? 64 : 128;
  const dim3 block(block_rows);
  const dim3 grid((q_len + block_rows - 1) / block_rows, q_heads, batch);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_d128_short_stream =
      is_causal && head_dim == 128 && block_rows == 128 && q_len <= 1024;
  const bool use_direct_stream_probs =
      use_d128_short_stream && q_len == 1024 && pv_accum_mode != 1;
  const bool use_d128_long_stream =
      is_causal && head_dim == 128 && block_rows == 128 &&
      q_len >= 2048 && pv_accum_mode != 1;
  const bool use_d64_noncausal_stream_direct =
      !is_causal && head_dim == 64 && block_rows == 128 &&
      q_len >= 1024 && pv_accum_mode != 1;
  const bool use_f16_d64_static_long =
      head_dim == 64 && (q_len == 2048 || q_len == 4096 || q_len == 8192);
  const bool use_f16_d128_static_long =
      head_dim == 128 && (q_len == 2048 || q_len == 4096 || q_len == 8192);
  const bool use_f16_d256_static =
      head_dim == 256 && is_causal && (q_len == 512 || q_len == 1024);
  const bool use_f16_d256_s512_stream_direct =
      use_f16_d256_static && q_len == 512;
  const bool use_static_nhd_no_tail =
      query.scalar_type() == torch::kFloat16 &&
      tensor_layout == kNHD && q_heads == kv_heads &&
      block_rows == 128 &&
      ((!is_causal && (q_len == 512 || q_len == 1024)) ||
       (is_causal && (q_len == 512 || q_len == 1024)) ||
       use_f16_d64_static_long ||
       use_f16_d128_static_long ||
       use_f16_d256_static) &&
      q_len == padded_kv_len && kv_len == padded_kv_len &&
      (head_dim == 64 || head_dim == 128 || use_f16_d256_static);

#define SAGEATTN_LAUNCH_RAWQ_F16_VALUE(BC_, HD_, HND_, BR_, CAUSAL_, QUERY_T_, F16ACC_, STREAM_, PVORDER_, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, PREFETCH_STREAM_V_, DIRECT_STREAM_PROBS_, DIRECT_PV_OUTFRAG_) \
  qk_int8_sv_f16_d64_native_2q_kernel<BC_, HND_, BR_, false, SAGEATTN_GFX12_NATIVE_F16_TV_PAD, CAUSAL_, false, F16ACC_, QUERY_T_, true, int8_t, false, PVORDER_, false, STREAM_, false, HD_, false, false, STATIC_NHD_, NO_TAIL_, SAME_HEADS_, NO_Q_TAIL_, PREFETCH_STREAM_V_, DIRECT_STREAM_PROBS_, DIRECT_PV_OUTFRAG_><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const QUERY_T_*>(query.data_ptr()), key.data_ptr<int8_t>(), \
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
      nullptr, key_scale.data_ptr<float>(), \
      batch, q_len, kv_len, q_heads, kv_heads, \
      query.stride(0), query.stride(tensor_layout == kNHD ? 1 : 2), query.stride(tensor_layout == kNHD ? 2 : 1), \
      key.stride(0), key.stride(tensor_layout == kNHD ? 1 : 2), key.stride(tensor_layout == kNHD ? 2 : 1), \
      value.stride(0), value.stride(tensor_layout == kNHD ? 1 : 2), value.stride(tensor_layout == kNHD ? 2 : 1), \
      output.stride(0), output.stride(tensor_layout == kNHD ? 1 : 2), output.stride(tensor_layout == kNHD ? 2 : 1), \
      0, 0, key_scale.stride(0), key_scale.stride(1), \
      tensor_layout, sm_scale, false)
#define SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, BR_, CAUSAL_, QUERY_T_, F16ACC_, STREAM_) \
  SAGEATTN_LAUNCH_RAWQ_F16_VALUE(64, HD_, HND_, BR_, CAUSAL_, QUERY_T_, F16ACC_, STREAM_, false, false, false, false, false, false, false, false)
#define SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, CAUSAL_, QUERY_T_, F16ACC_, STREAM_, PREFETCH_STREAM_V_, DIRECT_STREAM_PROBS_, DIRECT_PV_OUTFRAG_) \
  SAGEATTN_LAUNCH_RAWQ_F16_VALUE(BC_, HD_, false, BR_, CAUSAL_, QUERY_T_, F16ACC_, STREAM_, true, true, true, true, true, PREFETCH_STREAM_V_, DIRECT_STREAM_PROBS_, DIRECT_PV_OUTFRAG_)
#define SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(HD_, HND_, QUERY_T_) \
  if (is_causal) { \
    if (pv_accum_mode == 1) { \
      if (block_rows == 64) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 64, true, QUERY_T_, true, false); } \
      else if ((HD_) == 128 && use_d128_short_stream) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, true, QUERY_T_, true, true); } \
      else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, true, QUERY_T_, true, false); } \
    } else { \
      if (block_rows == 64) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 64, true, QUERY_T_, false, false); } \
      else if ((HD_) == 128 && use_d128_short_stream) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, true, QUERY_T_, false, true); } \
      else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, true, QUERY_T_, false, false); } \
    } \
  } else if (pv_accum_mode == 1) { \
    if (block_rows == 64) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 64, false, QUERY_T_, true, false); } \
    else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, false, QUERY_T_, true, false); } \
  } else { \
    if (block_rows == 64) { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 64, false, QUERY_T_, false, false); } \
    else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT(HD_, HND_, 128, false, QUERY_T_, false, false); } \
  }
#define SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_BR_DTYPE(BC_, HD_, BR_, QUERY_T_) \
  if (is_causal && pv_accum_mode == 1) { \
    if ((HD_) == 128 && use_d128_short_stream && (BR_) == 128) { \
      SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, true, true, true, false, false); \
    } else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, true, false, false, false, false); } \
  } else if (is_causal) { \
    if ((HD_) == 128 && use_d128_short_stream && (BR_) == 128) { \
      if (use_direct_stream_probs) { \
        SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, false, true, true, true, true); \
      } else { \
        SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, false, true, true, false, true); \
      } \
    } else if ((HD_) == 128 && (BR_) == 128 && use_d128_long_stream) { \
      SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, false, true, true, false, false); \
    } else if ((HD_) == 256 && (BR_) == 128 && use_f16_d256_s512_stream_direct) { \
      SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, false, true, true, true, true); \
    } else { SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, true, QUERY_T_, false, false, false, false, false); } \
  } else if (pv_accum_mode == 1) { \
    SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, false, QUERY_T_, true, false, false, false, false); \
  } else { \
    if ((HD_) == 64 && use_d64_noncausal_stream_direct) { \
      SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, false, QUERY_T_, false, true, true, true, false); \
    } else { \
      SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD(BC_, HD_, BR_, false, QUERY_T_, false, false, false, false, false); \
    } \
  }
#define SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE(BC_, HD_, QUERY_T_) \
  SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_BR_DTYPE(BC_, HD_, 128, QUERY_T_)
#define SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_DTYPE(HD_, QUERY_T_) \
  if constexpr ((HD_) == 128) { \
    if (q_len == 512) { \
      SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE(32, HD_, QUERY_T_); \
    } else { \
      SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE(64, HD_, QUERY_T_); \
    } \
  } else { \
    if ((HD_) == 256 && use_f16_d256_s512_stream_direct) { \
      SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE(32, HD_, QUERY_T_); \
    } else { \
      SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE(64, HD_, QUERY_T_); \
    } \
  }
#define SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_DTYPE(QUERY_T_) \
  if (use_static_nhd_no_tail) { \
    if (head_dim == 64) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_DTYPE(64, QUERY_T_); } \
    else if (head_dim == 128) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_DTYPE(128, QUERY_T_); } \
    else { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_DTYPE(256, QUERY_T_); } \
  } else if (hnd_contiguous) { \
    if (head_dim == 16) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(16, true, QUERY_T_); } \
    else if (head_dim == 64) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(64, true, QUERY_T_); } \
    else if (head_dim == 128) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(128, true, QUERY_T_); } \
    else { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(256, true, QUERY_T_); } \
  } else { \
    if (head_dim == 16) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(16, false, QUERY_T_); } \
    else if (head_dim == 64) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(64, false, QUERY_T_); } \
    else if (head_dim == 128) { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(128, false, QUERY_T_); } \
    else { SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND(256, false, QUERY_T_); } \
  }
  if (query.scalar_type() == torch::kFloat16) {
    SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_DTYPE(__half);
  } else {
    SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_DTYPE(__hip_bfloat16);
  }
#undef SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_DTYPE
#undef SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_DTYPE
#undef SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_DTYPE
#undef SAGEATTN_DISPATCH_RAWQ_F16_VALUE_STATIC_NHD_FOR_BC_BR_DTYPE
#undef SAGEATTN_DISPATCH_RAWQ_F16_VALUE_FOR_HND
#undef SAGEATTN_LAUNCH_RAWQ_F16_VALUE_STATIC_NHD
#undef SAGEATTN_LAUNCH_RAWQ_F16_VALUE_DEFAULT
#undef SAGEATTN_LAUNCH_RAWQ_F16_VALUE
  hip_kernel_launch_check();
  return output;
}

#endif // SAGEATTN_GFX12_BUILD_ATTN_F16
