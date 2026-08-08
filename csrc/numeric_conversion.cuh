/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS, https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
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
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#endif

#if !defined(__HIP_PLATFORM_AMD__)
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

__device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dest, uint32_t source) {
  uint16_t h0 = source & 0xFFFF;
  uint16_t h1 = (source >> 16) & 0xFFFF;
#if defined(__HIP_PLATFORM_AMD__)
  union {
    uint16_t bits;
    half value;
  } lo{h0}, hi{h1};
  dest[0] = __half2float(lo.value);
  dest[1] = __half2float(hi.value);
#else
  asm("cvt.f32.f16 %0, %1;" : "=f"(dest[0]) : "h"(h0));
  asm("cvt.f32.f16 %0, %1;" : "=f"(dest[1]) : "h"(h1));
#endif
}

__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
{
#if defined(__HIP_PLATFORM_AMD__)
  const auto lo = __hip_cvt_float2_to_fp8x2(
      make_float2(source0[0], source0[1]), __HIP_SATFINITE, __HIP_E4M3);
  const auto hi = __hip_cvt_float2_to_fp8x2(
      make_float2(source1[0], source1[1]), __HIP_SATFINITE, __HIP_E4M3);
  dest[0] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void floatx4_to_e5m2x4(uint32_t *dest, float *source0, float *source1)
{
#if defined(__HIP_PLATFORM_AMD__)
  const auto lo = __hip_cvt_float2_to_fp8x2(
      make_float2(source0[0], source0[1]), __HIP_SATFINITE, __HIP_E5M2);
  const auto hi = __hip_cvt_float2_to_fp8x2(
      make_float2(source1[0], source1[1]), __HIP_SATFINITE, __HIP_E5M2);
  dest[0] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source1[1]), "f"(source1[0]), "f"(source1[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void halfx4_to_e4m3x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
{
#if defined(__HIP_PLATFORM_AMD__)
  float s0[2];
  float s1[2];
  unpack_half2_from_uint32_to_float(s0, source0[0]);
  unpack_half2_from_uint32_to_float(s1, source1[0]);
  floatx4_to_e4m3x4(dest, s0, s1);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void halfx4_to_e5m2x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
{
#if defined(__HIP_PLATFORM_AMD__)
  float s0[2];
  float s1[2];
  unpack_half2_from_uint32_to_float(s0, source0[0]);
  unpack_half2_from_uint32_to_float(s1, source1[0]);
  floatx4_to_e5m2x4(dest, s0, s1);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n" \
      "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void e4m3x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
{
#if defined(__HIP_PLATFORM_AMD__)
  const auto lo = __hip_cvt_fp8x2_to_halfraw2(
      static_cast<__hip_fp8x2_storage_t>(source[0] & 0xFFFF), __HIP_E4M3);
  const auto hi = __hip_cvt_fp8x2_to_halfraw2(
      static_cast<__hip_fp8x2_storage_t>(source[0] >> 16), __HIP_E4M3);
  dest0[0] = static_cast<uint32_t>(lo.x.x) |
             (static_cast<uint32_t>(lo.y.x) << 16);
  dest1[0] = static_cast<uint32_t>(hi.x.x) |
             (static_cast<uint32_t>(hi.y.x) << 16);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void e5m2x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
{
#if defined(__HIP_PLATFORM_AMD__)
  const auto lo = __hip_cvt_fp8x2_to_halfraw2(
      static_cast<__hip_fp8x2_storage_t>(source[0] & 0xFFFF), __HIP_E5M2);
  const auto hi = __hip_cvt_fp8x2_to_halfraw2(
      static_cast<__hip_fp8x2_storage_t>(source[0] >> 16), __HIP_E5M2);
  dest0[0] = static_cast<uint32_t>(lo.x.x) |
             (static_cast<uint32_t>(lo.y.x) << 16);
  dest1[0] = static_cast<uint32_t>(hi.x.x) |
             (static_cast<uint32_t>(hi.y.x) << 16);
#elif defined(FP8_CAST_ENABLED)
  asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
#if defined(__HIP_PLATFORM_AMD__)
    const float clipped = fminf(127.0f, fmaxf(-128.0f, nearbyintf(x)));
    return static_cast<int8_t>(clipped);
#else
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
#endif
}
