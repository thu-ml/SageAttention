/*
 * Paged KV quantization kernels for SageAttention (SM89/SM120 decode path).
 *
 * Cache layout (matches vLLM SAGE slab): [num_blocks, page_size, H, D]
 * Logical token t -> physical (block_table[b, t/page_size], t % page_size).
 *
 * Outputs stay in the same paged layout so the seqlens_paged attention kernel
 * can read them without a dense gather.
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../dispatch_utils.h"
#include "../utils.cuh"
#include "../reduction_utils.cuh"
#include "../numeric_conversion.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

#include "fused.h"

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T val)
{
  if constexpr (std::is_same<T, half>::value)
    return __half2float(val);
  else
    return __bfloat162float(val);
}

template <typename T>
__device__ __forceinline__ const T *paged_token_ptr(
    const T *cache, const int *block_table_row, uint32_t token,
    uint32_t page_size, uint32_t page_elems, uint32_t head_offset)
{
  const uint32_t page_id = token / page_size;
  const uint32_t offset = token % page_size;
  const uint32_t block_id = static_cast<uint32_t>(block_table_row[page_id]);
  return cache +
         (static_cast<uint64_t>(block_id) * page_size + offset) * page_elems +
         head_offset;
}

template <typename T>
__device__ __forceinline__ T *paged_token_ptr_mut(
    T *cache, const int *block_table_row, uint32_t token,
    uint32_t page_size, uint32_t page_elems, uint32_t head_offset)
{
  return const_cast<T *>(paged_token_ptr<T>(
      cache, block_table_row, token, page_size, page_elems, head_offset));
}

// Mean of K over logical tokens: out [B, H, D]
// Grid (H, B, D): one block per channel; threads stride over tokens + blockReduceSum.
template <typename T>
__global__ void MeanKPagedKernel(
    const T *__restrict__ k_cache, float *__restrict__ mean,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const uint32_t page_size, const uint32_t num_kv_heads, const uint32_t head_dim,
    const uint32_t block_table_stride, const uint32_t stride_bz_mean,
    const uint32_t stride_h_mean)
{
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t d = blockIdx.z;
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;

  const int sl = seq_lens[batch_id];
  const uint32_t kv_actual = (sl > 0) ? static_cast<uint32_t>(sl) : 0u;
  const int *bt = block_table + batch_id * block_table_stride;
  const uint32_t page_elems = num_kv_heads * head_dim;
  const uint32_t head_off = head_id * head_dim;

  float sum = 0.f;
#pragma unroll 4
  for (uint32_t t = tid; t < kv_actual; t += nthreads)
  {
    const T *ptr =
        paged_token_ptr<T>(k_cache, bt, t, page_size, page_elems, head_off);
    sum += to_float(ptr[d]);
  }

  sum = vllm::blockReduceSum(sum);
  if (tid == 0)
  {
    const float denom = fmaxf(static_cast<float>(sl), 1.f);
    mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d] = sum / denom;
  }
}

// Per-block INT8 quant of K (BLKK tokens per scale), optional subtract mean.
// Grid: (ceil(kv_cap / BLKK), H, B)  Block: same as dense QuantInt8Kernel.
template <uint32_t HEAD_DIM, uint32_t BLKK, uint32_t num_pack_per_thread,
          bool sub_mean, typename T>
__global__ void QuantKPagedInt8Kernel(
    const T *__restrict__ k_cache, const float *__restrict__ mean,
    int8_t *__restrict__ k_out, float *__restrict__ scale,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const uint32_t page_size, const uint32_t num_kv_heads, const uint32_t kv_cap,
    const uint32_t block_table_stride, const uint32_t stride_bz_scale,
    const uint32_t stride_h_scale, const uint32_t stride_bz_mean,
    const uint32_t stride_h_mean)
{
  constexpr uint32_t pack = 8;
  constexpr uint32_t threads_per_token = HEAD_DIM / pack;
  constexpr uint32_t stride_tok = BLKK / num_pack_per_thread;

  const uint32_t bx = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t tid = threadIdx.x;

  const uint32_t token0 = bx * BLKK + tid / threads_per_token;
  const uint32_t d0 = (tid % threads_per_token) * pack;
  const int sl = seq_lens[batch_id];
  const uint32_t kv_actual = (sl > 0) ? static_cast<uint32_t>(sl) : 0u;
  if (bx * BLKK >= kv_actual)
    return;
  const int *bt = block_table + batch_id * block_table_stride;
  const uint32_t page_elems = num_kv_heads * HEAD_DIM;
  const uint32_t head_off = head_id * HEAD_DIM;

  float mean_v[8];
  if constexpr (sub_mean)
  {
    const float *mp =
        mean + batch_id * stride_bz_mean + head_id * stride_h_mean + d0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      mean_v[j] = mp[j];
  }

  float vals[num_pack_per_thread][8];
  float amax = 1e-7f;

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; ++i)
  {
    const uint32_t tok = token0 + i * stride_tok;
    if (tok < kv_cap && static_cast<int>(tok) < sl)
    {
      const T *src =
          paged_token_ptr<T>(k_cache, bt, tok, page_size, page_elems, head_off);
#pragma unroll
      for (int j = 0; j < 8; ++j)
      {
        float x = to_float(src[d0 + j]);
        if constexpr (sub_mean)
          x -= mean_v[j];
        vals[i][j] = x;
        amax = fmaxf(amax, fabsf(x));
      }
    }
    else
    {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        vals[i][j] = 0.f;
    }
  }

  __shared__ float s_amax;
  const float block_amax = vllm::blockReduceMax(amax);
  if (tid == 0)
  {
    s_amax = block_amax;
    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + bx] =
        s_amax / 127.f;
  }
  __syncthreads();
  const float inv = 127.f / s_amax;

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; ++i)
  {
    const uint32_t tok = token0 + i * stride_tok;
    if (tok >= kv_cap)
      continue;
    int8_t *dst = paged_token_ptr_mut<int8_t>(
        k_out, bt, tok, page_size, page_elems, head_off);
    if (static_cast<int>(tok) < sl)
    {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        dst[d0 + j] = float_to_int8_rn(vals[i][j] * inv);
    }
    else
    {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        dst[d0 + j] = 0;
    }
  }
}

// Per-channel FP8 quant of V — coalesced along D, tiled over tokens for occupancy.
//
// Pass 1 (amax): grid (H, B, n_tiles), block (HEAD_DIM)
//   each block covers TILE tokens; atomicMax into amax[B,H,D]
// Pass 2 (apply): same grid; scale = amax/scale_max, write fp8
template <typename T, uint32_t HEAD_DIM, uint32_t TILE>
__global__ void QuantVPagedAmaxKernel(
    const T *__restrict__ v_cache, float *__restrict__ amax,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    const uint32_t page_size, const uint32_t num_kv_heads,
    const uint32_t block_table_stride, const uint32_t stride_bz_amax,
    const uint32_t stride_h_amax)
{
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t tile = blockIdx.z;
  const uint32_t d = threadIdx.x;
  if (d >= HEAD_DIM)
    return;

  const int sl = seq_lens[batch_id];
  const uint32_t kv_actual = (sl > 0) ? static_cast<uint32_t>(sl) : 0u;
  const uint32_t t0 = tile * TILE;
  if (t0 >= kv_actual)
    return;

  const uint32_t t1 = min(t0 + TILE, kv_actual);
  const int *bt = block_table + batch_id * block_table_stride;
  const uint32_t page_elems = num_kv_heads * HEAD_DIM;
  const uint32_t head_off = head_id * HEAD_DIM;

  float local = 1e-7f;
#pragma unroll 4
  for (uint32_t t = t0; t < t1; ++t)
  {
    const T *src =
        paged_token_ptr<T>(v_cache, bt, t, page_size, page_elems, head_off);
    local = fmaxf(local, fabsf(to_float(__ldg(src + d))));
  }

  float *dst = amax + batch_id * stride_bz_amax + head_id * stride_h_amax + d;
  // atomic max on float via CAS
  int *addr = reinterpret_cast<int *>(dst);
  int old = *addr;
  while (true)
  {
    float old_f = __int_as_float(old);
    if (old_f >= local)
      break;
    int assumed = old;
    old = atomicCAS(addr, assumed, __float_as_int(local));
    if (old == assumed)
      break;
  }
}

template <typename T, uint32_t HEAD_DIM, uint32_t TILE>
__global__ void QuantVPagedApplyKernel(
    const T *__restrict__ v_cache, int8_t *__restrict__ v_out,
    const float *__restrict__ amax, const int *__restrict__ block_table,
    const int *__restrict__ seq_lens, const uint32_t page_size,
    const uint32_t num_kv_heads, const uint32_t block_table_stride,
    const float scale_max, const uint32_t stride_bz_amax,
    const uint32_t stride_h_amax)
{
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t tile = blockIdx.z;
  const uint32_t d = threadIdx.x;
  if (d >= HEAD_DIM)
    return;

  const int sl = seq_lens[batch_id];
  const uint32_t kv_actual = (sl > 0) ? static_cast<uint32_t>(sl) : 0u;
  const uint32_t t0 = tile * TILE;
  if (t0 >= kv_actual)
    return;

  // amax buffer is still raw amax here; host converts to scale afterwards.
  const float a =
      amax[batch_id * stride_bz_amax + head_id * stride_h_amax + d];
  const float inv = scale_max / a;

  const uint32_t t1 = min(t0 + TILE, kv_actual);
  const int *bt = block_table + batch_id * block_table_stride;
  const uint32_t page_elems = num_kv_heads * HEAD_DIM;
  const uint32_t head_off = head_id * HEAD_DIM;

#pragma unroll 4
  for (uint32_t t = t0; t < t1; ++t)
  {
    int8_t *dst = paged_token_ptr_mut<int8_t>(
        v_out, bt, t, page_size, page_elems, head_off);
    const T *src =
        paged_token_ptr<T>(v_cache, bt, t, page_size, page_elems, head_off);
    float x = to_float(__ldg(src + d)) * inv;
    float tmp[4] = {x, 0.f, 0.f, 0.f};
    uint32_t packed;
    floatx4_to_e4m3x4(&packed, tmp, tmp + 2);
    dst[d] = reinterpret_cast<int8_t *>(&packed)[0];
  }
}

} // namespace

void mean_k_paged_cuda(
    torch::Tensor k_cache, torch::Tensor mean, torch::Tensor block_table,
    torch::Tensor seq_lens, int page_size)
{
  CHECK_CUDA(k_cache);
  CHECK_CUDA(mean);
  CHECK_CUDA(block_table);
  CHECK_CUDA(seq_lens);
  CHECK_DIMS(k_cache, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(block_table, 2);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(seq_lens);

  const int num_kv_heads = k_cache.size(2);
  const int head_dim = k_cache.size(3);
  const int batch = block_table.size(0);
  TORCH_CHECK(k_cache.size(1) == page_size);

  // One block per (head, batch, d); 256 threads stride over tokens.
  dim3 grid(num_kv_heads, batch, head_dim);
  dim3 block(256);

  auto dt = k_cache.scalar_type();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(dt, c_type, {
    MeanKPagedKernel<c_type><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<c_type *>(k_cache.data_ptr()),
        mean.data_ptr<float>(), block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(), page_size, num_kv_heads, head_dim,
        block_table.stride(0), mean.stride(0), mean.stride(1));
  });
}

void quant_k_paged_int8_cuda(
    torch::Tensor k_cache, torch::Tensor k_out, torch::Tensor scale,
    torch::Tensor block_table, torch::Tensor seq_lens, int page_size,
    int kv_cap, int quant_block, c10::optional<torch::Tensor> mean)
{
  CHECK_CUDA(k_cache);
  CHECK_CUDA(k_out);
  CHECK_CUDA(scale);
  CHECK_CUDA(block_table);
  CHECK_CUDA(seq_lens);
  CHECK_DTYPE(k_out, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);
  TORCH_CHECK(quant_block == 64, "only BLKK=64 supported");
  TORCH_CHECK(k_cache.sizes() == k_out.sizes());
  TORCH_CHECK(k_cache.size(1) == page_size);

  const int num_kv_heads = k_cache.size(2);
  const int head_dim = k_cache.size(3);
  const int batch = block_table.size(0);
  const bool sub_mean = mean.has_value();
  if (sub_mean)
  {
    const torch::Tensor &mean_t = *mean;
    CHECK_CUDA(mean_t);
    CHECK_SHAPE(mean_t, batch, num_kv_heads, head_dim);
  }

  CHECK_SHAPE(scale, batch, num_kv_heads, (kv_cap + quant_block - 1) / quant_block);

  auto dt = k_cache.scalar_type();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(dt, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr int BLKK = 64;
      constexpr int num_pack_per_thread =
          (BLKK * (HEAD_DIM / 8) + 1023) / 1024;
      dim3 grid((kv_cap + BLKK - 1) / BLKK, num_kv_heads, batch);
      dim3 block(BLKK * (HEAD_DIM / 8) / num_pack_per_thread);
      const float *mean_ptr =
          sub_mean ? mean->data_ptr<float>() : nullptr;
      const uint32_t stride_bz_mean =
          sub_mean ? static_cast<uint32_t>(mean->stride(0)) : 0u;
      const uint32_t stride_h_mean =
          sub_mean ? static_cast<uint32_t>(mean->stride(1)) : 0u;
      if (sub_mean)
      {
        QuantKPagedInt8Kernel<HEAD_DIM, BLKK, num_pack_per_thread, true, c_type>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<c_type *>(k_cache.data_ptr()), mean_ptr,
                k_out.data_ptr<int8_t>(), scale.data_ptr<float>(),
                block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
                page_size, num_kv_heads, kv_cap, block_table.stride(0),
                scale.stride(0), scale.stride(1), stride_bz_mean,
                stride_h_mean);
      }
      else
      {
        QuantKPagedInt8Kernel<HEAD_DIM, BLKK, num_pack_per_thread, false, c_type>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<c_type *>(k_cache.data_ptr()), mean_ptr,
                k_out.data_ptr<int8_t>(), scale.data_ptr<float>(),
                block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
                page_size, num_kv_heads, kv_cap, block_table.stride(0),
                scale.stride(0), scale.stride(1), stride_bz_mean,
                stride_h_mean);
      }
    });
  });
}

void quant_v_paged_fp8_cuda(
    torch::Tensor v_cache, torch::Tensor v_out, torch::Tensor scale,
    torch::Tensor block_table, torch::Tensor seq_lens, int page_size,
    int kv_cap, float scale_max)
{
  CHECK_CUDA(v_cache);
  CHECK_CUDA(v_out);
  CHECK_CUDA(scale);
  CHECK_CUDA(block_table);
  CHECK_CUDA(seq_lens);
  CHECK_DTYPE(scale, torch::kFloat);
  TORCH_CHECK(v_cache.sizes() == v_out.sizes());
  TORCH_CHECK(v_cache.size(1) == page_size);

  const int num_kv_heads = v_cache.size(2);
  const int head_dim = v_cache.size(3);
  const int batch = block_table.size(0);
  CHECK_SHAPE(scale, batch, num_kv_heads, head_dim);

  // Reuse scale buffer as amax workspace for pass 1 (init to 1e-7).
  scale.fill_(1e-7f);

  constexpr uint32_t TILE = 64;
  const int n_tiles = (kv_cap + static_cast<int>(TILE) - 1) / static_cast<int>(TILE);
  dim3 grid(num_kv_heads, batch, n_tiles);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto dt = v_cache.scalar_type();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(dt, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(HEAD_DIM);
      QuantVPagedAmaxKernel<c_type, HEAD_DIM, TILE><<<grid, block, 0, stream>>>(
          reinterpret_cast<c_type *>(v_cache.data_ptr()), scale.data_ptr<float>(),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), page_size,
          num_kv_heads, block_table.stride(0), scale.stride(0), scale.stride(1));
      QuantVPagedApplyKernel<c_type, HEAD_DIM, TILE><<<grid, block, 0, stream>>>(
          reinterpret_cast<c_type *>(v_cache.data_ptr()),
          reinterpret_cast<int8_t *>(v_out.data_ptr()), scale.data_ptr<float>(),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), page_size,
          num_kv_heads, block_table.stride(0), scale_max, scale.stride(0),
          scale.stride(1));
    });
  });
  // Convert in-place amax -> scale.
  scale.div_(scale_max);
}
