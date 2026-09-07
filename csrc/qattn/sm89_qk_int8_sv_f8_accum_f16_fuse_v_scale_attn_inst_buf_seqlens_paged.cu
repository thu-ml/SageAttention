#include "attn_cuda_sm89.h"
#include "qk_int_sv_f8_cuda_sm89.cuh"

// Prefill/decode paged-K + paged-V seqlens kernel.
//
//   query      : dense int8  HND [B, num_qo_heads, qo_len, head_dim]
//   key_cache  : paged int8  [num_blocks, page_size, num_kv_heads, head_dim]
//   value_cache: paged fp8   [num_blocks, page_size, num_kv_heads, head_dim]
//                (logical token order; kernel applies fused.cu seq permute)
//   block_table: [B, max_pages] int32
//   seq_lens   : [B] int32
//   kv_len     : padded KV capacity used for scale-tensor indexing
//   is_causal  : 1 → causal mask (requires Q aligned to K start, typically
//                qo_len == kv_len_actual for full prefill)
//
// Launch on PyTorch's current stream so CUDA Graph capture works.
torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_seqlens_paged(
                    torch::Tensor query,
                    torch::Tensor key_cache,
                    torch::Tensor value_cache,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    torch::Tensor seq_lens,
                    torch::Tensor block_table,
                    int block_size,
                    int kv_len,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(block_table);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key_cache);
  CHECK_LASTDIM_CONTIGUOUS(value_cache);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_CONTIGUOUS(seq_lens);
  CHECK_CONTIGUOUS(block_table);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key_cache, torch::kInt8);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DTYPE(value_scale, torch::kFloat32);
  CHECK_DTYPE(seq_lens, torch::kInt32);
  CHECK_DTYPE(block_table, torch::kInt32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key_cache, 4);
  CHECK_DIMS(value_cache, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);
  CHECK_DIMS(seq_lens, 1);
  CHECK_DIMS(block_table, 2);

  const int batch_size = query.size(0);
  const int num_qo_heads = query.size(1);
  const int qo_len = query.size(2);
  const int head_dim = query.size(3);
  const int num_kv_heads = key_cache.size(2);

  TORCH_CHECK(block_size > 0, "block_size must be > 0");
  TORCH_CHECK(kv_len > 0, "kv_len must be > 0");
  TORCH_CHECK(key_cache.size(1) == block_size,
              "key_cache.size(1) must equal block_size");
  TORCH_CHECK(value_cache.size(1) == block_size,
              "value_cache.size(1) must equal block_size");
  TORCH_CHECK(key_cache.size(2) == value_cache.size(2) &&
                  key_cache.size(3) == value_cache.size(3),
              "key/value cache shape mismatch");
  TORCH_CHECK(key_cache.size(3) == head_dim, "cache head_dim mismatch");
  TORCH_CHECK(seq_lens.size(0) >= batch_size,
              "seq_lens must have at least batch_size entries");
  TORCH_CHECK(block_table.size(0) >= batch_size,
              "block_table must have at least batch_size rows");
  TORCH_CHECK(block_table.size(1) * block_size >= kv_len,
              "block_table capacity must cover kv_len");

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads
            << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  torch::Tensor lse = torch::empty({0});
  if (return_lse)
  {
    lse = torch::empty({batch_size, num_qo_heads, qo_len},
                       query.options().dtype(torch::kFloat32));
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;
  auto output_dtype = output.scalar_type();

  const int stride_bz_q = query.stride(0);
  const int stride_seq_q = query.stride(2);
  const int stride_h_q = query.stride(1);
  const int stride_bz_o = output.stride(0);
  const int stride_seq_o = output.stride(2);
  const int stride_h_o = output.stride(1);

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {

            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = 32;
            constexpr int WARP_K = 64;

            constexpr MaskMode mask_mode =
                IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

            if constexpr (QK_QUANT_GRAN ==
                          static_cast<int>(QuantGranularity::kPerWarp))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads,
                          div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads,
                          div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN ==
                               static_cast<int>(QuantGranularity::kPerThread))
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads,
                          div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads,
                          div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(
                  QK_QUANT_GRAN ==
                          static_cast<int>(QuantGranularity::kPerWarp) ||
                      QK_QUANT_GRAN ==
                          static_cast<int>(QuantGranularity::kPerThread),
                  "Unsupported quantization granularity");
            }

            CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);
            CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);

            // Q + K + V + V_stage (paged token-major permute scratch)
            size_t smem_max = std::max(
                CTA_Q * HEAD_DIM * sizeof(int8_t) +
                    CTA_K * HEAD_DIM * sizeof(int8_t) +
                    CTA_K * HEAD_DIM * sizeof(int8_t) +
                    CTA_K * HEAD_DIM * sizeof(int8_t),
                CTA_Q * HEAD_DIM * sizeof(half));

            auto kernel_func = qk_int_sv_f8_attn_kernel<
                CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
                static_cast<QuantGranularity>(QK_QUANT_GRAN),
                static_cast<QuantGranularity>(QK_QUANT_GRAN), float, true,
                DTypeOut, ComputeUnit::kCudaCore, mask_mode, RETURN_LSE, true,
                false, true>;

            cudaFuncSetAttribute(kernel_func,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_max);

            dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
            dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

            // Launch on PyTorch's current stream so CUDA Graph capture works.
            kernel_func<<<grid, block, smem_max,
                          at::cuda::getCurrentCUDAStream()>>>(
                query.data_ptr<int8_t>(),
                key_cache.data_ptr<int8_t>(),
                reinterpret_cast<int8_t*>(value_cache.data_ptr()),
                reinterpret_cast<DTypeOut*>(output.data_ptr()),
                (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr())
                             : nullptr,
                reinterpret_cast<float*>(query_scale.data_ptr()),
                reinterpret_cast<float*>(key_scale.data_ptr()),
                reinterpret_cast<float*>(value_scale.data_ptr()),
                nullptr,
                qo_len,
                kv_len,
                num_kv_groups,
                stride_bz_q, stride_seq_q, stride_h_q,
                0, 0, 0,
                0, 0, 0,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale, seq_lens.data_ptr<int>(),
                block_table.data_ptr<int>(),
                static_cast<uint32_t>(block_size),
                static_cast<uint32_t>(block_table.stride(0)));
          });
        });
      });
    });
  });

  return lse;
}
