#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, uint32_t qk_quant_gran, typename DTypePVAccum, bool use_inst_buffer, typename DTypeOut, bool is_causal, bool return_lse, bool fuse_v_scale>
void SageAttentionSM80Dispatched(
  int8_t* Q, int8_t* K, half* V, DTypeOut* O, float* Lse,
  float* Q_scale, float* K_scale, DTypeOut* V_mean,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale);

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, uint32_t qk_quant_gran, typename DTypePVAccum, bool use_inst_buffer, typename DTypeOut, bool is_causal, bool return_lse, bool fuse_v_scale, bool fuse_v_mean>
void SageAttentionSM89Dispatched(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, DTypeOut* O, float* Lse,
  float* Q_scale, float* K_scale, float* V_scale, float* V_mean,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale);

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, uint32_t qk_quant_gran, typename DTypeOut, bool is_causal, bool return_lse, bool fuse_v_scale>
void SageAttentionSM90Dispatched(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, DTypeOut* O, float* Lse,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t padded_kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale);