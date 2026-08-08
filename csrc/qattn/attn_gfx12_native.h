/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <vector>

#include <torch/extension.h>

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
    int64_t valid_kv_len = 0,
    int value_transposed_hnd = -1,
    int pv_accum_mode = -1);

torch::Tensor qk_rawq_int8_sv_f8_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len = 0,
    int value_transposed_hnd = -1,
    int key_hnd_layout = 0);

torch::Tensor qk_rawq_int8_sv_f16_native_attn_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    float sm_scale,
    int64_t valid_kv_len = 0,
    int pv_accum_mode = -1);

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
    int64_t valid_kv_len = 0);

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
    int64_t valid_kv_len = 0,
    int value_transposed_hnd = -1,
    int key_hnd_layout = 0);

torch::Tensor sage_fp8_nhd_short_mha_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int is_causal,
    float sm_scale,
    double scale_max);

torch::Tensor qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int is_causal,
    int value_is_fp8,
    int use_raw_f16_value,
    float sm_scale,
    int64_t valid_kv_len = 0,
    int pv_accum_mode = -1);

std::vector<torch::Tensor> quant_q_nhd_per_warp_gfx12(torch::Tensor query);

torch::Tensor transpose_value_fp8_hnd_gfx12(torch::Tensor value);

torch::Tensor transpose_value_fp8_scaled_hnd_gfx12(
    torch::Tensor value,
    torch::Tensor value_scale);

std::vector<torch::Tensor> fp8_value_nhd_short_gfx12(
    torch::Tensor value,
    double scale_max);

torch::Tensor mean_nhd_gfx12(torch::Tensor input);

torch::Tensor mean_nhd_d64_seq32_gfx12(torch::Tensor input);

torch::Tensor mean_hnd_gfx12(torch::Tensor input);

std::vector<torch::Tensor> prepare_qkv_hnd_smooth_f16_gfx12(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean);

std::vector<torch::Tensor> mean_and_fp8_value_nhd_short_gfx12(
    torch::Tensor key,
    torch::Tensor value,
    double scale_max);

torch::Tensor transpose_value_f16_hnd_gfx12(torch::Tensor value);

torch::Tensor convert_f16_to_bf16_gfx12(torch::Tensor input);
