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

#include "../cuda_ops.h"
#include "attn_cuda_sm89.h"

using sageattention::cuda_kernel;

TORCH_LIBRARY(sageattention_sm89, m)
{
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f32_attn, ""),
        cuda_kernel<&qk_int8_sv_f8_accum_f32_attn>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f32_attn_inst_buf, ""),
        cuda_kernel<&qk_int8_sv_f8_accum_f32_attn_inst_buf>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f16_attn_inst_buf, ""),
        cuda_kernel<&qk_int8_sv_f8_accum_f16_attn_inst_buf>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn, "Tensor value_scale, "),
        cuda_kernel<&qk_int8_sv_f8_accum_f32_fuse_v_scale_attn>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn,
                          "Tensor value_scale, Tensor value_mean, "),
        cuda_kernel<&qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf, "Tensor value_scale, "),
        cuda_kernel<&qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf>());
  m.def(SAGEATTN_QK_SCHEMA(qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf, "Tensor value_scale, "),
        cuda_kernel<&qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf>());
}
