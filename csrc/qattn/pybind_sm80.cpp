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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_cuda_sm80.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_accum_f32_attn", &qk_int8_sv_f16_accum_f32_attn, "QK int8 sv f16 accum f32 attn per warp");
  m.def("qk_int8_sv_f16_accum_f16_attn", &qk_int8_sv_f16_accum_f16_attn, "QK int8 sv f16 accum f16 attn per warp");
  m.def("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", &qk_int8_sv_f16_accum_f16_fuse_v_mean_attn, "QK int8 sv f16 accum f16 attn per warp fuse v mean");

  m.def("qk_int8_sv_f16_accum_f16_attn_inst_buf", &qk_int8_sv_f16_accum_f16_attn_inst_buf, "QK int8 sv f16 accum f16 attn inst buf");
}