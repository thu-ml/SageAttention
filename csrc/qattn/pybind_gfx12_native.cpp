/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "attn_gfx12_native.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_d64_native_attn", &qk_int8_sv_f16_d64_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("query_scale"), py::arg("key_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0,
        py::arg("value_transposed_hnd") = -1, py::arg("pv_accum_mode") = -1);
  m.def("qk_rawq_int8_sv_f8_native_attn", &qk_rawq_int8_sv_f8_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("key_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0,
        py::arg("value_transposed_hnd") = -1, py::arg("key_hnd_layout") = 0);
  m.def("qk_rawq_int8_sv_f16_native_attn", &qk_rawq_int8_sv_f16_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("key_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0,
        py::arg("pv_accum_mode") = -1);
  m.def("qk_int8_sv_f8_scaled_native_attn", &qk_int8_sv_f8_scaled_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("query_scale"), py::arg("key_scale"), py::arg("value_scale"),
        py::arg("tensor_layout"), py::arg("is_causal"), py::arg("sm_scale"),
        py::arg("valid_kv_len") = 0);
  m.def("qk_rawq_int8_sv_f8_scaled_native_attn", &qk_rawq_int8_sv_f8_scaled_native_attn_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("key_scale"), py::arg("value_scale"), py::arg("tensor_layout"),
        py::arg("is_causal"), py::arg("sm_scale"), py::arg("valid_kv_len") = 0,
        py::arg("value_transposed_hnd") = -1, py::arg("key_hnd_layout") = 0);
  m.def("sage_fp8_nhd_short_mha", &sage_fp8_nhd_short_mha_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("is_causal"),
        py::arg("sm_scale"), py::arg("scale_max"));
  m.def("qk_int8_sv_f16_d64_prepare_attn_hnd", &qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("is_causal"),
        py::arg("value_is_fp8"), py::arg("use_raw_f16_value"), py::arg("sm_scale"),
        py::arg("valid_kv_len") = 0, py::arg("pv_accum_mode") = -1);
  m.def("quant_q_nhd_per_warp", &quant_q_nhd_per_warp_gfx12);
  m.def("transpose_value_fp8_hnd", &transpose_value_fp8_hnd_gfx12);
  m.def("transpose_value_fp8_scaled_hnd", &transpose_value_fp8_scaled_hnd_gfx12);
  m.def("fp8_value_nhd_short", &fp8_value_nhd_short_gfx12,
        py::arg("value"), py::arg("scale_max"));
  m.def("mean_nhd", &mean_nhd_gfx12);
  m.def("mean_nhd_d64_seq32", &mean_nhd_d64_seq32_gfx12);
  m.def("mean_hnd", &mean_hnd_gfx12);
  m.def("prepare_qkv_hnd_smooth_f16", &prepare_qkv_hnd_smooth_f16_gfx12,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("key_mean"));
  m.def("mean_and_fp8_value_nhd_short", &mean_and_fp8_value_nhd_short_gfx12,
        py::arg("key"), py::arg("value"), py::arg("scale_max"));
  m.def("transpose_value_f16_hnd", &transpose_value_f16_hnd_gfx12);
  m.def("convert_f16_to_bf16", &convert_f16_to_bf16_gfx12);
}
