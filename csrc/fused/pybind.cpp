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

#include <torch/extension.h>
#include <cuda_fp16.h>
#include "fused.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("quant_per_block_int8_cuda", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, float, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_cuda", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_fuse_sub_mean_cuda", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>(&quant_per_block_int8_fuse_sub_mean_cuda), "quant_per_block_int8_fuse_sub_mean_cuda");
}