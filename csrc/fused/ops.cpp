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
#include "fused.h"

using sageattention::cuda_kernel;

TORCH_LIBRARY(sageattention_fused, m)
{
  using at::Tensor;
  using BlockQuant = void (*)(Tensor, Tensor, Tensor, int, int);
  using ScaledBlockQuant = void (*)(Tensor, Tensor, Tensor, float, int, int);

  m.def("quant_per_block_int8_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
        "int block_size, int tensor_layout) -> ()",
        cuda_kernel<static_cast<BlockQuant>(&quant_per_block_int8_cuda)>());
  m.def("quant_per_block_int8_cuda.scaled(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
        "float sm_scale, int block_size, int tensor_layout) -> ()",
        cuda_kernel<static_cast<ScaledBlockQuant>(&quant_per_block_int8_cuda)>());
  m.def("quant_per_block_int8_fuse_sub_mean_cuda(Tensor input, Tensor mean, Tensor(a!) output, "
        "Tensor(b!) scale, int block_size, int tensor_layout) -> ()",
        cuda_kernel<&quant_per_block_int8_fuse_sub_mean_cuda>());
  m.def("quant_per_warp_int8_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
        "int block_size, int warp_block_size, int tensor_layout) -> ()",
        cuda_kernel<&quant_per_warp_int8_cuda>());
  m.def("sub_mean_cuda(Tensor input, Tensor mean, Tensor(a!) output, int tensor_layout) -> ()",
        cuda_kernel<&sub_mean_cuda>());
  m.def("transpose_pad_permute_cuda(Tensor input, Tensor(a!) output, int tensor_layout) -> ()",
        cuda_kernel<&transpose_pad_permute_cuda>());
  m.def("scale_fuse_quant_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
        "int num_tokens, float scale_max, int tensor_layout) -> ()", cuda_kernel<&scale_fuse_quant_cuda>());
  m.def("mean_scale_fuse_quant_cuda(Tensor input, Tensor(a!) output, Tensor(b!) mean, Tensor(c!) scale, "
        "int num_tokens, float scale_max, int tensor_layout) -> ()", cuda_kernel<&mean_scale_fuse_quant_cuda>());
}
