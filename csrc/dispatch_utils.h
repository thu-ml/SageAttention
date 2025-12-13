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

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else if (head_dim == 256) {                                 \
    constexpr int HEAD_DIM = 256;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)              \
  if (is_causal == 1) {                                         \
    constexpr bool IS_CAUSAL = true;                            \
    __VA_ARGS__                                                 \
  } else if (is_causal == 0) {                                  \
    constexpr bool IS_CAUSAL = false;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported causal mode: " << int(is_causal);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)              \
  if (qk_quant_gran == 2) {                                         \
    constexpr int QK_QUANT_GRAN = 2;                            \
    __VA_ARGS__                                                 \
  } else if (qk_quant_gran == 3) {                                  \
    constexpr int QK_QUANT_GRAN = 3;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported qk_quant_gran: " << int(qk_quant_gran);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)             \
  if (return_lse == 1) {                                         \
    constexpr bool RETURN_LSE = true;                            \
    __VA_ARGS__                                                  \
  } else if (return_lse == 0) {                                  \
    constexpr bool RETURN_LSE = false;                           \
    __VA_ARGS__                                                  \
  }  else {                                                      \
    std::ostringstream err_msg;                                  \
    err_msg << "Unsupported causal mode: " << int(return_lse);   \
    throw std::invalid_argument(err_msg.str());                  \
  }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                \
  if (pytorch_dtype == at::ScalarType::Half) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (pytorch_dtype == at::ScalarType::BFloat16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;    \
    TORCH_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
  if (block_size == 64) {                                       \
    constexpr int BLOCK_SIZE = 64;                              \
    __VA_ARGS__                                                 \
  } else if (block_size == 128) {                               \
    constexpr int BLOCK_SIZE = 128;                             \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported block_size " << int(block_size);    \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, ...)  \
  if (warp_block_size == 16) {                                           \
    constexpr int WARP_BLOCK_SIZE = 16;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 32) {                                    \
    constexpr int WARP_BLOCK_SIZE = 32;                                  \
    __VA_ARGS__                                                          \
  }  else {                                                              \
    std::ostringstream err_msg;                                          \
    err_msg << "Unsupported warp_block_size " << int(warp_block_size);   \
    throw std::invalid_argument(err_msg.str());                          \
  }
