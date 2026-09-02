/*
 * Copyright (c) 2025 by SageAttention team.
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

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "params.h"
#include "launch.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t unpadded_seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor delta_s,
                      at::Tensor out,
                      const at::Tensor sfq,
                      const at::Tensor sfk,
                      const at::Tensor sfv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool per_block_mean,
                      bool is_bf16,
                      bool is_mxfp8,
                      bool seqlenq_ngroups_swapped=false) {

    // Reset the parameters
    params = {};
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.delta_s_ptr = delta_s.data_ptr();
    params.sfq_ptr = sfq.data_ptr();
    params.sfk_ptr = sfk.data_ptr();
    params.sfv_ptr = sfv.data_ptr();
    
    // All stride are in elements, not bytes.
    const int qkv_stride_factor = is_mxfp8 ? 1 : 2;
    params.q_row_stride = q.stride(-2) * qkv_stride_factor;
    params.k_row_stride = k.stride(-2) * qkv_stride_factor;
    params.v_row_stride = v.stride(-2) * qkv_stride_factor;
    params.q_head_stride = q.stride(-3) * qkv_stride_factor;
    params.k_head_stride = k.stride(-3) * qkv_stride_factor;
    params.v_head_stride = v.stride(-3) * qkv_stride_factor; // for packed q k v

    params.ds_row_stride = delta_s.stride(-2);
    params.ds_head_stride = delta_s.stride(-3);
    
    params.sfq_row_stride = sfq.stride(-2);
    params.sfk_row_stride = sfk.stride(-2);
    params.sfv_row_stride = sfv.stride(-2);
    params.sfq_head_stride = sfq.stride(-3);
    params.sfk_head_stride = sfk.stride(-3);
    params.sfv_head_stride = sfv.stride(-3);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-2);
    params.o_head_stride = out.stride(-3);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0) * qkv_stride_factor;
        params.k_batch_stride = k.stride(0) * qkv_stride_factor;
        params.v_batch_stride = v.stride(0) * qkv_stride_factor;
        params.ds_batch_stride = delta_s.stride(0);
        params.sfq_batch_stride = sfq.stride(0);
        params.sfk_batch_stride = sfk.stride(0);
        params.sfv_batch_stride = sfv.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.unpadded_seqlen_k = unpadded_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.head_divmod = cutlass::FastDivmod(int(h));

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.per_block_mean = per_block_mean;
    if (per_block_mean) {
        params.seqlen_s = seqlen_q;
    } else {
        params.seqlen_s = 128; // size of BLOCK_M
    }
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = is_bf16;
    params.is_e4m3 = is_mxfp8;
    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
}

template<bool IsBF16, bool IsMXFP8>
void run_mha_fwd_dispatch_dtype(Flash_fwd_params &params, cudaStream_t stream) {
    using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
    using QKVType = std::conditional_t<IsMXFP8, cutlass::float_e4m3_t, cutlass::nv_float4_t<cutlass::float_e2m1_t>>;
    if constexpr (IsMXFP8) {
        TORCH_CHECK(params.d == 128, "Tensor-Core MXFP8 path currently supports head_dim=128");
        run_mha_fwd_<QKVType, 128, OType>(params, stream);
    } else {
        if (params.d == 64) {
            run_mha_fwd_<QKVType, 64, OType>(params, stream);
        } else if (params.d == 128) {
            run_mha_fwd_<QKVType, 128, OType>(params, stream);
        }
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    BOOL_SWITCH(params.is_bf16, IsBF16, ([&] {
        if (params.is_e4m3) {
            run_mha_fwd_dispatch_dtype<IsBF16, true>(params, stream);
        } else {
            run_mha_fwd_dispatch_dtype<IsBF16, false>(params, stream);
        }
    }));
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x (head_size // 2)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &sfq,
        const at::Tensor &sfk,
        const at::Tensor &sfv,
        const at::Tensor &delta_s,
        int unpadded_k,
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal, 
        bool per_block_mean,
        bool is_bf16,
        bool is_mxfp8=false
    ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm120 = dprops->major == 12 && dprops->minor == 0;
    bool is_sm121 = dprops->major == 12 && dprops->minor == 1;
    TORCH_CHECK(is_sm120 || is_sm121, "only supports Blackwell GPUs or newer.");

    auto q_dtype = q.dtype();
    auto sfq_dtype = sfq.dtype();
    if (is_mxfp8) {
        TORCH_CHECK(q_dtype == torch::kFloat8_e4m3fn, "q dtype must be float8_e4m3fn for MXFP8");
    } else {
        TORCH_CHECK(q_dtype == torch::kUInt8, "q dtype must be uint8 for NVFP4");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    if (is_mxfp8) {
        TORCH_CHECK(sfq_dtype == torch::kUInt8, "scale dtype must be uint8 for MXFP8 UE8M0");
    } else {
        TORCH_CHECK(sfq_dtype == torch::kFloat8_e4m3fn, "scale dtype must be float8_e4m3fn for NVFP4");
    }
    TORCH_CHECK(sfk.dtype() == sfq_dtype, "query and key must have the same dtype");
    TORCH_CHECK(sfv.dtype() == sfq_dtype, "query and value must have the same dtype");
    CHECK_DEVICE(sfq); CHECK_DEVICE(sfk); CHECK_DEVICE(sfv);
    
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(delta_s.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    TORCH_CHECK(q.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor must be contiguous");

    const auto sizes = q.sizes();
    auto opts = q.options();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[2];
    int num_heads = sizes[1];
    const int head_size_og = sizes[3];
    const int unpacked_head_size = is_mxfp8 ? head_size_og : head_size_og * 2;
    const int seqlen_k = k.size(2);
    const int num_heads_k = k.size(1);
    
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(unpacked_head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(!is_mxfp8 || unpacked_head_size == 128, "MXFP8 path currently supports head_dim=128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_heads == num_heads_k, "We do not support MQA/GQA yet");

    TORCH_CHECK(unpacked_head_size == 64 || unpacked_head_size == 128 || unpacked_head_size == 256, "Only support head size 64, 128, and 256 for now");

    CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
    CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
    // Vt is (B, H, D, seqlen_k); NVFP4 packs two e2m1 per byte so its last extent
    // is halved, MXFP8 stores one e4m3 per byte and keeps it whole.
    if (is_mxfp8) {
        CHECK_SHAPE(v, batch_size, num_heads_k, unpacked_head_size, seqlen_k);
    } else {
        CHECK_SHAPE(v, batch_size, num_heads_k, unpacked_head_size, seqlen_k/2);
    }
    // CHECK_SHAPE(delta_s, batch_size, num_heads, seqlen_q / 128, seqlen_k);
    // CHECK_SHAPE(sfq, batch_size, seqlen_q, num_heads, unpacked_head_size);
    // CHECK_SHAPE(sfk, batch_size, seqlen_k, num_heads_k, unpacked_head_size);
    // CHECK_SHAPE(sfv, batch_size, unpacked_head_size, num_heads_k, seqlen_k);
    TORCH_CHECK(unpacked_head_size % 8 == 0, "head_size must be a multiple of 8");
    
    auto dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    at::Tensor out = torch::empty({batch_size, num_heads, seqlen_q, unpacked_head_size}, opts.dtype(dtype));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // const int head_size = round_multiple(head_size_og, 8);
    // const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k, unpadded_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     unpacked_head_size, unpacked_head_size,
                     q, k, v, delta_s, out, 
                     sfq, sfk, sfv,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/-1,
                     /*window_size_right=*/is_causal ? 0 : -1,
                     per_block_mean,
                     is_bf16,
                     is_mxfp8
                    );
    // TODO: 132 sm count?
    auto tile_count_semaphore = is_causal ? torch::full({1}, 132, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // at::Tensor out_padded = out;
    // if (head_size_og % 8 != 0) {
    //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     if (out_.has_value()) { out_.value().copy_(out); }
    // }

    // return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(err));
    return {out, softmax_lse};
}




namespace mxfp8_simple_impl {

__device__ __forceinline__ float ue8m0_to_float_kernel(uint8_t bits) {
    return ldexpf(1.0f, int(bits) - 127);
}

template <typename OutT>
__global__ void mxfp8_fwd_simple_kernel(
    cutlass::float_e4m3_t const* __restrict__ q,
    cutlass::float_e4m3_t const* __restrict__ k,
    cutlass::float_e4m3_t const* __restrict__ v,
    uint8_t const* __restrict__ sfq,
    uint8_t const* __restrict__ sfk,
    uint8_t const* __restrict__ sfv,
    OutT* __restrict__ out,
    int B, int H, int QL, int KL, int D,
    float softmax_scale,
    bool is_causal) {

    extern __shared__ float smem[];
    float* red = smem;
    __shared__ float shared_score;
    __shared__ float row_max;
    __shared__ float row_sum;

    int row = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    int q_base = ((b * H + h) * QL + row) * D;
    int out_base = q_base;

    if (tid == 0) {
        row_max = -INFINITY;
        row_sum = 0.0f;
    }
    __syncthreads();

    for (int col = 0; col < KL; ++col) {
        bool masked = is_causal && col > row;
        float partial = 0.0f;
        if (!masked) {
            int k_base = ((b * H + h) * KL + col) * D;
            for (int d = tid; d < D; d += blockDim.x) {
                float qs = ue8m0_to_float_kernel(sfq[((b * H + h) * QL + row) * (D / 32) + d / 32]);
                float ks = ue8m0_to_float_kernel(sfk[((b * H + h) * KL + col) * (D / 32) + d / 32]);
                partial += float(q[q_base + d]) * qs * float(k[k_base + d]) * ks;
            }
        }
        red[tid] = partial;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) { red[tid] += red[tid + offset]; }
            __syncthreads();
        }
        if (tid == 0) {
            float score = masked ? -INFINITY : red[0] * softmax_scale;
            row_max = fmaxf(row_max, score);
        }
        __syncthreads();
    }

    for (int col = 0; col < KL; ++col) {
        bool masked = is_causal && col > row;
        float partial = 0.0f;
        if (!masked) {
            int k_base = ((b * H + h) * KL + col) * D;
            for (int d = tid; d < D; d += blockDim.x) {
                float qs = ue8m0_to_float_kernel(sfq[((b * H + h) * QL + row) * (D / 32) + d / 32]);
                float ks = ue8m0_to_float_kernel(sfk[((b * H + h) * KL + col) * (D / 32) + d / 32]);
                partial += float(q[q_base + d]) * qs * float(k[k_base + d]) * ks;
            }
        }
        red[tid] = partial;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) { red[tid] += red[tid + offset]; }
            __syncthreads();
        }
        if (tid == 0) {
            float score = masked ? -INFINITY : red[0] * softmax_scale;
            float p = expf(score - row_max);
            shared_score = p;
            row_sum += p;
        }
        __syncthreads();
    }

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int col = 0; col < KL; ++col) {
            bool masked = is_causal && col > row;
            float partial = 0.0f;
            if (!masked) {
                int k_base = ((b * H + h) * KL + col) * D;
                for (int kd = tid; kd < D; kd += blockDim.x) {
                    float qs = ue8m0_to_float_kernel(sfq[((b * H + h) * QL + row) * (D / 32) + kd / 32]);
                    float ks = ue8m0_to_float_kernel(sfk[((b * H + h) * KL + col) * (D / 32) + kd / 32]);
                    partial += float(q[q_base + kd]) * qs * float(k[k_base + kd]) * ks;
                }
            }
            red[tid] = partial;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset) { red[tid] += red[tid + offset]; }
                __syncthreads();
            }
            if (tid == 0) {
                float score = masked ? -INFINITY : red[0] * softmax_scale;
                shared_score = expf(score - row_max) / row_sum;
            }
            __syncthreads();
            int v_base = ((b * H + h) * KL + col) * D;
            float vs = ue8m0_to_float_kernel(sfv[((b * H + h) * KL + col) * (D / 32) + d / 32]);
            acc += shared_score * float(v[v_base + d]) * vs;
            __syncthreads();
        }
        out[out_base + d] = OutT(acc);
    }
}

std::vector<at::Tensor> mxfp8_fwd_simple(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor sfq, at::Tensor sfk, at::Tensor sfv,
    int64_t unpadded_k,
    bool is_causal,
    bool is_bf16) {

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(sfq); CHECK_DEVICE(sfk); CHECK_DEVICE(sfv);
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn, "q must be float8_e4m3fn");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype(), "q/k/v dtype mismatch");
    TORCH_CHECK(sfq.dtype() == torch::kUInt8, "sfq must be uint8 UE8M0");
    TORCH_CHECK(sfk.dtype() == sfq.dtype() && sfv.dtype() == sfq.dtype(), "scale dtype mismatch");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(sfq.is_contiguous() && sfk.is_contiguous() && sfv.is_contiguous(), "scales must be contiguous");

    int B = q.size(0), H = q.size(1), QL = q.size(2), D = q.size(3);
    int KL = int(unpadded_k);
    TORCH_CHECK(D == 128, "mxfp8_fwd_simple currently supports head_dim=128");
    CHECK_SHAPE(k, B, H, k.size(2), D);
    CHECK_SHAPE(v, B, H, k.size(2), D);
    CHECK_SHAPE(sfq, B, H, QL, D / 32);
    CHECK_SHAPE(sfk, B, H, k.size(2), D / 32);
    CHECK_SHAPE(sfv, B, H, k.size(2), D / 32);
    TORCH_CHECK(KL <= k.size(2), "unpadded_k exceeds K length");

    auto out_dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    auto out = torch::empty({B, H, QL, D}, q.options().dtype(out_dtype));
    dim3 grid(QL, H, B);
    int threads = 128;
    size_t shmem = threads * sizeof(float);
    float softmax_scale = 1.0f / sqrtf(float(D));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    if (is_bf16) {
        mxfp8_fwd_simple_kernel<cutlass::bfloat16_t><<<grid, threads, shmem, stream>>>(
            reinterpret_cast<cutlass::float_e4m3_t const*>(q.data_ptr()),
            reinterpret_cast<cutlass::float_e4m3_t const*>(k.data_ptr()),
            reinterpret_cast<cutlass::float_e4m3_t const*>(v.data_ptr()),
            sfq.data_ptr<uint8_t>(), sfk.data_ptr<uint8_t>(), sfv.data_ptr<uint8_t>(),
            reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
            B, H, QL, KL, D, softmax_scale, is_causal);
    } else {
        mxfp8_fwd_simple_kernel<cutlass::half_t><<<grid, threads, shmem, stream>>>(
            reinterpret_cast<cutlass::float_e4m3_t const*>(q.data_ptr()),
            reinterpret_cast<cutlass::float_e4m3_t const*>(k.data_ptr()),
            reinterpret_cast<cutlass::float_e4m3_t const*>(v.data_ptr()),
            sfq.data_ptr<uint8_t>(), sfk.data_ptr<uint8_t>(), sfv.data_ptr<uint8_t>(),
            reinterpret_cast<cutlass::half_t*>(out.data_ptr()),
            B, H, QL, KL, D, softmax_scale, is_causal);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out};
}

} // namespace mxfp8_simple_impl

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("mxfp8_fwd_simple", &mxfp8_simple_impl::mxfp8_fwd_simple, "Simple native CUDA MXFP8 forward baseline");
}
