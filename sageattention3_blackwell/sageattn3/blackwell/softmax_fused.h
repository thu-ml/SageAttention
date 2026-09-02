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
#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Rows>
struct SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum, row_max, scores_scale;
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f; //log2f(fp8_scalexfp4_scale)
    static constexpr float fp4_scale_log2 = -2.584962500721156f; // log2f(fp4_scale)
    static constexpr int RowReductionThr = 4;

    CUTLASS_DEVICE SoftmaxFused(){};

    template<typename TensorAcc>
    CUTLASS_DEVICE void init_skip_softmax(TensorAcc&) {
        fill(row_max, 0.f);
        fill(row_sum, 1.f);
        fill(scores_scale, 1.f);
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void update_skip_softmax(TensorAcc&) {
        fill(scores_scale, 1.f);
    }

    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc, 
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));
        
        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
                // scores_scale(mi) = max_scaled;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
        }
    }



    template<
        bool FirstTile,
        bool CrossLaneReduce = true,
        bool UseSoftwareExp2 = false,
        bool SkipScaleExp2 = false,
        bool SkipScoreExp2 = false,
        bool InfCheck = false,
        typename TensorAcc
    >
    CUTLASS_DEVICE auto online_softmax_no_quant(
        TensorAcc& acc,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_max(mi) = fmaxf(row_max(mi), acc_reduction_view(mi, ni));
                }
                if constexpr (CrossLaneReduce) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 1; i < RowReductionThr; i <<= 1) {
                        float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), i);
                        row_max(mi) = fmaxf(row_max(mi), max_recv);
                    }
                }
                float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : row_max(mi) * softmax_scale_log2)
                    : row_max(mi) * softmax_scale_log2;
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    if constexpr (SkipScoreExp2) {
                        acc_reduction_view(mi, ni) = 1.f;
                    } else {
                        acc_reduction_view(mi, ni) = flash::softmax_exp2<UseSoftwareExp2>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    }
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_max(mi) = fmaxf(row_max(mi), acc_reduction_view(mi, ni));
                }
                if constexpr (CrossLaneReduce) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 1; i < RowReductionThr; i <<= 1) {
                        float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), i);
                        row_max(mi) = fmaxf(row_max(mi), max_recv);
                    }
                }
                float scores_max_cur = !InfCheck
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                if constexpr (SkipScaleExp2) {
                    scores_scale(mi) = 1.f;
                } else {
                    scores_scale(mi) = flash::softmax_exp2<UseSoftwareExp2>((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                }
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : row_max(mi) * softmax_scale_log2)
                    : row_max(mi) * softmax_scale_log2;
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    if constexpr (SkipScoreExp2) {
                        acc_reduction_view(mi, ni) = 1.f;
                    } else {
                        acc_reduction_view(mi, ni) = flash::softmax_exp2<UseSoftwareExp2>(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    }
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
    }

    template<bool FirstTile, typename TensorAcc>
    CUTLASS_DEVICE auto online_softmax_mxfp8_simple(
        TensorAcc& acc,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
            float prev_max = row_max(mi);
            float cur_max = FirstTile ? -INFINITY : prev_max;

            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                cur_max = fmaxf(cur_max, acc_reduction_view(mi, ni));
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float max_recv = __shfl_xor_sync(int32_t(-1), cur_max, i);
                cur_max = fmaxf(cur_max, max_recv);
            }

            row_max(mi) = cur_max;
            float max_scaled = cur_max * softmax_scale_log2;
            if constexpr (!FirstTile) {
                float scale = flash::ptx_exp2((prev_max - cur_max) * softmax_scale_log2);
                scores_scale(mi) = scale;
                row_sum(mi) *= scale;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                float p = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                acc_reduction_view(mi, ni) = p;
                row_sum(mi) += p;
            }
        }
    }

    template<typename TensorAcc, typename TensorPStage>
    CUTLASS_DEVICE void pack_mxfp8_p_stage_static(
        TensorAcc& acc,
        TensorPStage& tOrP_stage,
        const int base
    ) {
        constexpr float kPInvScale = 256.f;
        tOrP_stage(0) = cutlass::float_e4m3_t(acc(base + 0) * kPInvScale);
        tOrP_stage(1) = cutlass::float_e4m3_t(acc(base + 1) * kPInvScale);
        tOrP_stage(2) = cutlass::float_e4m3_t(acc(base + 4) * kPInvScale);
        tOrP_stage(3) = cutlass::float_e4m3_t(acc(base + 5) * kPInvScale);
        tOrP_stage(4) = cutlass::float_e4m3_t(acc(base + 2) * kPInvScale);
        tOrP_stage(5) = cutlass::float_e4m3_t(acc(base + 3) * kPInvScale);
        tOrP_stage(6) = cutlass::float_e4m3_t(acc(base + 6) * kPInvScale);
        tOrP_stage(7) = cutlass::float_e4m3_t(acc(base + 7) * kPInvScale);
        tOrP_stage(8) = cutlass::float_e4m3_t(acc(base + 8) * kPInvScale);
        tOrP_stage(9) = cutlass::float_e4m3_t(acc(base + 9) * kPInvScale);
        tOrP_stage(10) = cutlass::float_e4m3_t(acc(base + 12) * kPInvScale);
        tOrP_stage(11) = cutlass::float_e4m3_t(acc(base + 13) * kPInvScale);
        tOrP_stage(12) = cutlass::float_e4m3_t(acc(base + 10) * kPInvScale);
        tOrP_stage(13) = cutlass::float_e4m3_t(acc(base + 11) * kPInvScale);
        tOrP_stage(14) = cutlass::float_e4m3_t(acc(base + 14) * kPInvScale);
        tOrP_stage(15) = cutlass::float_e4m3_t(acc(base + 15) * kPInvScale);
    }

    template<typename TensorAcc, typename TensorPStage>
    CUTLASS_DEVICE void pack_mxfp8_p_stage_u32(
        TensorAcc& acc,
        TensorPStage& tOrP_stage,
        const int base
    ) {
        constexpr float kPInvScale = 256.f;
        Tensor tOrP_u32 = recast<uint32_t>(tOrP_stage);
        flash::packed_float_to_ue4m3(
            acc(base + 0) * kPInvScale,
            acc(base + 1) * kPInvScale,
            acc(base + 4) * kPInvScale,
            acc(base + 5) * kPInvScale,
            tOrP_u32(0));
        flash::packed_float_to_ue4m3(
            acc(base + 2) * kPInvScale,
            acc(base + 3) * kPInvScale,
            acc(base + 6) * kPInvScale,
            acc(base + 7) * kPInvScale,
            tOrP_u32(1));
        flash::packed_float_to_ue4m3(
            acc(base + 8) * kPInvScale,
            acc(base + 9) * kPInvScale,
            acc(base + 12) * kPInvScale,
            acc(base + 13) * kPInvScale,
            tOrP_u32(2));
        flash::packed_float_to_ue4m3(
            acc(base + 10) * kPInvScale,
            acc(base + 11) * kPInvScale,
            acc(base + 14) * kPInvScale,
            acc(base + 15) * kPInvScale,
            tOrP_u32(3));
    }

    template<
        bool FirstTile,
        bool FillSFP = true,
        bool StaticMap = false,
        bool PackedP = false,
        bool UseFma = false,
        typename TensorAcc,
        typename TensorP,
        typename TensorSFP
    >
    CUTLASS_DEVICE auto online_softmax_mxfp8_pack(
        TensorAcc& acc,
        TensorP& tOrP,
        TensorSFP& tOrSFP,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
            float prev_max = row_max(mi);
            float cur_max = FirstTile ? -INFINITY : prev_max;

            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                cur_max = fmaxf(cur_max, acc_reduction_view(mi, ni));
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float max_recv = __shfl_xor_sync(int32_t(-1), cur_max, i);
                cur_max = fmaxf(cur_max, max_recv);
            }

            row_max(mi) = cur_max;
            float max_scaled = UseFma ? fmaf(cur_max, softmax_scale_log2, 0.f) : cur_max * softmax_scale_log2;
            if constexpr (!FirstTile) {
                float scale_arg = UseFma
                    ? fmaf(prev_max - cur_max, softmax_scale_log2, 0.f)
                    : (prev_max - cur_max) * softmax_scale_log2;
                float scale = flash::ptx_exp2(scale_arg);
                scores_scale(mi) = scale;
                row_sum(mi) *= scale;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                float p_arg = UseFma
                    ? fmaf(acc_reduction_view(mi, ni), softmax_scale_log2, -max_scaled)
                    : acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled;
                float p = flash::ptx_exp2(p_arg);
                acc_reduction_view(mi, ni) = p;
                row_sum(mi) += p;
            }
        }

        constexpr float kPInvScale = 256.f;
        constexpr float kPScale = 1.f / kPInvScale;
        CUTLASS_PRAGMA_UNROLL
        for (int mma_k = 0; mma_k < size<2>(tOrP); ++mma_k) {
            Tensor tOrP_stage = tOrP(_, _, mma_k);
            Tensor tOrSFP_stage = tOrSFP(_, _, mma_k);
            if constexpr (FillSFP) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(tOrSFP_stage); ++i) {
                    tOrSFP_stage(i) = cutlass::float_ue8m0_t(kPScale);
                }
            }
            if constexpr (PackedP) {
                pack_mxfp8_p_stage_u32(acc, tOrP_stage, mma_k << 4);
            } else if constexpr (StaticMap) {
                pack_mxfp8_p_stage_static(acc, tOrP_stage, mma_k << 4);
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(tOrP_stage); ++i) {
                    int local_i = i & 15;
                    int src_i = (mma_k << 4) | (local_i & 0x9) | ((local_i & 0x2) << 1) | ((local_i & 0x4) >> 1);
                    tOrP_stage(i) = cutlass::float_e4m3_t(acc(src_i) * kPInvScale);
                }
            }
        }
    }

    template<bool FirstTile, bool CrossLaneReduce = true, typename TensorAcc>
    CUTLASS_DEVICE auto online_softmax_no_exp2(
        TensorAcc& acc,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);
        } else {
            fill(scores_scale, 1.f);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                row_max(mi) = fmaxf(row_max(mi), acc_reduction_view(mi, ni));
            }
            if constexpr (CrossLaneReduce) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 1; i < RowReductionThr; i <<= 1) {
                    float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), i);
                    row_max(mi) = fmaxf(row_max(mi), max_recv);
                }
            }
            (void)softmax_scale_log2;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                acc_reduction_view(mi, ni) = 1.f;
                row_sum(mi) += 1.f;
            }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) *= inv_sum;
             }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize_no_xlane_reduce(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            float sum = row_sum(mi);
            sum += __shfl_xor_sync(int32_t(-1), sum, 1);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) *= inv_sum;
            }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }

    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void accumulate_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) += o_tmp_reduction_view(mi, ni);
            }
        }
    }


};
} // namespace flash
