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

#include "sm89_fp8_arch.hpp"
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = e4m3 * e4m3 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,  _2,  _2>>,
                          Stride<Stride<_64,_1>,Stride<_16,  _8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape < _4,  _2>>,
                          Stride<Stride<_32,_1>,Stride< _8,_128>>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,  _2>>,
                          Stride<Stride<_32,_1>,Stride<_16,  _8>>>;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = e4m3 * e5m2 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E5M2F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> 
{
  using ValTypeB = float_e5m2_t;
};


///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = e5m2 * e4m3 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E4M3F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
{
  using ValTypeA = float_e5m2_t;
};



///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = e5m2 * e5m2 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E5M2F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
{
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = e4m3 * e4m3 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM89_16x8x32_F16E4M3E4M3F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
{
    using ValTypeD = half_t;
    using ValTypeC = half_t;
};

} // end namespace cute