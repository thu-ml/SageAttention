#pragma once

#include <cstdint>

namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 32u,
        BLOCKS_X  = 1u,
        BLOCKS_Y  = 4u,
        TBLOCK_X  = 256u,
        TBLOCK_Y  = 1u,
        WARP_SIZE = rocwmma::Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 32u,
        BLOCKS_X  = 1u,
        BLOCKS_Y  = 4u,
        TBLOCK_X  = 128u,
        TBLOCK_Y  = 1u,
        WARP_SIZE = rocwmma::Constants::AMDGCN_WAVE_SIZE_32
    };
}

