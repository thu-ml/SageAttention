"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import triton
import triton.language as tl
from torch import Tensor


def hyperparameter_check(hyper, H, device):
    if type(hyper) == float or type(hyper) == int:
        hyper = torch.full((H,), float(hyper), device=device)
    elif isinstance(hyper, Tensor):
        assert len(hyper.shape) <= 1, "Hyperparameter tensor must be 1D"
        if len(hyper.shape) == 0:
            hyper = torch.full((H,), hyper.item(), device=device)
        assert hyper.numel() == H, f"Hyperparameter tensor must have {H} elements, but has {hyper.numel()}"
        hyper = hyper.to(device)
    else:
        print(hyper)
        raise ValueError("Hyperparameter must be a float or a tensor")
    return hyper


@triton.jit
def triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    valid_block_num = 0

    map_ptr = map_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    lut_ptr = lut_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    valid_block_num_ptr = valid_block_num_ptr + b * H * Q + h * Q + q
    
    valid_block_num = 0

    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            # use actual index instead of incremental index
            tl.store(lut_ptr + valid_block_num, i)
            valid_block_num += 1

    tl.store(valid_block_num_ptr, valid_block_num)

def block_map_lut_triton(block_map):
    assert block_map.dim() == 4
    assert block_map.is_contiguous()

    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    valid_block_num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)

    grid = (B, H, Q)
    triton_block_map_to_lut_kernel[grid](block_map, lut, valid_block_num, K)

    return lut, valid_block_num

@triton.jit
def triton_bmm_pool_sim_simmean(x_ptr, pool_ptr, sim_ptr, simthreshd1, N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)
    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)
    
    
def get_pool_sim_triton_simmean(x, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks


@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)
    

def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map

@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)

def fill_causal_mask_triton(mask, BqdivBk:float):
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


def get_block_map_meansim(q, k, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.1, cdfthreshd=0.9, topk=None, is_sparse=True, return_lut=False, attention_sink=False):
    assert (cdfthreshd is None and topk is not None) \
        or (cdfthreshd is not None and topk is None), "Only one of cdfthreshd and topk can be set."

    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    if cdfthreshd is not None:
        cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    if topk is not None:
        topk = hyperparameter_check(topk, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    if cdfthreshd is not None:
        cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
        cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    else:
        num_to_select = (topk * K).to(torch.int64).view(1, H, 1).expand(B, -1, Q).contiguous()

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    if not return_lut:
        return final_map
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)
        return lut, valid_block_num

