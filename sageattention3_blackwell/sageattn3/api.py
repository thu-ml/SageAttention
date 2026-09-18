"""
Copyright (c) 2025 by SageAttention team.

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
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    # Do not mutate the caller's K tensor: the in-place subtraction silently
    # corrupts K for any caller that reuses or shares it across grouped heads.
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads % num_kv_heads:
        raise ValueError("the number of Q heads must be divisible by KV heads")
    if num_q_heads == num_kv_heads:
        delta_s = torch.matmul(qm, k.transpose(-2, -1))
    else:
        # GQA: compute the mean correction with grouped indexing instead of
        # materializing K num_q_heads / num_kv_heads times.
        q_per_kv = num_q_heads // num_kv_heads
        grouped_qm = qm.reshape(
            qm.size(0), num_kv_heads, q_per_kv, qm.size(-2), qm.size(-1)
        )
        grouped_kt = k.transpose(-2, -1).unsqueeze(2)
        delta_s = torch.matmul(grouped_qm, grouped_kt).reshape(
            qm.size(0), num_q_heads, qm.size(-2), k.size(-2)
        )
    delta_s = delta_s.to(torch.float32).contiguous()
    return q, k, v, delta_s

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False,
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         softmax_scale: float = None,
                         num_sms: int = 0,
                        ):
    if softmax_scale is None:
        softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16, num_sms)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    head_dim = q.size(-1)
    if k.size(-1) != head_dim or v.size(-1) != head_dim:
        raise ValueError("Q, K, and V must have the same head dimension")
    if head_dim not in (64, 128):
        # The kernel only compiles head_dim 64 and 128; pad smaller head dims
        # (the zero tail is transparent for the block-scaled FP4 quantization
        # and the dot products) and slice the output back.
        if head_dim < 64:
            padded_head_dim = 64
        elif head_dim < 128:
            padded_head_dim = 128
        else:
            raise ValueError(f"Unsupported head dimension: {head_dim}")
        pad = padded_head_dim - head_dim
        q, k, v = (F.pad(x, (0, pad)) for x in (q, k, v))
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda,
    vlist_from_cuda,
    delta_s,
    KL,
    is_causal,
    per_block_mean,
    is_bf16,
    kwargs.get("sm_scale", head_dim ** (-0.5)),
    # num_sms <= 0 lets the kernel size its persistent grid from the
    # current device's SM count instead of a hardcoded value.
    kwargs.get("num_sms", 0),
    )[0][:, :, :QL, :head_dim].contiguous()
    return o_fp4