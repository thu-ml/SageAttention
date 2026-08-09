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


# The fused kernels amortize their launch configuration at long sequence lengths;
# retain the lower-overhead PyTorch path below this model-independent cutoff.
_FUSED_PREPROCESS_MIN_SEQLEN = 4096


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


@triton.jit
def group_mean_padded_kernel(
    q_ptr,
    q_out_ptr,
    qm_out_ptr,
    L,
    D: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    stride_qmb,
    stride_qmh,
    stride_qml,
    stride_qmd,
    GROUP_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    offsets_l = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    offsets_d = tl.arange(0, D)
    q_offsets = (
        pid_b * stride_qb
        + pid_h * stride_qh
        + offsets_l[:, None] * stride_ql
        + offsets_d[None, :] * stride_qd
    )
    q_group = tl.load(q_ptr + q_offsets, mask=offsets_l[:, None] < L, other=0.0)
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE

    out_offsets = (
        pid_b * stride_ob
        + pid_h * stride_oh
        + offsets_l[:, None] * stride_ol
        + offsets_d[None, :] * stride_od
    )
    tl.store(q_out_ptr + out_offsets, q_group - qm_group)

    qm_offset = (
        pid_b * stride_qmb
        + pid_h * stride_qmh
        + pid_group * stride_qml
        + offsets_d * stride_qmd
    )
    tl.store(qm_out_ptr + qm_offset, qm_group)


@triton.jit
def center_and_pad_k_kernel(
    k_ptr,
    km_ptr,
    k_out_ptr,
    L,
    PADDED_L,
    D: tl.constexpr,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_kmb,
    stride_kmh,
    stride_kmd,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_l = tl.program_id(2)

    offsets_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offsets_d = tl.arange(0, D)
    live = offsets_l[:, None] < L
    k_offsets = (
        pid_b * stride_kb
        + pid_h * stride_kh
        + offsets_l[:, None] * stride_kl
        + offsets_d[None, :] * stride_kd
    )
    values = tl.load(k_ptr + k_offsets, mask=live, other=0.0)
    means = tl.load(
        km_ptr
        + pid_b * stride_kmb
        + pid_h * stride_kmh
        + offsets_d[None, :] * stride_kmd
    )
    values = tl.where(live, values - means, 0.0)

    out_offsets = (
        pid_b * stride_ob
        + pid_h * stride_oh
        + offsets_l[:, None] * stride_ol
        + offsets_d[None, :] * stride_od
    )
    tl.store(k_out_ptr + out_offsets, values, mask=offsets_l[:, None] < PADDED_L)


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


def triton_group_mean_padded(q: torch.Tensor, padded_len: int):
    B, H, _, D = q.shape
    GROUP_SIZE = 128
    num_groups = padded_len // GROUP_SIZE
    q_out = torch.empty(B, H, padded_len, D, device=q.device, dtype=q.dtype)
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)
    group_mean_padded_kernel[(B, H, num_groups)](
        q,
        q_out,
        qm,
        q.size(2),
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        q_out.stride(3),
        qm.stride(0),
        qm.stride(1),
        qm.stride(2),
        qm.stride(3),
        GROUP_SIZE=GROUP_SIZE,
    )
    return q_out, qm


def center_and_pad_k(k: torch.Tensor, padded_len: int):
    B, H, _, D = k.shape
    k_mean = k.mean(dim=-2, keepdim=True)
    k_out = torch.empty(B, H, padded_len, D, device=k.device, dtype=k.dtype)
    BLOCK_L = 32
    center_and_pad_k_kernel[(B, H, triton.cdiv(padded_len, BLOCK_L))](
        k,
        k_mean,
        k_out,
        k.size(2),
        padded_len,
        D,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_mean.stride(0),
        k_mean.stride(1),
        k_mean.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        BLOCK_L=BLOCK_L,
        num_warps=8,
    )
    return k_out


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    if k.size(2) >= _FUSED_PREPROCESS_MIN_SEQLEN:
        k_padded_len = triton.cdiv(k.size(2), 128) * 128
        k = center_and_pad_k(k, k_padded_len)
    else:
        # Keep the launch-efficient PyTorch path for short sequences, but avoid
        # exposing the original in-place K centering to callers.
        k = k - k.mean(dim=-2, keepdim=True)
        k = pad_128(k)
    v = pad_128(v)
    if per_block_mean:
        if q.size(2) >= _FUSED_PREPROCESS_MIN_SEQLEN:
            q_padded_len = triton.cdiv(q.size(2), 128) * 128
            q, qm = triton_group_mean_padded(q, q_padded_len)
        else:
            q = pad_128(q)
            q, qm = triton_group_mean(q)
    else:
        q = pad_128(q)
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
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
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
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
    is_bf16
    )[0][:, :, :QL, :].contiguous()
    return o_fp4
