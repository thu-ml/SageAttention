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
from typing import Optional, Tuple
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
    
    k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
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


_GROUP_SIZE = 128
_DEFAULT_Q_CHUNK_SIZE = 1024
_INT32_MAX = 2**31 - 1


def _pad_sequence_to_group(x: torch.Tensor) -> torch.Tensor:
    pad_tokens = (-x.shape[2]) % _GROUP_SIZE
    if pad_tokens:
        return F.pad(x, (0, 0, 0, pad_tokens), value=0).contiguous()
    return x.contiguous()


def _delta_s_batch_stride(q: torch.Tensor, k: torch.Tensor) -> int:
    q_groups = (q.shape[2] + _GROUP_SIZE - 1) // _GROUP_SIZE
    padded_k_tokens = (
        (k.shape[2] + _GROUP_SIZE - 1) // _GROUP_SIZE * _GROUP_SIZE
    )
    return q.shape[1] * q_groups * padded_k_tokens


def _requires_delta_s_chunking(q: torch.Tensor, k: torch.Tensor) -> bool:
    # The Blackwell TMA delta_s layout stores its dynamic strides as int32_t.
    # A larger batch stride overflows during descriptor construction.
    return _delta_s_batch_stride(q, k) > _INT32_MAX


def _normalize_q_chunk_size(
    q_chunk_size: int,
    heads: int,
    padded_k_tokens: int,
) -> int:
    if q_chunk_size <= 0 or q_chunk_size % _GROUP_SIZE:
        raise ValueError(
            f"q_chunk_size must be a positive multiple of {_GROUP_SIZE}, "
            f"got {q_chunk_size}"
        )
    chunk_groups = q_chunk_size // _GROUP_SIZE
    if heads * chunk_groups * padded_k_tokens > _INT32_MAX:
        max_groups = _INT32_MAX // (heads * padded_k_tokens)
        max_tokens = max_groups * _GROUP_SIZE
        raise ValueError(
            "q_chunk_size still overflows the int32 delta_s TMA stride; "
            f"use q_chunk_size <= {max_tokens}"
        )
    return q_chunk_size


def _validate_chunked_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must use HND layout [B, H, N, D]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"SA3 expects FP16 or BF16 input, got {q.dtype}")
    if not q.is_cuda:
        raise ValueError("SA3 requires CUDA tensors")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must have matching batch and head counts")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("q, k, and v must have the same head dimension")


def _sageattn3_blackwell_full(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=False,
    per_block_mean=True,
    **kwargs,
):
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


@torch.inference_mode()
def _sageattn3_blackwell_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_chunk_size: int,
) -> torch.Tensor:
    _validate_chunked_inputs(q, k, v)

    q_tokens = q.shape[2]
    k_tokens = k.shape[2]
    padded_k_tokens = (
        (k_tokens + _GROUP_SIZE - 1) // _GROUP_SIZE * _GROUP_SIZE
    )
    q_chunk_size = _normalize_q_chunk_size(
        int(q_chunk_size), q.shape[1], padded_k_tokens
    )
    is_bf16 = q.dtype == torch.bfloat16

    # Match upstream preprocessing while keeping K/V quantization outside the
    # Q-chunk loop. Unlike preprocess_qkv, this does not mutate the caller's K.
    k_padded = _pad_sequence_to_group(k - k.mean(dim=-2, keepdim=True))
    v_padded = _pad_sequence_to_group(v)
    k_quant = scale_and_quant_fp4_permute(k_padded)
    v_quant = scale_and_quant_fp4_transpose(v_padded)
    del v_padded
    k_transposed = k_padded.transpose(-2, -1)

    output = torch.empty_like(q)
    for start in range(0, q_tokens, q_chunk_size):
        end = min(start + q_chunk_size, q_tokens)
        q_chunk = _pad_sequence_to_group(q[:, :, start:end, :])
        q_centered, q_mean = triton_group_mean(q_chunk)
        delta_s = torch.matmul(q_mean, k_transposed).to(torch.float32).contiguous()
        q_quant = scale_and_quant_fp4(q_centered)

        chunk_output = blockscaled_fp4_attn(
            q_quant,
            k_quant,
            v_quant,
            delta_s,
            k_tokens,
            False,
            True,
            is_bf16,
        )[0]
        output[:, :, start:end, :].copy_(
            chunk_output[:, :, : end - start, :]
        )

    return output


def sageattn3_blackwell(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=False,
    per_block_mean=True,
    q_chunk_size: Optional[int] = None,
    **kwargs,
):
    """Run SageAttention3 on Blackwell GPUs.

    Non-causal per-block-mean attention is automatically split into aligned Q
    chunks when the full ``delta_s`` batch stride would overflow the kernel's
    int32 TMA layout. Pass ``q_chunk_size`` explicitly to bound ``delta_s``
    memory for smaller inputs as well. The size must be a multiple of 128.
    """
    can_chunk = (
        not is_causal
        and per_block_mean
        and attn_mask is None
        and q.size(-1) < 256
    )
    if not can_chunk:
        return _sageattn3_blackwell_full(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            **kwargs,
        )

    if q_chunk_size is None and not _requires_delta_s_chunking(q, k):
        return _sageattn3_blackwell_full(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            per_block_mean=per_block_mean,
            **kwargs,
        )

    if q_chunk_size is None:
        q_chunk_size = _DEFAULT_Q_CHUNK_SIZE
    return _sageattn3_blackwell_chunked(q, k, v, q_chunk_size)
