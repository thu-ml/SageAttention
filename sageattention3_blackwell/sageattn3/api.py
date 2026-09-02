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

def scale_and_quant_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference quantizer: row-major scales, for the mxfp8_ref correctness baseline.
    The native kernel needs the swizzled 128x4 scale layout, so it uses the CUDA
    quantizers below instead."""
    assert x.ndim == 4
    B, H, N, D = x.shape
    assert D % 32 == 0
    x_blocks = x.to(torch.float32).reshape(B, H, N, D // 32, 32)
    amax = x_blocks.abs().amax(dim=-1)
    min_scale = torch.full_like(amax, 2.0 ** -126)
    scale = torch.maximum(2.0 ** torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=2.0 ** -126))), min_scale)
    ue8m0_scale = torch.clamp(torch.log2(scale).to(torch.int16) + 127, 0, 255).to(torch.uint8)
    fp8 = (x_blocks / scale.unsqueeze(-1)).reshape(B, H, N, D).to(torch.float8_e4m3fn)
    return fp8, ue8m0_scale

def scale_and_quant_mxfp8_cuda(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    fp8 = torch.empty((B, H, N, D), device=x.device, dtype=torch.float8_e4m3fn)
    ue8m0_scale = torch.empty((B, H, N, D // 32), device=x.device, dtype=torch.uint8)
    fp4quant_cuda.scaled_mxfp8_quant(x, fp8, ue8m0_scale, 1)
    return fp8, ue8m0_scale

def scale_and_quant_mxfp8_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    fp8 = torch.empty((B, H, N, D), device=x.device, dtype=torch.float8_e4m3fn)
    ue8m0_scale = torch.empty((B, H, N, D // 32), device=x.device, dtype=torch.uint8)
    fp4quant_cuda.scaled_mxfp8_quant_permute(x, fp8, ue8m0_scale, 1)
    return fp8, ue8m0_scale

def scale_and_quant_mxfp8_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    fp8 = torch.empty((B, H, D, N), device=x.device, dtype=torch.float8_e4m3fn)
    ue8m0_scale = torch.empty((B, H, D, N // 32), device=x.device, dtype=torch.uint8)
    fp4quant_cuda.scaled_mxfp8_quant_trans(x, fp8, ue8m0_scale, 1)
    return fp8, ue8m0_scale

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
    # pybind11 does not expose C++ default arguments, so is_mxfp8 must be passed.
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16, False)

def blockscaled_mxfp8_attn(qlist: Tuple,
                           klist: Tuple,
                           vlist: Tuple,
                           delta_s: torch.Tensor,
                           KL: int,
                           is_causal: bool = False,
                           per_block_mean: bool = True,
                           is_bf16: bool = True
                          ):
    # Unlike NVFP4, the last extent is the real head_dim (one e4m3 per byte).
    softmax_scale = qlist[0].shape[-1] ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None,
                            softmax_scale, is_causal, per_block_mean, is_bf16, True)

def blockscaled_mxfp8_attn_ref(qlist: Tuple,
                               klist: Tuple,
                               vlist: Tuple,
                               delta_s: torch.Tensor,
                               KL: int,
                               is_causal: bool = False,
                               per_block_mean: bool = True,
                               is_bf16: bool = True
                              ):
    def dequant_mxfp8(tensor: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
        scale = torch.ldexp(
            torch.ones_like(scale_ue8m0, dtype=torch.float32),
            scale_ue8m0.to(torch.int16) - 127,
        )
        return (tensor.to(torch.float32).reshape(*scale.shape, 32) * scale.unsqueeze(-1)).flatten(-2)

    q = dequant_mxfp8(*qlist)
    k = dequant_mxfp8(*klist)
    v = dequant_mxfp8(*vlist)
    out = sdpa(q, k[:, :, :KL, :], v[:, :, :KL, :], is_causal=is_causal)
    return (out.to(torch.bfloat16 if is_bf16 else torch.float16),)

_buf_cache = {}

def _buf(tag, shape, dtype, device):
    """Reused scratch buffer. Creating a tensor from Python costs ~2.5us regardless
    of size, and the mxfp8 path needs six of them per call -- more than the CUDA
    quantization work itself at short sequence lengths. Safe here because every
    buffer is consumed by the attention kernel inside the same call; not safe for
    concurrent calls with identical shapes.

    The tag is part of the key on purpose: in self-attention Q and K have identical
    shapes, so keying on shape alone hands out the same buffer for both and K's
    quantization overwrites Q's."""
    key = (tag, shape, dtype, device)
    b = _buf_cache.get(key)
    if b is None:
        b = torch.empty(shape, dtype=dtype, device=device)
        _buf_cache[key] = b
    return b

def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, quant_format = "mxfp8_tc", **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    if quant_format == "mxfp8_ref":
        # Dequantize-and-SDPA baseline. Measures pure quantization round-trip error,
        # and does NOT run the SageAttention algorithm (no smoothing, no delta_s).
        qlist_from_cuda = scale_and_quant_mxfp8(q.contiguous())
        klist_from_cuda = scale_and_quant_mxfp8(k.contiguous())
        vlist_from_cuda = scale_and_quant_mxfp8(v.contiguous())
        out = blockscaled_mxfp8_attn_ref(
            qlist_from_cuda, klist_from_cuda, vlist_from_cuda,
            None, KL, is_causal, per_block_mean, is_bf16
        )[0]
        return out[:, :, :QL, :].contiguous()
    if quant_format == "mxfp8":
        q8, qsf = scale_and_quant_mxfp8(q.contiguous())
        k8, ksf = scale_and_quant_mxfp8(k.contiguous())
        v8, vsf = scale_and_quant_mxfp8(v.contiguous())
        out = fp4attn_cuda.mxfp8_fwd_simple(
            q8, k8, v8, qsf, ksf, vsf, KL, is_causal, is_bf16
        )[0]
        return out[:, :, :QL, :].contiguous()
    if quant_format == "mxfp8_tc":
        if q.size(-1) != 128:
            raise ValueError("mxfp8_tc currently supports head_dim=128")
        def pad_128(x):
            L = x.size(2)
            pad_len = (128 - L % 128) % 128
            if pad_len == 0:
                return x.contiguous()
            return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
        q_pad, k_pad, v_pad = map(pad_128, [q, k, v])
        qlist_from_cuda = scale_and_quant_mxfp8_cuda(q_pad)
        klist_from_cuda = scale_and_quant_mxfp8_permute(k_pad)
        vlist_from_cuda = scale_and_quant_mxfp8_transpose(v_pad)
        delta_s = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
        out = blockscaled_mxfp8_attn(
            qlist_from_cuda, klist_from_cuda, vlist_from_cuda,
            delta_s, KL, is_causal, False, is_bf16
        )[0]
        return out[:, :, :QL, :].contiguous()
    if quant_format == "nvfp4":
        raise ValueError(
            "quant_format='nvfp4' is not validated in this MXFP8 D128 clean branch. "
            "Use the official SageAttention3 Blackwell package for the original NVFP4 path."
        )
    raise ValueError(f"Unsupported quant_format: {quant_format}")
