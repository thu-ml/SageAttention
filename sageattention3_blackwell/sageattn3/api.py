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
from functools import lru_cache
from pathlib import Path
import sys
from typing import Tuple

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.nn.functional import scaled_dot_product_attention as sdpa

_extension_suffix = ".pyd" if sys.platform == "win32" else ".abi3.so"
for _library in ("_fp4attn", "_fp4quant"):
    torch.ops.load_library(str(Path(__file__).with_name(_library + _extension_suffix)))
_ops = torch.ops.sageattn3


# The fused kernels amortize their launch configuration at long sequence lengths;
# retain the lower-overhead PyTorch path below this model-independent cutoff.
_FUSED_PREPROCESS_MIN_SEQLEN = 4096
_BLOCK_SIZE = 128
_INT32_MAX = 2**31 - 1

# P50 margin and P95 checks on Windows, RTX 5090 and PyTorch 2.12.1+cu130.
_SAGE2_CUTOFFS = {
    # (head dim, heads): (HND contiguous, NHD view, fused QKV view)
    (64, 8): (32256, 28160, 28160),
    (64, 16): (21760, 21760, 20736),
    (64, 24): (17920, 16640, 17920),
    (64, 32): (15104, 15360, 15104),
    (64, 48): (12544, 12032, 12544),
    (64, 56): (10880, 11520, 11008),
    (128, 8): (14848, 14592, 14592),
    (128, 16): (9984, 10112, 10240),
    (128, 24): (7680, 7808, 7680),
    (128, 32): (6784, 6912, 6784),
    (128, 48): (5120, 5248, 5120),
    (128, 56): (4480, 4608, 4736),
}
_SAGE2_ROUTING_SUPPORTED = (
    sys.platform == "win32"
    and torch.version.cuda == "13.0"
    and torch.__version__.split("+", 1)[0] == "2.12.1"
)


def _sage2_shape_matches(q, k, v):
    if q.ndim != 4 or q.size(0) != 1 or q.shape != k.shape or q.shape != v.shape:
        return False
    _, heads, tokens, dim = q.shape
    cutoffs = _SAGE2_CUTOFFS.get((dim, heads))
    if cutoffs is None or tokens < 255 or tokens > max(cutoffs):
        return False
    if not q.is_cuda or any(
        x.device != q.device or x.dtype != torch.bfloat16 or x.storage_offset() % 16
        for x in (q, k, v)
    ):
        return False
    strides = q.stride()[1:]
    if strides != k.stride()[1:] or strides != v.stride()[1:]:
        return False
    if q.is_contiguous():
        return tokens <= cutoffs[0]
    if strides == (dim, heads * dim, 1):
        return tokens <= cutoffs[1]
    if strides == (dim, 3 * heads * dim, 1):
        return tokens <= cutoffs[2]
    return False


@lru_cache(maxsize=None)
def _sage2_for_device(device):
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090":
        return None
    try:
        from sageattention.core import SM89_ENABLED, sageattn
    except (ImportError, OSError, RuntimeError):
        return None
    return sageattn if SM89_ENABLED else None


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


def pad_128(x):
    pad_len = -x.size(2) % _BLOCK_SIZE
    if pad_len:
        x = F.pad(x, (0, 0, 0, pad_len))
    return x.contiguous()


def _center_k(k):
    if k.size(2) >= _FUSED_PREPROCESS_MIN_SEQLEN:
        return center_and_pad_k(k, triton.cdiv(k.size(2), _BLOCK_SIZE) * _BLOCK_SIZE)
    return pad_128(k - k.mean(dim=-2, keepdim=True))


def _center_q(q, per_block_mean):
    if per_block_mean:
        if q.size(2) >= _FUSED_PREPROCESS_MIN_SEQLEN:
            return triton_group_mean_padded(q, triton.cdiv(q.size(2), _BLOCK_SIZE) * _BLOCK_SIZE)
        return triton_group_mean(pad_128(q))
    q = pad_128(q)
    qm = q.mean(dim=-2, keepdim=True)
    return q - qm, qm


@triton.jit
def _mean_correction_kernel(
    qm, k, out, rows: tl.constexpr, tokens: tl.constexpr, dim: tl.constexpr,
    tile_m: tl.constexpr, tile_n: tl.constexpr,
):
    tiles_n = tl.cdiv(tokens, tile_n)
    tiles_m = tl.cdiv(rows, tile_m)
    pid = tl.program_id(0)
    head = (pid // (tiles_m * tiles_n)).to(tl.int64)
    m = (pid // tiles_n % tiles_m) * tile_m + tl.arange(0, tile_m)
    n = (pid % tiles_n) * tile_n + tl.arange(0, tile_n)
    d = tl.arange(0, dim)
    q_block = tl.load(qm + (head * rows + m[:, None]) * dim + d[None, :],
                      mask=m[:, None] < rows, other=0)
    k_block = tl.load(k + (head * tokens + n[:, None]) * dim + d[None, :],
                      mask=n[:, None] < tokens, other=0)
    # Match the BF16 matmul rounding before storing the FP32 correction for TMA.
    correction = tl.dot(q_block, tl.trans(k_block)).to(tl.bfloat16).to(tl.float32)
    tl.store(out + (head * rows + m[:, None]) * tokens + n[None, :], correction,
             mask=(m[:, None] < rows) & (n[None, :] < tokens))


def _mean_correction(qm, k):
    batch, heads, blocks, dim = qm.shape
    # Group Q rows for each KV head so the matmul doesn't expand K.
    qm = qm.reshape(batch, k.size(1), -1, dim)
    if (sys.platform == "win32" and qm.is_cuda and qm.device == k.device
            and qm.dtype == k.dtype == torch.bfloat16 and dim in (64, 128)
            and qm.numel() and k.numel() and qm.is_contiguous() and k.is_contiguous()
            and not torch.is_autocast_enabled()
            and not (torch.is_grad_enabled() and (qm.requires_grad or k.requires_grad))
            and torch.cuda.get_device_capability(qm.device) == (12, 0)):
        rows, tokens = qm.size(2), k.size(2)
        out = torch.empty((batch, heads, blocks, tokens), device=qm.device, dtype=torch.float32)
        tile_m, tile_n = 32, 64
        grid = (batch * k.size(1) * triton.cdiv(rows, tile_m) * triton.cdiv(tokens, tile_n),)
        with torch.cuda.device(qm.device):
            _mean_correction_kernel[grid](qm, k, out, rows, tokens, dim, tile_m, tile_n, num_warps=4)
        return out
    return (qm @ k.transpose(-2, -1)).reshape(batch, heads, blocks, k.size(2)).float().contiguous()


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    k = _center_k(k)
    q, qm = _center_q(q, per_block_mean)
    return q, k, pad_128(v), _mean_correction(qm, k)

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _quantize_q(x)


def _quantize_q(x: torch.Tensor, mean=None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    N = triton.cdiv(N, _BLOCK_SIZE) * _BLOCK_SIZE
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _ops.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1, mean)
    return packed_fp4, fp8_scale


def _prepare_q(q, per_block_mean):
    # The quantizer vector-loads 16 elements. Unaligned views use Triton centering first.
    aligned = q.stride(-1) == 1 and q.storage_offset() % 16 == 0 and all(s % 16 == 0 for s in q.stride()[:-1])
    if not per_block_mean or not aligned:
        q, mean = _center_q(q, per_block_mean)
        return scale_and_quant_fp4(q), mean
    batch, heads, tokens, dim = q.shape
    mean = torch.empty((batch, heads, triton.cdiv(tokens, _BLOCK_SIZE), dim), device=q.device, dtype=torch.float32)
    return _quantize_q(q, mean), mean.to(q.dtype)

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    N = triton.cdiv(N, _BLOCK_SIZE) * _BLOCK_SIZE
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _ops.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    N = triton.cdiv(N, _BLOCK_SIZE) * _BLOCK_SIZE
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _ops.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         softmax_scale: float | None = None,
                        ):
    if softmax_scale is None:
        softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return _ops.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(
    q, k, v, attn_mask=None, is_causal=False, per_block_mean=True, *,
    sm_scale=None, scale=None, q_chunk_size=None, **kwargs,
):
    """Run SageAttention 3, routing calibrated eager shapes to SageAttention 2.2."""
    if sm_scale is None:
        sm_scale = scale
    if (
        _SAGE2_ROUTING_SUPPORTED and attn_mask is None and not is_causal
        and per_block_mean and q_chunk_size is None and not torch.compiler.is_compiling()
        and _sage2_shape_matches(q, k, v)
    ):
        sage2 = _sage2_for_device(q.device)
        if sage2 is not None:
            with torch.cuda.device(q.device):
                return sage2(q, k, v, sm_scale=sm_scale)
    return _sageattn3_blackwell(q, k, v, attn_mask, is_causal, per_block_mean, sm_scale, q_chunk_size)


def _sageattn3_blackwell(q, k, v, attn_mask, is_causal, per_block_mean, sm_scale, q_chunk_size):
    q_len, k_len, dim = q.size(2), k.size(2), q.size(-1)
    heads, kv_heads = q.size(1), k.size(1)
    if not kv_heads or not heads or heads % kv_heads or v.size(1) != kv_heads:
        raise ValueError("K and V must have matching head counts that divide the Q head count")
    if is_causal and attn_mask is None and q_len > k_len:
        raise ValueError("Causal attention requires query length <= key length")
    # Separately padded Q and K would shift a right-aligned mask by the padding difference.
    needs_causal_bias = is_causal and q_len % _BLOCK_SIZE != k_len % _BLOCK_SIZE
    if attn_mask is not None or dim > 128 or not q_len or not k_len or needs_causal_bias:
        if attn_mask is None and is_causal and q_len < k_len:
            attn_mask = causal_lower_right(q_len, k_len)
            is_causal = False
        return sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                    scale=sm_scale, enable_gqa=heads != kv_heads)
    if k.size(2) != v.size(2) or k.size(-1) != dim or v.size(-1) != dim:
        raise ValueError("K and V must have matching lengths, and Q, K and V must have matching head dimensions")

    padded_dim = 64 if dim <= 64 else 128
    if dim != padded_dim:
        q, k, v = (F.pad(x, (0, padded_dim - dim)) for x in (q, k, v))
    scale = dim ** -0.5 if sm_scale is None else sm_scale
    bf16 = q.dtype == torch.bfloat16
    padded_k = triton.cdiv(k_len, _BLOCK_SIZE) * _BLOCK_SIZE
    max_groups = _INT32_MAX // (heads * padded_k)
    groups = triton.cdiv(q_len, _BLOCK_SIZE) if per_block_mean else 1
    if q_chunk_size is not None or groups > max_groups:
        if is_causal or not per_block_mean or max_groups == 0:
            raise ValueError("Q chunking requires non-causal per-block-mean attention and a valid TMA stride")
        if q_chunk_size is None:
            q_chunk_size = min(8, max_groups) * _BLOCK_SIZE
        if not isinstance(q_chunk_size, int) or not 0 < q_chunk_size <= max_groups * _BLOCK_SIZE or q_chunk_size % _BLOCK_SIZE:
            raise ValueError("q_chunk_size must be a positive multiple of 128 that fits the TMA stride")

    k = _center_k(k)
    packed_k = scale_and_quant_fp4_permute(k)
    packed_v = scale_and_quant_fp4_transpose(v)

    def attention(q_chunk):
        packed_q, qm = _prepare_q(q_chunk, per_block_mean)
        return blockscaled_fp4_attn(
            packed_q, packed_k, packed_v, _mean_correction(qm, k),
            k_len, is_causal, per_block_mean, bf16, scale,
        )[0]

    if q_chunk_size is None:
        return attention(q)[:, :, :q_len, :dim].contiguous()
    output = torch.empty((*q.shape[:-1], dim), device=q.device, dtype=q.dtype)
    for start in range(0, q_len, q_chunk_size):
        end = min(start + q_chunk_size, q_len)
        output[:, :, start:end] = attention(q[:, :, start:end])[:, :, :end - start, :dim]
    return output
