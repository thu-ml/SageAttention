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

Modified (perf #4, final form): single-pass Q quantization.
- The fp4 quant kernel reads UNPADDED q once: it computes the per-128-block
  mean in-kernel (atomic smem reduction), subtracts it before quantizing, and
  emits qm to gmem (float4 store, verified by test_qm_emission). The old
  pad-128 copy and the triton group-mean pass are gone.
- Quant OUTPUTS are allocated 128-padded (the kernel writes full 128-token
  blocks; a ragged allocation would be an OOB write — that was the root cause
  of the "unpadded-Q" attention corruption). Pad rows come out as exact zeros
  and are dropped by the [:QL] output slice.
- K/V are zero-padded to 128 in torch before the permute/transpose quant
  (those bindings require 128-multiple token dims; cheap for GQA).
- delta_s is built from the kernel-emitted qm (cast to bf16, matching the old
  triton qm dtype) against the padded K via a broadcast matmul over (Hkv, G) —
  no repeat_interleave K expansion.
"""
import torch
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    # center K globally (in-place; small for GQA). Q's per-block mean is folded
    # into the quant kernel (qm emitted there).
    k -= k.mean(dim=-2, keepdim=True)
    return q, k, v


def _pad_tokens_128(x: torch.Tensor) -> torch.Tensor:
    N = x.size(2)
    pad = (-N) % 128
    return torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x.contiguous()


def scale_and_quant_fp4(x: torch.Tensor, qmean: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    N_pad = ((N + 127) // 128) * 128  # kernel writes full 128-token blocks
    packed_fp4 = torch.empty((B, H, N_pad, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N_pad, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1, qmean)  # qmean=None -> nullopt
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
    if q.size(-1) > 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v = preprocess_qkv(q, k, v, per_block_mean)
    if per_block_mean:
        B, H, L, D = q.shape
        # Single pass: in-kernel block mean + subtract + quantize + qm emission.
        # Padded output allocations (kernel writes full blocks; pad rows come
        # out as exact zeros and are sliced away after attention).
        n_groups = (L + 127) // 128
        qm = torch.empty(B, H, n_groups, D, device=q.device, dtype=torch.float32)
        qlist_from_cuda = scale_and_quant_fp4(q, qm)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        qlist_from_cuda = scale_and_quant_fp4(q - qm)
    kp = _pad_tokens_128(k)
    klist_from_cuda = scale_and_quant_fp4_permute(kp)
    vlist_from_cuda = scale_and_quant_fp4_transpose(_pad_tokens_128(v))
    # DS rows must equal ceil(seqlen_q / kBlockM) (the kernel tiles delta_s by
    # kBlockM=128 and indexes it by m_block); width follows the PADDED K extent
    # so the kernel's DS descriptor {seqlen_s, seqlen_k} matches. Kernel-emitted
    # qm is fp32 — cast to bf16, matching the old triton qm dtype.
    # GQA without expansion: ds[h] = qm[h] @ k'[h // G]^T as one strided batched
    # gemm ((B*Hkv, G*ng, D) @ (B*Hkv, D, S), kp as a transposed view) — no
    # repeat_interleave K materialization. Measured at kv=32k: 0.28ms vs 0.59ms
    # for the expansion path; a broadcast-bmm variant is 1.45ms (slow cublas
    # path — do not "simplify" to that).
    _g = q.shape[1] // k.shape[1]
    qm_b = qm.to(torch.bfloat16) if per_block_mean else qm
    if _g > 1:
        B_, Hkv_, ng_, D_ = kp.shape[0], kp.shape[1], qm_b.shape[2], qm_b.shape[3]
        delta_s = torch.matmul(qm_b.view(B_, Hkv_, _g * ng_, D_), kp.transpose(-2, -1))
        delta_s = delta_s.to(torch.float32).reshape(B_, Hkv_ * _g, ng_, kp.shape[2]).contiguous()
    else:
        delta_s = torch.matmul(qm_b, kp.transpose(-2, -1)).to(torch.float32).contiguous()
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
