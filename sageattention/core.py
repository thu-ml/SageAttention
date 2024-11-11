import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .quant_per_block_varlen import per_block_int8 as per_block_int8_varlen
from .quant_per_block_hd96 import per_block_int8_hd96
from .attn_qk_int8_per_block_h96 import forward as attn_h96_false
from .attn_qk_int8_per_block_h96_causal import forward as attn_h96_true
from .attn_qk_int8_per_block import forward as attn_true
from .attn_qk_int8_per_block_causal import forward as attn_false
from .attn_qk_int8_block_varlen import forward as attn_false_varlen
from .attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

def sageattn(q, k, v, tensor_layout="HND", attn_mask=None, dropout_p=0.0, is_causal=False, sm_scale=None, smooth_k=True):

    dtype = q.dtype

    headdim = q.size(-1)
    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k -= km
    else:
        km = None

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    if headdim == 96:
        q_int8, q_scale, k_int8, k_scale = per_block_int8_hd96(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout)
        if is_causal:
            return attn_h96_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
        else:
            return attn_h96_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout)

    if is_causal:
        o = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
    else:
        o = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    return o

def sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, attn_mask=None, dropout_p=0.0, is_causal=False, sm_scale=None, smooth_k=True):
    
    dtype = q.dtype

    head_dim = q.size(-1)
    assert head_dim in [64, 128], "varlen only support head_dim [64, 128]."

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(dim=0, keepdim=True) # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k -= km

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    return o