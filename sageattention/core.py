import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .quant_per_block_hd96 import per_block_int8_hd96
from .attn_qk_int8_per_block_h96 import forward as attn_h96_false
from .attn_qk_int8_per_block_h96_causal import forward as attn_h96_true
from .attn_qk_int8_per_block import forward as attn_true
from .attn_qk_int8_per_block_causal import forward as attn_false


def sageattn(q, k, v, tensor_layout="HND", attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True):

    dtype = q.dtype

    headdim = q.size(-1)
    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim)
    else:
        km = None

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    if headdim == 96:
        q_int8, q_scale, k_int8, k_scale = per_block_int8_hd96(q, k)
        if is_causal:
            return attn_h96_true(q_int8, k_int8, v, q_scale, k_scale)
        else:
            return attn_h96_false(q_int8, k_int8, v, q_scale, k_scale)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km, tensor_layout=tensor_layout)

    if is_causal:
        o = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
    else:
        o = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    return o