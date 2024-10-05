import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .attn_qk_int8_per_block_h64 import forward as attn_h64_false
from .attn_qk_int8_per_block_hd128 import forward as attn_h128_false
from .attn_qk_int8_per_block_hd64_causal import forward as attn_h64_true
from .attn_qk_int8_per_block_hd128_causal import forward as attn_h128_true


def sageattn(q, k, v, is_causal=False, smooth_k=True):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if smooth_k:
        k -= k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    
    if is_causal==False:
        if headdim==64:
            o = attn_h64_false(q_int8, k_int8, v, q_scale, k_scale)

        elif headdim==128:
            o = attn_h128_false(q_int8, k_int8, v, q_scale, k_scale)

    elif is_causal==True:
        if headdim==64:
            o = attn_h64_true(q_int8, k_int8, v, q_scale, k_scale)

        elif headdim==128:
            o = attn_h128_true(q_int8, k_int8, v, q_scale, k_scale)

    return o
