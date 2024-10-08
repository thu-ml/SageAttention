import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .quant_per_block_hd96 import per_block_int8_hd96

from .attn_qk_int8_per_block_h64 import forward as attn_h64_false
from .attn_qk_int8_per_block_hd128 import forward as attn_h128_false
from .attn_qk_int8_per_block_hd64_causal import forward as attn_h64_true
from .attn_qk_int8_per_block_hd128_causal import forward as attn_h128_true
from .attn_qk_int8_per_block_h96 import forward as attn_h96_false
from .attn_qk_int8_per_block_h96_causal import forward as attn_h96_true

from .attn_qk_int8_per_block_h64_bf16 import forward as attn_h64_false_bf16
from .attn_qk_int8_per_block_hd128_bf16 import forward as attn_h128_false_bf16
from .attn_qk_int8_per_block_hd64_causal_bf16 import forward as attn_h64_true_bf16
from .attn_qk_int8_per_block_hd128_causal_bf16 import forward as attn_h128_true_bf16
from .attn_qk_int8_per_block_h96_bf16 import forward as attn_h96_false_bf16
from .attn_qk_int8_per_block_h96_causal_bf16 import forward as attn_h96_true_bf16


def sageattn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if smooth_k:
        k -= k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)
    dtype = q.dtype


    if dtype == torch.float16:

        if headdim==96:
            q_int8, q_scale, k_int8, k_scale = per_block_int8_hd96(q, k)
        else:
            q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
        
        if is_causal==False:
            if headdim==64:
                o = attn_h64_false(q_int8, k_int8, v, q_scale, k_scale)

            if headdim==96:
                o = attn_h96_false(q_int8, k_int8, v, q_scale, k_scale)

            elif headdim==128:
                o = attn_h128_false(q_int8, k_int8, v, q_scale, k_scale)

        elif is_causal==True:
            if headdim==64:
                o = attn_h64_true(q_int8, k_int8, v, q_scale, k_scale)
            
            if headdim==96:
                o = attn_h96_true(q_int8, k_int8, v, q_scale, k_scale)

            elif headdim==128:
                o = attn_h128_true(q_int8, k_int8, v, q_scale, k_scale)


    elif dtype == torch.bfloat16:

        if headdim==96:
            q_int8, q_scale, k_int8, k_scale = per_block_int8_hd96(q, k)
        else:
            q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
        
        if is_causal==False:
            if headdim==64:
                o = attn_h64_false_bf16(q_int8, k_int8, v, q_scale, k_scale)

            if headdim==96:
                o = attn_h96_false_bf16(q_int8, k_int8, v, q_scale, k_scale)

            elif headdim==128:
                o = attn_h128_false_bf16(q_int8, k_int8, v, q_scale, k_scale)

        elif is_causal==True:
            if headdim==64:
                o = attn_h64_true_bf16(q_int8, k_int8, v, q_scale, k_scale)
            
            if headdim==96:
                o = attn_h96_true_bf16(q_int8, k_int8, v, q_scale, k_scale)

            elif headdim==128:
                o = attn_h128_true_bf16(q_int8, k_int8, v, q_scale, k_scale)


    return o
