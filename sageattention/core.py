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

from .attn_qk_int8_per_block import forward as attn_true
from .attn_qk_int8_per_block_causal import forward as attn_false

def sageattn_old(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True):
    assert q.size(-2)>=128, "seq_len should be not less than 128."

    dtype = q.dtype

    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.bfloat16)

    if smooth_k:
        k -= k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]. Other headdim < 128 can be processed by padding with 0."

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

def sageattn(q, k, v, tensor_layout="HND", attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True):

    dtype = q.dtype

    headdim = q.size(-1)

    if headdim == 96 or (q.dtype != torch.float16 and q.dtype != torch.bfloat16):
        sageattn_old(q, k, v, attn_mask, dropout_p, is_causal, scale, smooth_k)

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        k -= k.mean(dim=seq_dim, keepdim=True)

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, tensor_layout=tensor_layout)

    if is_causal:
        o = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
    else:
        o = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    return o