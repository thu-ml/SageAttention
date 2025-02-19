from sageattention import sageattn_qk_int8_pv_fp8_cuda_dsk_sm90 as sageattn_qk_int8_pv_fp8_cuda_sm90a
from torch.nn.functional import scaled_dot_product_attention as sdpa

import torch
import numpy as np
import os
import argparse


def precision_cmp_torch(t1: torch.Tensor, t2: torch.Tensor):
    x, xx = t1.to(dtype=torch.float32), t2.to(dtype=torch.float32)
    # 重塑张量并计算余弦相似度
    x_reshaped = torch.reshape(x, [1, -1])
    xx_reshaped = torch.reshape(xx, [1, -1])
    sim = torch.nn.functional.cosine_similarity(x_reshaped, xx_reshaped).item()
    
    # 计算 L1 误差
    l1 = (torch.abs(x - xx).sum() / torch.abs(xx).sum()).item()

    max_diff = torch.max(x - xx)
    
    return sim, l1, max_diff

bsz = 2
seq_len = 1026
num_heads = 128
head_dim_qk = 128 + 64
head_dim_v = 128

tensor_layout = "NHD"
is_causal = True
return_lse = False

torch.backends.cuda.enable_flash_sdp(True)

q = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
k = torch.randn((bsz, seq_len, num_heads, head_dim_qk), dtype=torch.float16).cuda()
v = torch.randn((bsz, seq_len, num_heads, head_dim_v), dtype=torch.float16).cuda()

# to HND for sdpa input
q = q.transpose(2, 1)
k = k.transpose(2, 1)
v = v.transpose(2, 1)

o_torch_fa2 = sdpa(q, k, v, is_causal=is_causal)
torch.cuda.synchronize()

torch.backends.cuda.enable_flash_sdp(False)
o_torch_sdpa = sdpa(q, k, v, is_causal=is_causal)
torch.cuda.synchronize()

head_dim_og = head_dim_qk
sm_scale = head_dim_og**-0.5

# to NHD for sageattn input
q = q.transpose(2, 1)
k = k.transpose(2, 1)
v = v.transpose(2, 1)
o_sa = sageattn_qk_int8_pv_fp8_cuda_sm90a(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", return_lse=return_lse, pv_accum_dtype="fp32+fp32")
torch.cuda.synchronize()

sim, l1, max_diff = precision_cmp_torch(o_torch_fa2.transpose(2, 1), o_sa)
print(f"Sim and Diff of Sage Attn: {sim}, {max_diff}")

sim, l1, max_diff = precision_cmp_torch(o_torch_fa2, o_torch_sdpa)
print(f"Sim and Diff of Flash Attn: {sim}, {max_diff}")
