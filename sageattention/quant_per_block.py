import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(X, X_int8, BLK: tl.constexpr, Scale, L, C: tl.constexpr, scale_stride):
    off_b = tl.program_id(1) 
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C 
    offs_m = off_blk*BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk  

    x = tl.load(x_ptrs, mask=offs_m[:, None] < L)
    x *= (C**-0.5 * 1.44269504)
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def k_kernel_per_block_int8(X, X_int8, BLK: tl.constexpr, Scale, L, C: tl.constexpr, scale_stride):
    off_b = tl.program_id(1) 
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C 
    offs_m = off_blk*BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk  

    x = tl.load(x_ptrs, mask=offs_m[:, None] < L)
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8(q, k, BLKQ=128, BLKK=64):
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = q_int8.clone()

    if q.dim() == 3:
        q_scale = torch.empty((q.shape[-3], (q.shape[-2] + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((k.shape[-3], (k.shape[-2] + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32)
    elif q.dim() == 4:
        q_scale = torch.empty((q.shape[-4], q.shape[-3], (q.shape[-2] + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((k.shape[-4], k.shape[-3], (k.shape[-2] + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32)

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    B, L, C = q.shape
    grid = ((L+BLKQ-1)//BLKQ, B, )
    q_kernel_per_block_int8[grid](
        q, 
        q_int8,
        BLKQ,
        q_scale,
        L, C, q_scale.stride(0) if q_scale.dim() == 3 else q_scale.stride(1),
    )

    grid = ((L+BLKK-1)//BLKK, B, )
    k_kernel_per_block_int8[grid](
        k, 
        k_int8,
        BLKK,
        k_scale,
        L, C, k_scale.stride(0) if k_scale.dim() == 3 else k_scale.stride(1),
    )

    return q_int8, q_scale, k_int8, k_scale
