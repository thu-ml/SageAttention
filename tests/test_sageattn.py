#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from sageattention import sageattn
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_rtol_atol(actual, expect):
    actual = actual.float()
    expect = expect.float()
    diff = (actual - expect).abs()
    eps = torch.tensor(torch.finfo(actual.dtype).eps, device=actual.device, dtype=actual.dtype)
    rdiff = diff / torch.maximum(torch.maximum(actual.abs(), expect.abs()), eps)
    return (
        f"mean_rtol={rdiff.mean().item():.3g} "
        f"max_rtol={rdiff.max().item():.3g} "
        f"mean_atol={diff.max().item():.3g} "
        f"max_atol={diff.max().item():.3g}"
    )


def test_standard():
    print("\n--- Testing Standard Shape (B=4, H=32, N=64, D=128) ---")
    batch_size = 4
    head_num = 32
    seq_len = 64
    head_dim = 128
    dtype = torch.float16

    q = torch.randn(batch_size, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    print("q", tuple(q.shape), q.device, q.dtype)

    torch.backends.cuda.enable_math_sdp(True)
    with sdpa_kernel(SDPBackend.MATH):
        out_math = F.scaled_dot_product_attention(q, k, v)

    out_sage = sageattn(q, k, v)
    print("sage vs math:", get_rtol_atol(out_sage, out_math))
    assert not torch.isnan(out_sage).any(), "Standard shape output contains NaNs!"


def test_non_multiples():
    print("\n--- Testing Non-Multiple Shapes (Q_len=1001, KV_len=503) ---")
    batch_size = 2
    num_heads = 32
    qo_len = 1001
    kv_len = 503
    head_dim = 128
    dtype = torch.float16

    for layout in ["HND", "NHD"]:
        print(f"Testing layout: {layout}")
        if layout == "HND":
            q = torch.randn(batch_size, num_heads, qo_len, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn(batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
            
            torch.backends.cuda.enable_math_sdp(True)
            with sdpa_kernel(SDPBackend.MATH):
                out_math = F.scaled_dot_product_attention(q, k, v)
        else: # NHD
            q = torch.randn(batch_size, qo_len, num_heads, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch_size, kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
            v = torch.randn(batch_size, kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
            
            q_ref = q.transpose(1, 2)
            k_ref = k.transpose(1, 2)
            v_ref = v.transpose(1, 2)
            torch.backends.cuda.enable_math_sdp(True)
            with sdpa_kernel(SDPBackend.MATH):
                out_math_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
            out_math = out_math_ref.transpose(1, 2)

        out_sage = sageattn(q, k, v, tensor_layout=layout)
        print("sage vs math:", get_rtol_atol(out_sage, out_math))
        assert not torch.isnan(out_sage).any(), f"Non-multiple shape containing NaNs in layout {layout}!"


def test_short_sequences():
    print("\n--- Testing Short Sequence Lengths (1 to 65) ---")
    batch_size = 4
    num_heads = 32
    head_dim = 128
    dtype = torch.float16

    for seq_len in range(1, 66):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        
        out_sage = sageattn(q, k, v)
        nans = torch.isnan(out_sage).sum().item()
        assert nans == 0, f"NaNs found for seq_len={seq_len}! Count: {nans}"
        
    print("All short sequence lengths from 1 to 65 verified: 0 NaNs found.")


def main():
    test_standard()
    test_non_multiples()
    test_short_sequences()
    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()
