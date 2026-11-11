import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6),
    reason="sageattn selects the Triton backend on SM86",
)
def test_sageattn_smooth_k_matches_triton():
    from sageattention import sageattn, sageattn_qk_int8_pv_fp16_triton

    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = torch.randn(3, 1, 2, 128, 64, device="cuda", dtype=torch.float16, generator=generator)
    # A shared key offset makes mean subtraction matter for int8 quantization.
    k = k + 8.0

    smoothed = sageattn_qk_int8_pv_fp16_triton(q, k, v, smooth_k=True)
    unsmoothed = sageattn_qk_int8_pv_fp16_triton(q, k, v, smooth_k=False)
    assert not torch.equal(smoothed, unsmoothed)
    torch.testing.assert_close(sageattn(q, k, v), smoothed, rtol=0, atol=0)
    torch.testing.assert_close(sageattn(q, k, v, smooth_k=False), unsmoothed, rtol=0, atol=0)
