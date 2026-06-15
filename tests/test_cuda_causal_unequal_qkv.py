import os

import pytest
import torch

from sageattention import sageattn_qk_int8_pv_fp8_cuda, sageattn_varlen


def _require_sm89_or_newer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 89:
        pytest.skip("FP8 CUDA kernel requires SM89 or newer")


def _index_value_tensors(qo_len, kv_len, *, dtype=torch.bfloat16):
    q = torch.ones(1, qo_len, 1, 128, device="cuda", dtype=dtype)
    k = torch.ones(1, kv_len, 1, 128, device="cuda", dtype=dtype)
    v_index = torch.arange(kv_len, device="cuda", dtype=torch.float32) / max(kv_len - 1, 1)
    v_index = v_index.to(dtype)
    v = v_index.view(1, kv_len, 1, 1).expand(1, kv_len, 1, 128).contiguous()
    return q, k, v


def _varlen_cu_seqlens(qo_len, kv_len):
    cu_q = torch.tensor([0, qo_len], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    return cu_q, cu_k


def _index_value_tensors_varlen(qo_len, kv_len, *, dtype=torch.bfloat16):
    q = torch.ones(qo_len, 1, 128, device="cuda", dtype=dtype)
    k = torch.ones(kv_len, 1, 128, device="cuda", dtype=dtype)
    v_index = torch.arange(kv_len, device="cuda", dtype=torch.float32) / max(kv_len - 1, 1)
    v_index = v_index.to(dtype)
    v = v_index.view(kv_len, 1, 1).expand(kv_len, 1, 128).contiguous()
    return q, k, v


@pytest.mark.parametrize("qo_len,kv_len", [(5, 9), (128, 129), (33, 129)])
def test_fp8_cuda_causal_uses_right_aligned_mask_for_unequal_qkv(qo_len, kv_len):
    _require_sm89_or_newer()
    q, k, v = _index_value_tensors(qo_len, kv_len)

    out = sageattn_qk_int8_pv_fp8_cuda(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=True,
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp32+fp16",
        smooth_k=False,
    )
    torch.cuda.synchronize()

    allowed_counts = torch.arange(kv_len - qo_len + 1, kv_len + 1, device="cuda", dtype=torch.float32)
    expected = (allowed_counts - 1.0) / (2.0 * (kv_len - 1))
    observed = out[0, :, 0, 0].float()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.02)


def test_fp8_cuda_causal_rejects_more_queries_than_keys():
    _require_sm89_or_newer()
    q, k, v = _index_value_tensors(9, 5)

    with pytest.raises(ValueError, match="qo_len <= kv_len"):
        sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=True,
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp16",
            smooth_k=False,
        )


@pytest.mark.parametrize("qo_len,kv_len", [(5, 9), (33, 129), (128, 1024)])
def test_varlen_causal_uses_right_aligned_mask_for_unequal_qkv(qo_len, kv_len):
    _require_sm89_or_newer()
    q, k, v = _index_value_tensors_varlen(qo_len, kv_len)
    cu_q, cu_k = _varlen_cu_seqlens(qo_len, kv_len)

    out = sageattn_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        qo_len,
        kv_len,
        is_causal=True,
        smooth_k=False,
    )
    torch.cuda.synchronize()

    allowed_counts = torch.arange(kv_len - qo_len + 1, kv_len + 1, device="cuda", dtype=torch.float32)
    expected = (allowed_counts - 1.0) / (2.0 * (kv_len - 1))
    observed = out[:, 0, 0].float()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.02)


def test_varlen_causal_rejects_more_queries_than_keys():
    _require_sm89_or_newer()
    q, k, v = _index_value_tensors_varlen(9, 5)
    cu_q, cu_k = _varlen_cu_seqlens(9, 5)

    with pytest.raises(ValueError, match="qo_len <= kv_len"):
        sageattn_varlen(
            q,
            k,
            v,
            cu_q,
            cu_k,
            9,
            5,
            is_causal=True,
            smooth_k=False,
        )


@pytest.mark.parametrize("is_causal", [False, True])
def test_varlen_attention_is_finite_for_large_finite_scores(is_causal):
    _require_sm89_or_newer()
    qo_len, kv_len = 33, 129
    q = torch.full((qo_len, 1, 128), 1e20, device="cuda", dtype=torch.bfloat16)
    k = torch.full((kv_len, 1, 128), 1e20, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_len, 1, 128, device="cuda", dtype=torch.bfloat16)
    cu_q, cu_k = _varlen_cu_seqlens(qo_len, kv_len)

    out = sageattn_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        qo_len,
        kv_len,
        is_causal=is_causal,
        smooth_k=False,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(q).all()
    assert torch.isfinite(k).all()
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    os.getenv("SAGEATTN_RUN_SLOW_TESTS") != "1",
    reason="set SAGEATTN_RUN_SLOW_TESTS=1 to run large-shape smoke",
)
def test_fp8_cuda_causal_user_reported_shape_is_finite():
    _require_sm89_or_newer()
    q, k, v = _index_value_tensors(4129, 36385)

    out = sageattn_qk_int8_pv_fp8_cuda(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=True,
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp32+fp16",
        smooth_k=False,
    )
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    os.getenv("SAGEATTN_RUN_SLOW_TESTS") != "1",
    reason="set SAGEATTN_RUN_SLOW_TESTS=1 to run large-shape smoke",
)
def test_varlen_causal_user_reported_shape_is_finite():
    _require_sm89_or_newer()
    qo_len, kv_len = 4129, 36385
    q, k, v = _index_value_tensors_varlen(qo_len, kv_len)
    cu_q, cu_k = _varlen_cu_seqlens(qo_len, kv_len)

    out = sageattn_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        qo_len,
        kv_len,
        is_causal=True,
        smooth_k=False,
    )
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
