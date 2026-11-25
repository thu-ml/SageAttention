import pytest
import torch

from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda, sageattn_varlen
from sageattention.quant import per_channel_fp8, per_warp_int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_recomputes_attention_after_input_change():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("exercises the SM120 FP8 dispatch")
    torch.manual_seed(24)
    q, k, v = [torch.randn((1, 4, 129, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        sageattn(q, k, v, is_causal=True)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        result = sageattn(q, k, v, is_causal=True)
    graph.replay()
    original = result.clone()
    v.neg_()
    graph.replay()
    expected = sageattn(q, k, v, is_causal=True)
    assert not torch.equal(result, original)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm120_dispatch_preserves_smooth_k_argument():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("exercises the SM120 FP8 dispatch")
    torch.manual_seed(25)
    q, k, v = [torch.randn((1, 4, 257, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    k.add_(8)
    actual = sageattn(q, k, v, smooth_k=False)
    expected = sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, smooth_k=False, qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp16",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("q_len,k_len", [(9, 9), (1001, 503)])
def test_cuda_tail_tiles_are_finite(q_len, k_len):
    torch.manual_seed(26)
    q, k, v = [torch.randn((2, 4, n, 128), device="cuda", dtype=torch.float16)
               for n in (q_len, k_len, k_len)]
    result = sageattn(q, k, v).float()
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    assert torch.isfinite(result).all()
    assert (result - reference).norm() / reference.norm() < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sdpa_scale_alias_dense_and_varlen():
    torch.manual_seed(27)
    q, k, v = [torch.randn((1, 2, 129, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    actual = sageattn(q, k, v, scale=0.2)
    torch.testing.assert_close(actual, sageattn(q, k, v, sm_scale=0.2), rtol=0, atol=0)
    assert not torch.equal(actual, sageattn(q, k, v))

    q, k, v = (x[0].transpose(0, 1).contiguous() for x in (q, k, v))
    lengths = torch.tensor([0, 129], dtype=torch.int32, device=q.device)
    args = q, k, v, lengths, lengths, 129, 129
    actual = sageattn_varlen(*args, scale=0.2)
    torch.testing.assert_close(actual, sageattn_varlen(*args, sm_scale=0.2), rtol=0, atol=0)
    assert not torch.equal(actual, sageattn_varlen(*args))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("smooth_v", [False, True])
def test_fp8_value_quantization_handles_zero_channels(smooth_v):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires FP8 support")
    v = torch.zeros(1, 2, 117, 64, device="cuda", dtype=torch.float16)
    packed, scale, _ = per_channel_fp8(v, smooth_v=smooth_v)
    assert torch.count_nonzero(scale).item() == 0
    assert torch.count_nonzero(packed.float()).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("return_lse", [0, 1])
def test_dispatcher_fp8_ops_compile_with_mutable_outputs(return_lse):
    if torch.cuda.get_device_capability() not in {(8, 9), (12, 0), (12, 1)}:
        pytest.skip("requires the SM89/120/121 FP8 path")
    from sageattention.sm89_compile import qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf

    def attention(q, k, v):
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k)
        v_fp8, v_scale, _ = per_channel_fp8(v, scale_max=2.25, smooth_v=False)
        out = torch.empty_like(q)
        lse = qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
            1, 0, 2, q.size(-1) ** -0.5, return_lse,
        )
        return out, lse

    torch.manual_seed(28)
    q, k, v = [torch.randn(1, 2, 129, 64, dtype=torch.float16, device="cuda") for _ in range(3)]
    compiled = torch.compile(attention, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v), attention(q, k, v), rtol=0, atol=0)
