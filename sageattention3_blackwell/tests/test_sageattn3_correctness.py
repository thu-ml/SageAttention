"""sageattn3_blackwell against float32 SDPA on random inputs (needs a Blackwell GPU)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="needs a Blackwell GPU",
)


def relative_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).norm() / ref.float().norm())


def random_qkv(length: int, head_dim: int, key_scale: float = 1.0) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(2, 8, length, head_dim, device="cuda", dtype=torch.bfloat16)
    k = (torch.randn(2, 8, length, head_dim, device="cuda") * key_scale).to(torch.bfloat16)
    v = torch.randn(2, 8, length, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [40, 100, 128, 147, 333])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_matches_sdpa(causal: bool, length: int, head_dim: int) -> None:
    from sageattn3 import sageattn3_blackwell

    q, k, v = random_qkv(length, head_dim)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    out = sageattn3_blackwell(q, k, v, is_causal=causal)
    # FP4 Q, K, P and V leave a relative L2 of about 0.19 on random inputs, in every block of 32 rows
    assert relative_l2(out, ref) < 0.25
    for start in range(0, length, 32):
        assert relative_l2(out[:, :, start:start + 32], ref[:, :, start:start + 32]) < 0.3


@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa_with_more_query_tiles_than_sms(causal: bool) -> None:
    from sageattn3 import sageattn3_blackwell

    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 16, 1024, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    out = sageattn3_blackwell(q, k, v, is_causal=causal)
    # 512 query tiles: the persistent kernel runs several of them on each SM
    assert relative_l2(out, ref) < 0.25


def test_causal_row_with_one_visible_key_returns_its_value() -> None:
    from sageattn3 import sageattn3_blackwell

    q, k, v = random_qkv(147, 128, key_scale=8.0)  # large logits on the masked keys of row 0
    out = sageattn3_blackwell(q, k, v, is_causal=True)
    # row 0 sees key 0 only, so it returns V[0] up to the FP4 rounding of V (about 0.1)
    assert relative_l2(out[:, :, 0], v[:, :, 0]) < 0.15
