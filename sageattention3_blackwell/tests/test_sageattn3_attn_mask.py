"""sageattn3_blackwell called with an attn_mask (needs a Blackwell GPU)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="needs a Blackwell GPU",
)


def test_attn_mask_falls_back_to_sdpa() -> None:
    from sageattn3 import sageattn3_blackwell

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mask = torch.ones(64, 64, dtype=torch.bool, device="cuda")
    mask[:, 48:] = False  # padded keys
    with pytest.warns(UserWarning, match="attn_mask"):
        out = sageattn3_blackwell(q, k, v, attn_mask=mask)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
