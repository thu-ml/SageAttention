import unittest

import torch
import torch.nn.functional as F

try:
    from sageattn3.api import preprocess_qkv, triton_group_mean
except ModuleNotFoundError as error:
    if error.name not in ("fp4attn_cuda", "fp4quant_cuda"):
        raise
    preprocess_qkv = None
    triton_group_mean = None


def reference_preprocess_qkv(q, k, v, per_block_mean):
    def pad_128(x):
        pad_len = (128 - x.size(2) % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    k = k.clone()
    k -= k.mean(dim=-2, keepdim=True)
    q, k, v = map(pad_128, (q, k, v))
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


@unittest.skipUnless(
    torch.cuda.is_available() and preprocess_qkv is not None,
    "requires a CUDA build of SageAttention3",
)
class TestFusedPreprocess(unittest.TestCase):
    def test_matches_reference_without_mutating_inputs(self):
        shapes = (
            (250, 250, 64),
            (256, 256, 128),
            (250, 330, 128),
            (129, 257, 64),
            (4100, 4200, 128),
        )
        for dtype in (torch.bfloat16, torch.float16):
            for q_len, kv_len, head_dim in shapes:
                for per_block_mean in (True, False):
                    with self.subTest(
                        dtype=dtype,
                        q_len=q_len,
                        kv_len=kv_len,
                        head_dim=head_dim,
                        per_block_mean=per_block_mean,
                    ):
                        torch.manual_seed(q_len + kv_len + head_dim)
                        q_storage = torch.randn(
                            1,
                            q_len,
                            2,
                            2 * head_dim,
                            device="cuda",
                            dtype=dtype,
                        )
                        kv_storage = torch.randn(
                            1,
                            kv_len,
                            2,
                            2 * head_dim,
                            device="cuda",
                            dtype=dtype,
                        )
                        q = q_storage[..., :head_dim].transpose(1, 2)
                        k = kv_storage[..., :head_dim].transpose(1, 2)
                        v = kv_storage[..., head_dim:].transpose(1, 2)
                        inputs_before = tuple(x.clone() for x in (q, k, v))

                        expected = reference_preprocess_qkv(
                            q, k, v, per_block_mean
                        )
                        actual = preprocess_qkv(q, k, v, per_block_mean)
                        torch.cuda.synchronize()

                        for actual_tensor, expected_tensor in zip(actual, expected):
                            torch.testing.assert_close(
                                actual_tensor,
                                expected_tensor,
                                rtol=0,
                                atol=0,
                            )
                        for tensor, before in zip((q, k, v), inputs_before):
                            torch.testing.assert_close(tensor, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
