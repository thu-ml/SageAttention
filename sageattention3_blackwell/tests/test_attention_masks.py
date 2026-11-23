import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(12, 0), (12, 1)},
    "SageAttention 3 mask tests require a supported Blackwell GPU",
)
class AttentionMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sageattn3 import api

        cls.api = api

    @staticmethod
    def identity_slabs(length, dim, dtype):
        eye = torch.eye(dim, device="cuda", dtype=dtype)
        for start in range(0, length, dim):
            width = min(dim, length - start)
            values = torch.zeros((1, 1, length, dim), device="cuda", dtype=dtype)
            values[0, 0, start : start + width, :width] = eye[:width, :width]
            yield values, width

    @torch.inference_mode()
    def test_causal_mask_keeps_the_previous_block(self):
        for dtype in (torch.float16, torch.bfloat16):
            for length, dim in ((259, 64), (512, 128)):
                with self.subTest(dtype=dtype, length=length, dim=dim):
                    torch.manual_seed(0)
                    q, k = [torch.randn(1, 1, length, dim, device="cuda", dtype=dtype) for _ in range(2)]
                    # Identity V recovers the effective attention matrix a slab at a time.
                    weights = torch.cat([
                        self.api.sageattn3_blackwell(q, k.clone(), v, is_causal=True)[0, 0, :, :width]
                        for v, width in self.identity_slabs(length, dim, dtype)
                    ], dim=-1).float()
                    self.assertEqual(torch.count_nonzero(weights.triu(1)).item(), 0)
                    future = torch.ones_like(weights, dtype=torch.bool).triu_(1)
                    scores = (q[0, 0].float() @ k[0, 0].float().T) * dim ** -0.5
                    reference = scores.masked_fill(future, float("-inf")).softmax(dim=-1)

                    prior_block = weights[128:256, :128].sum(dim=-1)
                    prior_ref = reference[128:256, :128].sum(dim=-1)
                    torch.testing.assert_close(prior_block, prior_ref, atol=0.10, rtol=0.15)
                    relative_l2 = (weights - reference).norm() / reference.norm()
                    self.assertLess(relative_l2.item(), 0.30)

    @torch.inference_mode()
    def test_padding_masks_permuted_columns(self):
        length, dim, dtype = 141, 64, torch.float16
        torch.manual_seed(0)
        q, k = [torch.randn(1, 1, length, dim, device="cuda", dtype=dtype) for _ in range(2)]
        q, k, _, correction = self.api.preprocess_qkv(q, k, torch.zeros_like(q))
        packed_q = self.api.scale_and_quant_fp4(q)
        packed_k = self.api.scale_and_quant_fp4_permute(k)
        # Put identity values in padded positions so any leaked weight is observable.
        weights = torch.cat([
            self.api.blockscaled_fp4_attn(
                packed_q, packed_k, self.api.scale_and_quant_fp4_transpose(v), correction,
                length, is_bf16=False,
            )[0][0, 0, :length, :width]
            for v, width in self.identity_slabs(k.size(-2), dim, dtype)
        ], dim=-1)
        self.assertEqual(torch.count_nonzero(weights[:, length:]).item(), 0)

    @torch.inference_mode()
    def test_gqa_matches_repeated_kv_and_preserves_input(self):
        torch.manual_seed(17)
        q, k, v = [torch.randn(2, h, n, 96, device="cuda", dtype=torch.bfloat16)
                   for h, n in ((6, 131), (2, 259), (2, 259))]
        original_k = k.clone()
        for causal in (False, True):
            with self.subTest(causal=causal):
                actual = self.api.sageattn3_blackwell(q, k, v, is_causal=causal)
                expected = self.api.sageattn3_blackwell(
                    q, k.repeat_interleave(3, dim=1), v.repeat_interleave(3, dim=1), is_causal=causal,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(k, original_k, rtol=0, atol=0)

    @torch.inference_mode()
    def test_ragged_chunking_matches_full_attention(self):
        torch.manual_seed(18)
        q, k, v = [torch.randn(1, h, n, 64, device="cuda", dtype=torch.float16)
                   for h, n in ((4, 513), (2, 397), (2, 397))]
        expected = self.api.sageattn3_blackwell(q, k, v)
        actual = self.api.sageattn3_blackwell(q, k, v, q_chunk_size=256)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for quantize in (self.api.scale_and_quant_fp4, self.api.scale_and_quant_fp4_permute,
                         self.api.scale_and_quant_fp4_transpose):
            packed, scales = quantize(q[..., :129, :].contiguous())
            pad_packed, pad_scales = quantize(self.api.pad_128(q[..., :129, :]))
            torch.testing.assert_close(packed, pad_packed, rtol=0, atol=0)
            torch.testing.assert_close(scales.view(torch.uint8), pad_scales.view(torch.uint8), rtol=0, atol=0)

        _, mean = self.api._prepare_q(q, True)
        expected = self.api.pad_128(q).reshape(1, 4, -1, 128, 64).float().mean(-2).to(q.dtype)
        torch.testing.assert_close(mean, expected, rtol=0.001, atol=1e-6)

    @torch.inference_mode()
    def test_explicit_mask_and_causal_length_guards(self):
        from torch.nn.attention.bias import causal_lower_right
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        torch.manual_seed(19)
        q, k, v = [torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16) for n in (17, 129, 129)]
        mask = torch.arange(129, device="cuda")[None, :] <= torch.arange(17, device="cuda")[:, None]
        for bias in (mask, torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))):
            expected = sdpa(q, k, v, attn_mask=bias, scale=0.2)
            actual = self.api.sageattn3_blackwell(q, k, v, attn_mask=bias, scale=0.2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual = self.api.sageattn3_blackwell(q, k, v, scale=0.2)
        torch.testing.assert_close(actual, self.api.sageattn3_blackwell(q, k, v, sm_scale=0.2), rtol=0, atol=0)
        expected = sdpa(q, k, v, attn_mask=causal_lower_right(17, 129))
        actual = self.api.sageattn3_blackwell(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "query length <= key length"):
            self.api.sageattn3_blackwell(k, q, q, is_causal=True)


if __name__ == "__main__":
    unittest.main()
