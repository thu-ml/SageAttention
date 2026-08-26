import os
import unittest

import torch

from sageattn3 import api


class TestLongSequenceChunking(unittest.TestCase):
    def test_h3_shape_requires_chunking(self):
        q = torch.empty((1, 56, 73_774, 128), device="meta")
        k = torch.empty_like(q)

        self.assertEqual(api._delta_s_batch_stride(q, k), 2_386_435_072)
        self.assertTrue(api._requires_delta_s_chunking(q, k))

    def test_typical_shape_keeps_full_path(self):
        q = torch.empty((1, 24, 8_192, 128), device="meta")
        k = torch.empty_like(q)

        self.assertFalse(api._requires_delta_s_chunking(q, k))

    def test_rejects_misaligned_chunk_size(self):
        with self.assertRaises(ValueError):
            api._normalize_q_chunk_size(1000, heads=56, padded_k_tokens=73_856)

    def test_rejects_chunk_that_still_overflows_tma_stride(self):
        with self.assertRaises(ValueError):
            api._normalize_q_chunk_size(
                73_728,
                heads=56,
                padded_k_tokens=73_856,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_chunked_matches_full_per_block_mean(self):
        torch.manual_seed(42)
        shape = (1, 8, 4_097, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        expected = api._sageattn3_blackwell_full(
            q,
            k.clone(),
            v,
            is_causal=False,
            per_block_mean=True,
        )
        actual = api.sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=False,
            per_block_mean=True,
            q_chunk_size=512,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(
        os.getenv("SAGEATTN3_RUN_LONG_TEST") == "1",
        "set SAGEATTN3_RUN_LONG_TEST=1 to run the 73,774-token regression",
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_h3_shape_uses_automatic_chunking(self):
        torch.manual_seed(42)
        shape = (1, 56, 73_774, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        output = api.sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=False,
            per_block_mean=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(output.shape, q.shape)
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
