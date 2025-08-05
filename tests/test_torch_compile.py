import pytest
import torch
from torch.testing._internal.optests import fake_check

from sageattention.core import (
    SM80_ENABLED,
    SM89_ENABLED,
    SM90_ENABLED,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda_sm90,
)

def run_fake_check(fn):
    def wrapper(*args, **kwargs):
        fake_check(fn, args, kwargs)
    return wrapper


@pytest.mark.skipif(not SM80_ENABLED, reason="SM80 not enabled")
class TestSM80:
    def get_kernel(self, pv_accum_dtype):
        return sageattn_qk_int8_pv_fp16_cuda

    @pytest.mark.parametrize("is_causal", (False, True))
    @pytest.mark.parametrize("seq_len", (64, 128))
    @pytest.mark.parametrize("head", (32,))
    @pytest.mark.parametrize("batch", (4,))
    @pytest.mark.parametrize("headdim", (32, 64))
    @pytest.mark.parametrize("quant_gran", ("per_warp", "per_thread"))
    @pytest.mark.parametrize("pv_accum_dtype", ("fp16", "fp16+fp32", "fp32"))
    @pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
    @pytest.mark.parametrize("smooth_k", (False, True))
    @pytest.mark.parametrize("smooth_v", (False, True))
    @pytest.mark.parametrize("return_lse", (False, True))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_SM80(self, is_causal, seq_len, head, batch, headdim, quant_gran, pv_accum_dtype, tensor_layout, smooth_k, smooth_v, return_lse, dtype):
        q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")
        k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")

        v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
        sm_scale = 1 / (headdim ** 0.5)

        kernel = self.get_kernel(pv_accum_dtype)
        run_fake_check(kernel)(q, k, v, tensor_layout, is_causal, quant_gran,
                               sm_scale, pv_accum_dtype, smooth_k, smooth_v,
                               return_lse)


@pytest.mark.skipif(not SM89_ENABLED, reason="SM89 not enabled")
class TestSM89:

    def get_kernel(self):
        return sageattn_qk_int8_pv_fp8_cuda

    @pytest.mark.parametrize("is_causal", (False, True))
    @pytest.mark.parametrize("seq_len", (64, 128))
    @pytest.mark.parametrize("head", (32,))
    @pytest.mark.parametrize("batch", (4,))
    @pytest.mark.parametrize("headdim", (32, 64))
    @pytest.mark.parametrize("quant_gran", ("per_warp", "per_thread"))
    @pytest.mark.parametrize("pv_accum_dtype", ("fp32+fp32", "fp32+fp16", "fp32"))
    @pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
    @pytest.mark.parametrize("smooth_k", (False, True))
    @pytest.mark.parametrize("smooth_v", (False, True))
    @pytest.mark.parametrize("return_lse", (False, True))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_kernels(self, is_causal, seq_len, head, batch, headdim, quant_gran, pv_accum_dtype, tensor_layout, smooth_k, smooth_v, return_lse, dtype):
        kernel = self.get_kernel()


        if tensor_layout == "HND":
            q = torch.randint(-128, 127, (batch, head, seq_len, headdim), dtype=dtype, device="cuda")
            k = torch.randint(-128, 127, (batch, head, seq_len, headdim), dtype=dtype, device="cuda")
            v = torch.randn(batch, head, seq_len, headdim, dtype=dtype, device="cuda")
        else:  # NHD
            q = torch.randint(-128, 127, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")
            k = torch.randint(-128, 127, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")
            v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")

        sm_scale = 1.0 / (headdim ** 0.5)

        run_fake_check(kernel)(q, k, v, tensor_layout, is_causal, quant_gran,
                               sm_scale, pv_accum_dtype, smooth_k, smooth_v,
                               return_lse)


@pytest.mark.skipif(not SM90_ENABLED, reason="SM90 not enabled")
class TestSM90:
    def get_kernel(self):
        return sageattn_qk_int8_pv_fp8_cuda_sm90

    @pytest.mark.parametrize("is_causal", (False, True))
    @pytest.mark.parametrize("seq_len", (64, 128))
    @pytest.mark.parametrize("head", (32,))
    @pytest.mark.parametrize("batch", (4,))
    @pytest.mark.parametrize("headdim", (32, 64))
    @pytest.mark.parametrize("quant_gran", ("per_warp", "per_thread"))
    @pytest.mark.parametrize("pv_accum_dtype", ("fp32+fp32",))
    @pytest.mark.parametrize("tensor_layout", ("NHD", "HND"))
    @pytest.mark.parametrize("smooth_k", (False, True))
    @pytest.mark.parametrize("return_lse", (False, True))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_kernels(self, is_causal, seq_len, head, batch, headdim, quant_gran, pv_accum_dtype, tensor_layout, smooth_k, return_lse, dtype):
        kernel = self.get_kernel()

        q = torch.randint(-128, 127, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")
        k = torch.randint(-128, 127, (batch, seq_len, head, headdim), dtype=dtype, device="cuda")

        if tensor_layout == "HND":
            v = torch.randn(batch, head, seq_len, headdim, dtype=dtype, device="cuda")
        else:  # NHD
            v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")

        sm_scale = 1.0 / (headdim ** 0.5)

        run_fake_check(kernel)(q, k, v, tensor_layout, is_causal, quant_gran,
                               sm_scale, pv_accum_dtype, smooth_k, return_lse)
