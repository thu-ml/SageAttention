import pytest
import torch
from torch.testing._internal.optests import fake_check

import sageattention._qattn_sm80 as qattn
from sageattention.core import (
    SM80_ENABLED,
    SM89_ENABLED,
    SM90_ENABLED,
    qk_int8_sv_f16_accum_f32_attn,
    qk_int8_sv_f16_accum_f16_attn_inst_buf,
    qk_int8_sv_f16_accum_f16_attn,
    # qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn,
    # qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
    # qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
)


def run_fake_check(fn):
    def wrapper(*args, **kwargs):
        fake_check(fn, args, kwargs)
    return wrapper


@pytest.mark.skipif(not SM80_ENABLED, reason="SM80 not enabled")
class Test_SM80:
    kernels = {
        "fp32": qk_int8_sv_f16_accum_f32_attn,
        "fp16+fp32": qk_int8_sv_f16_accum_f16_attn_inst_buf,
        "fp16": qk_int8_sv_f16_accum_f16_attn
    }

    def get_kernel(self, pv_accum_dtype):
        return self.kernels[pv_accum_dtype]

    @pytest.mark.parametrize("is_causal", (False, True))
    @pytest.mark.parametrize("seq_len", (1024, 2048,))
    @pytest.mark.parametrize("head", (32,))
    @pytest.mark.parametrize("batch", (4,))
    @pytest.mark.parametrize("headdim", (128,))
    @pytest.mark.parametrize("quant_gran", ("per_warp", "per_thread"))
    @pytest.mark.parametrize("pv_accum_dtype", ("fp16", "fp16+fp32", "fp32"))
    def test_qk_int8_sv_f16_accum_f16_attn(self, is_causal, seq_len, head, batch, headdim, quant_gran, pv_accum_dtype):
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        
        q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
        k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")

        vm = torch.randn(batch, head, headdim, dtype=torch.float16, device="cuda")

        WARP_Q = 16 if (headdim == 128 and pv_accum_dtype == "fp16+fp32") else 32
        WARP_K = 64

        if quant_gran == 'per_warp':
            q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
            k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
        else:
            q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
            k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")
    
        v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
        o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
        sm_scale = 1 / (headdim ** 0.5)
        _qk_quant_gran = 3 if quant_gran == 'per_thread' else 2

        kernel = self.get_kernel(pv_accum_dtype)
        run_fake_check(kernel)(q, k, v, o, q_scale, k_scale, 0, is_causal, _qk_quant_gran, sm_scale, 0)


# @pytest.mark.skipif(not SM89_ENABLED, reason="SM89 not enabled")
# class Test_SM89(_TestTorchCompileBase):
#     kernels = {
#         "fp32": qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn,
#         "fp16+fp32": qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
#         "fp16": qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
#     }


# @pytest.mark.skipif(not SM90_ENABLED, reason="SM90 not enabled")
# class Test_SM90(_TestTorchCompileBase):
#     kernels = {
#         "fp16+fp32": qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
#     }