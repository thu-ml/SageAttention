import torch

from ._ops import _attention_lse, fake_attention, fake_attention_with_v_scale, load_ops


_ops = load_ops("_qattn_sm89", "sageattention_sm89")
qk_int8_sv_f8_accum_f32_fuse_v_scale_attn = _ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.default
qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf = _ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.default
qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf = _ops.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.default
qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn = _ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.default

for _op in (
    _ops.qk_int8_sv_f8_accum_f32_attn.default,
    _ops.qk_int8_sv_f8_accum_f32_attn_inst_buf.default,
    _ops.qk_int8_sv_f8_accum_f16_attn_inst_buf.default,
):
    torch.library.register_fake(_op)(fake_attention)

for _op in (
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn,
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
    qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
):
    torch.library.register_fake(_op)(fake_attention_with_v_scale)


@torch.library.register_fake(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn)
def _fake_attention_with_v_scale_and_mean(
    query, key, value, output, query_scale, key_scale, value_scale, value_mean,
    tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
):
    return _attention_lse(query, tensor_layout, return_lse)
