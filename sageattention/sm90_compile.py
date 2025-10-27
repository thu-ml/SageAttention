from . import _qattn_sm90
import torch


@torch.library.custom_op("sageattention_sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf", mutates_args=("output",), device_types="cuda")
def qk_int8_sv_f8_accum_f32_attn_inst_buf(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _qattn_sm90.qk_int8_sv_f8_accum_f32_attn_inst_buf(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )


@torch.library.register_fake("sageattention_sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf")
def qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0))
    return lse


@torch.library.custom_op("sageattention_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", mutates_args=("output",), device_types="cuda")
def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    value_scale: torch.Tensor,
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        query, key, value, output, query_scale, key_scale, value_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )


@torch.library.register_fake("sageattention_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")
def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    value_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )
