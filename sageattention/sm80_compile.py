from . import _qattn_sm80
import torch


@torch.library.custom_op("sageattention::qk_int8_sv_f16_accum_f16_attn", mutates_args=(), device_types="cuda")
def qk_int8_sv_f16_accum_f16_attn(
    q_int8: torch.Tensor, 
    k_int8: torch.Tensor, 
    v: torch.Tensor, 
    o: torch.Tensor, 
    q_scale: torch.Tensor, 
    k_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    """
    Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP16 accumulation.
    """
    return _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse)


@torch.library.custom_op("sageattention::qk_int8_sv_f16_accum_f32_attn", mutates_args=(), device_types="cuda")
def qk_int8_sv_f16_accum_f32_attn(
    q_int8: torch.Tensor, 
    k_int8: torch.Tensor, 
    v: torch.Tensor, 
    o: torch.Tensor, 
    q_scale: torch.Tensor, 
    k_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    """
    Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP32 accumulation.
    """
    return _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse)


@torch.library.custom_op("sageattention::qk_int8_sv_f16_accum_f16_attn_inst_buf", mutates_args=(), device_types="cuda")
def qk_int8_sv_f16_accum_f16_attn_inst_buf(
    q_int8: torch.Tensor, 
    k_int8: torch.Tensor, 
    v: torch.Tensor, 
    o: torch.Tensor, 
    q_scale: torch.Tensor, 
    k_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    """
    Custom CUDA kernel for SageAttention with INT8 quantization for Q and K, FP16 PV with FP16 accumulation.
    """
    return _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse)


def qk_int8_sv_f16_accum_attn_fake(
    q_int8: torch.Tensor, 
    k_int8: torch.Tensor, 
    v: torch.Tensor, 
    o: torch.Tensor, 
    q_scale: torch.Tensor, 
    k_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:

    batch_size = q_int8.size(0)
    num_qo_heads = q_int8.size(2)

    if tensor_layout == 0:
        qo_len = q_int8.size(1)
    else:
        qo_len = q_int8.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device="cuda")
    else:
        lse = torch.empty((0))
    return lse


torch.library.register_fake("sageattention::qk_int8_sv_f16_accum_f16_attn")(qk_int8_sv_f16_accum_attn_fake)
torch.library.register_fake("sageattention::qk_int8_sv_f16_accum_f32_attn")(qk_int8_sv_f16_accum_attn_fake)
torch.library.register_fake("sageattention::qk_int8_sv_f16_accum_f16_attn_inst_buf")(qk_int8_sv_f16_accum_attn_fake)
