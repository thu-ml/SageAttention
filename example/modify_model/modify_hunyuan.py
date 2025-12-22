# ================================================================
# SageAttention in HunyuanVideo
# ================================================================
# NOTE: This file is kept for reference and is currently DISABLED.
#
# Diffusers HunyuanVideo pipeline commonly uses `attention_mask`, which is not supported by `sageattn`.
# So a naive injection here will not work reliably (falls back to SDPA, or gives wrong results).
#
# What to do instead:
# (1) Official SageAttention implementation:
#     https://huggingface.co/tencent/HunyuanVideo-1.5
#     -> use CLI flags: `--use_sageattn` and `--sage_blocks_range`
#
# (2) Workaround / selective SageAttention (mask-free image tokens):
#     https://github.com/thu-ml/SageAttention/issues/115
#     -> split text vs image tokens; use Sage only on large image-token self-attention.
# ================================================================


# from typing import Optional, Callable
# import torch
# import torch.nn.functional as F
# from diffusers import HunyuanVideoTransformer3DModel
# from diffusers.models.attention_processor import Attention
# from sageattention import sageattn

# class SageHunyuanVideoAttnProcessor2_0:
#     def __init__(self, attn_func: Callable):
#         self.attn_func = attn_func
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError(
#                 "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
#             )

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if attn.add_q_proj is None and encoder_hidden_states is not None:
#             hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

#         # 1. QKV projections
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)

#         query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
#         key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
#         value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

#         # 2. QK normalization
#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # 3. Rotational positional embeddings applied to latent stream
#         if image_rotary_emb is not None:
#             from diffusers.models.embeddings import apply_rotary_emb
            
#             if attn.add_q_proj is None and encoder_hidden_states is not None:
#                 query = torch.cat(
#                     [
#                         apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
#                         query[:, :, -encoder_hidden_states.shape[1] :],
#                     ],
#                     dim=2,
#                 )
#                 key = torch.cat(
#                     [
#                         apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
#                         key[:, :, -encoder_hidden_states.shape[1] :],
#                     ],
#                     dim=2,
#                 )
#             else:
#                 query = apply_rotary_emb(query, image_rotary_emb)
#                 key = apply_rotary_emb(key, image_rotary_emb)

#         # 4. Encoder condition QKV projection and normalization
#         if attn.add_q_proj is not None and encoder_hidden_states is not None:
#             encoder_query = attn.add_q_proj(encoder_hidden_states)
#             encoder_key = attn.add_k_proj(encoder_hidden_states)
#             encoder_value = attn.add_v_proj(encoder_hidden_states)

#             encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
#             encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
#             encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

#             if attn.norm_added_q is not None:
#                 encoder_query = attn.norm_added_q(encoder_query)
#             if attn.norm_added_k is not None:
#                 encoder_key = attn.norm_added_k(encoder_key)

#             query = torch.cat([query, encoder_query], dim=2)
#             key = torch.cat([key, encoder_key], dim=2)
#             value = torch.cat([value, encoder_value], dim=2)

#         # 5. Attention
#         if attention_mask is not None:
#             hidden_states = F.scaled_dot_product_attention(
#                 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             )
#         else:
#             if self.attn_func is sageattn: print("Using sageattn")
#             hidden_states = self.attn_func(
#                 query, key, value, dropout_p=0.0, is_causal=False
#             )
#         hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
#         hidden_states = hidden_states.to(query.dtype)

#         # 6. Output projection
#         if encoder_hidden_states is not None:
#             hidden_states, encoder_hidden_states = (
#                 hidden_states[:, : -encoder_hidden_states.shape[1]],
#                 hidden_states[:, -encoder_hidden_states.shape[1] :],
#             )

#             if getattr(attn, "to_out", None) is not None:
#                 hidden_states = attn.to_out[0](hidden_states)
#                 hidden_states = attn.to_out[1](hidden_states)

#             if getattr(attn, "to_add_out", None) is not None:
#                 encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

#         return hidden_states, encoder_hidden_states


# def set_sage_attn_hunyuan(
#     model: HunyuanVideoTransformer3DModel,
#     attn_func,
# ):
#     # Dual-stream blocks
#     for idx, block in enumerate(model.transformer_blocks):
#         processor = SageHunyuanVideoAttnProcessor2_0(attn_func=attn_func)
#         block.attn.set_processor(processor)
#     # Single-stream blocks
#     for idx, block in enumerate(model.single_transformer_blocks):
#         processor = SageHunyuanVideoAttnProcessor2_0(attn_func=attn_func)
#         block.attn.set_processor(processor)
