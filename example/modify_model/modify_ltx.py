from typing import Optional, Callable

import torch

from diffusers import LTXVideoTransformer3DModel
from diffusers.utils import is_torch_version
from diffusers.models.transformers.transformer_ltx import LTXAttention, apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn


class SageLTXVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0). This is used in the LTX
    model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    _attention_backend = None

    def __init__(self, attn_func: Callable):
        self.attn_func = attn_func
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if self.attn_func is dispatch_attention_fn:
            hidden_states = self.attn_func(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
        else:
            hidden_states = self.attn_func(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                tensor_layout="NHD",
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sage_attn_ltx(
    model: LTXVideoTransformer3DModel,
    attn_func: Callable,
):
    """
    Replace attn1 processor in every transformer block with SageLTXVideoAttnProcessor(attn_func).
    """
    for idx, block in enumerate(model.transformer_blocks):
        origin = block.attn1.get_processor()
        processor = SageLTXVideoAttnProcessor(attn_func)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin
