from pathlib import Path
import sys

import torch


def load_ops(library, namespace):
    suffix = ".pyd" if sys.platform == "win32" else ".abi3.so"
    path = Path(__file__).with_name(library + suffix)
    if not path.is_file():
        raise ImportError(f"Missing SageAttention CUDA library: {path}")
    torch.ops.load_library(str(path))
    return getattr(torch.ops, namespace)


def _attention_lse(query, tensor_layout, return_lse):
    if not return_lse:
        return torch.empty((0,))
    heads, length = query.shape[1:3]
    if tensor_layout == 0:
        heads, length = length, heads
    return query.new_empty((query.size(0), heads, length), dtype=torch.float32)


def fake_attention(
    query, key, value, output, query_scale, key_scale,
    tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
):
    return _attention_lse(query, tensor_layout, return_lse)


def fake_attention_with_v_scale(
    query, key, value, output, query_scale, key_scale, value_scale,
    tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
):
    return _attention_lse(query, tensor_layout, return_lse)
