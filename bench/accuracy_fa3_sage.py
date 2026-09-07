"""
Accuracy comparison:
  reference = FA3 (fp16/bf16)
  candidates = FA3-FP8, SageAttention

Reports max_abs_error / mean_abs_error / rmse / cosine / rel_l2.
"""

import argparse
import inspect

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Accuracy: FA3 vs FA3-FP8 vs SageAttention")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=32)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--seq_len", type=int, nargs="+", default=[1024, 2048, 4096])
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--methods",
    type=str,
    nargs="+",
    default=["fa3_fp8", "sage"],
    choices=["fa3_fp8", "sage"],
    help="Methods to compare against FA3 reference",
)
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
batch, head, headdim = args.batch_size, args.num_heads, args.head_dim
sm_scale = 1.0 / (headdim**0.5)

# ---- FA3 ----
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
except ModuleNotFoundError as e:
    import sys
    raise SystemExit(
        "flash_attn_interface not found in current Python.\n"
        f"  python = {sys.executable}\n"
        "FA3 was installed into conda env `vllm_0.25`. Activate it first:\n"
        "  conda activate vllm_0.25\n"
        "  cd SageAttention\n"
        "  python bench/accuracy_fa3_sage.py --batch_size 2 --num_heads 32 --head_dim 128 --dtype fp16\n"
    ) from e

_fa3_params = set(inspect.signature(flash_attn_func_v3).parameters.keys())


def _fa3_call(q, k, v, causal, *, q_scale=None, k_scale=None, v_scale=None):
    """q/k/v layout: NHD [B, S, H, D]. Supports both old/new descale arg names."""
    kwargs = {}
    if "causal" in _fa3_params:
        kwargs["causal"] = causal
    if "softmax_scale" in _fa3_params:
        kwargs["softmax_scale"] = sm_scale

    # new hopper API
    if q_scale is not None and "q_descale" in _fa3_params:
        kwargs["q_descale"] = q_scale
        kwargs["k_descale"] = k_scale
        kwargs["v_descale"] = v_scale
    # old API used by sageattention.fa3_wrapper
    elif q_scale is not None and "descale_q" in _fa3_params:
        kwargs["descale_q"] = q_scale
        kwargs["descale_k"] = k_scale
        kwargs["descale_v"] = v_scale

    out = flash_attn_func_v3(q, k, v, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def run_fa3_ref(q_nhd, k_nhd, v_nhd, causal):
    return _fa3_call(q_nhd, k_nhd, v_nhd, causal)


def run_fa3_fp8(q_nhd, k_nhd, v_nhd, causal):
    # Current FA3 Hopper API requires descale shape [batch, nheads_kv].
    # Use per-(batch, head) scales over seq/dim, then broadcast for quantization.
    # q/k/v layout: NHD [B, S, H, D]
    q_scale = (q_nhd.abs().amax(dim=(1, 3)).to(torch.float32) / 448.0).contiguous()  # [B, H]
    k_scale = (k_nhd.abs().amax(dim=(1, 3)).to(torch.float32) / 448.0).contiguous()
    v_scale = (v_nhd.abs().amax(dim=(1, 3)).to(torch.float32) / 448.0).contiguous()
    q_f8 = (q_nhd / q_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    k_f8 = (k_nhd / k_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    v_f8 = (v_nhd / v_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    out = _fa3_call(q_f8, k_f8, v_f8, causal, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
    return out.to(dtype)


# ---- SageAttention ----
_sage_fn = None
if "sage" in args.methods:
    try:
        from sageattention import sageattn as _sage_fn
    except Exception as e:
        print(f"[warn] sageattention unavailable: {e}")
        args.methods = [m for m in args.methods if m != "sage"]


def run_sage(q_hnd, k_hnd, v_hnd, causal):
    return _sage_fn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=causal, sm_scale=sm_scale)


def err_metrics(ref, pred):
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (pred_f - ref_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rmse = torch.sqrt((diff**2).mean()).item()
    cos = F.cosine_similarity(ref_f.flatten(), pred_f.flatten(), dim=0).item()
    rel_l2 = (torch.linalg.vector_norm(pred_f - ref_f) / (torch.linalg.vector_norm(ref_f) + 1e-12)).item()
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine": cos,
        "rel_l2": rel_l2,
    }


def fmt(m):
    return (
        f"max_abs={m['max_abs']:.6e}  mean_abs={m['mean_abs']:.6e}  "
        f"rmse={m['rmse']:.6e}  cosine={m['cosine']:.8f}  rel_l2={m['rel_l2']:.6e}"
    )


print("Accuracy Benchmark")
print(f"batch={batch}, head={head}, headdim={headdim}, dtype={args.dtype}")
print(f"reference=FA3({args.dtype}), methods={args.methods}")
print(f"FA3 params contain descale: "
      f"q_descale={'q_descale' in _fa3_params}, descale_q={'descale_q' in _fa3_params}")

for causal in [False, True]:
    print(f"\n=== is_causal: {causal} ===")
    for seq_len in args.seq_len:
        torch.manual_seed(args.seed + seq_len + int(causal))

        # NHD for FA3: [B, S, H, D]
        q_nhd = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
        k_nhd = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
        v_nhd = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
        # HND for Sage: [B, H, S, D]
        q_hnd = q_nhd.transpose(1, 2).contiguous()
        k_hnd = k_nhd.transpose(1, 2).contiguous()
        v_hnd = v_nhd.transpose(1, 2).contiguous()

        with torch.no_grad():
            ref = run_fa3_ref(q_nhd, k_nhd, v_nhd, causal)

            print(f"\nseq_len={seq_len}")
            print(f"  FA3({args.dtype})  [reference]")

            if "fa3_fp8" in args.methods:
                try:
                    out_fp8 = run_fa3_fp8(q_nhd, k_nhd, v_nhd, causal)
                    print(f"  FA3-FP8      {fmt(err_metrics(ref, out_fp8))}")
                except Exception as e:
                    print(f"  FA3-FP8      FAILED: {e}")

            if "sage" in args.methods and _sage_fn is not None:
                try:
                    out_sage = run_sage(q_hnd, k_hnd, v_hnd, causal)
                    # sage returns HND; compare in NHD against FA3
                    out_sage_nhd = out_sage.transpose(1, 2).contiguous()
                    print(f"  SageAttn     {fmt(err_metrics(ref, out_sage_nhd))}")
                except Exception as e:
                    print(f"  SageAttn     FAILED: {e}")
