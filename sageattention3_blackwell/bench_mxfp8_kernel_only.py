import argparse
import statistics

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

import fp4quant_cuda
from sageattn3 import sageattn3_blackwell
from sageattn3.api import blockscaled_mxfp8_attn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SageAttention3 MXFP8 tensor-core path on Blackwell."
    )
    parser.add_argument("--lengths", default="2048,4096,8192,16384")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help="Use cudaProfilerStart/Stop around one MXFP8 attention launch.",
    )
    return parser.parse_args()


def event_time_ms(fn, iters):
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summarize_ms(times):
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "mean": statistics.fmean(times),
    }


def effective_attention_tops(B, H, L, D, ms, causal):
    # Standard attention-equivalent FLOP count:
    # QK^T costs 2*B*H*L*L*D, P@V costs another 2*B*H*L*L*D.
    # Causal masks roughly halve the valid score matrix.
    ops = 4.0 * B * H * L * L * D
    if causal:
        ops *= 0.5
    return ops / (ms * 1e-3) / 1e12


def make_inputs(B, H, L, D):
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    q8 = torch.empty((B, H, L, D), device="cuda", dtype=torch.float8_e4m3fn)
    qs = torch.empty((B, H, L, D // 32), device="cuda", dtype=torch.uint8)
    k8 = torch.empty((B, H, L, D), device="cuda", dtype=torch.float8_e4m3fn)
    ks = torch.empty((B, H, L, D // 32), device="cuda", dtype=torch.uint8)
    v8 = torch.empty((B, H, D, L), device="cuda", dtype=torch.float8_e4m3fn)
    vs = torch.empty((B, H, D, L // 32), device="cuda", dtype=torch.uint8)
    delta_s = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
    return q, k, v, q8, qs, k8, ks, v8, vs, delta_s


def quantize(q, k, v, q8, qs, k8, ks, v8, vs):
    fp4quant_cuda.scaled_mxfp8_quant(q, q8, qs, 1)
    fp4quant_cuda.scaled_mxfp8_quant_permute(k, k8, ks, 1)
    fp4quant_cuda.scaled_mxfp8_quant_trans(v, v8, vs, 1)


def main():
    args = parse_args()
    assert args.D == 128, "mxfp8_tc path currently supports D=128"
    assert args.D % 32 == 0
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        f"{'L':>7} {'quant_ms':>10} {'kernel_ms':>10} {'kernel_TOPS':>12} "
        f"{'api_ms':>10} {'api_TOPS':>10} {'mean_err':>10} {'max_err':>10}"
    )

    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        tensors = make_inputs(args.B, args.H, L, args.D)
        q, k, v, q8, qs, k8, ks, v8, vs, delta_s = tensors
        kernel_fn = lambda: blockscaled_mxfp8_attn(
            (q8, qs), (k8, ks), (v8, vs), delta_s, L, args.causal, False, True
        )[0]
        api_fn = lambda: sageattn3_blackwell(
            q, k, v, is_causal=args.causal, quant_format="mxfp8_tc"
        )

        for _ in range(args.warmup):
            quantize(q, k, v, q8, qs, k8, ks, v8, vs)
            kernel_fn()
            api_fn()
        torch.cuda.synchronize()

        if args.profile_once:
            quantize(q, k, v, q8, qs, k8, ks, v8, vs)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            kernel_fn()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()

        quant_ms = summarize_ms(
            event_time_ms(lambda: quantize(q, k, v, q8, qs, k8, ks, v8, vs), args.iters)
        )["median"]
        quantize(q, k, v, q8, qs, k8, ks, v8, vs)
        kernel_ms = summarize_ms(event_time_ms(kernel_fn, args.iters))["median"]
        api_ms = summarize_ms(event_time_ms(api_fn, args.iters))["median"]

        mean_err = float("nan")
        max_err = float("nan")
        if args.check:
            ref = sdpa(q, k, v, is_causal=args.causal).float()
            out = api_fn().float()
            diff = (out - ref).abs()
            mean_err = diff.mean().item()
            max_err = diff.max().item()

        print(
            f"{L:7d} {quant_ms:10.4f} {kernel_ms:10.4f} "
            f"{effective_attention_tops(args.B, args.H, L, args.D, kernel_ms, args.causal):12.1f} "
            f"{api_ms:10.4f} "
            f"{effective_attention_tops(args.B, args.H, L, args.D, api_ms, args.causal):10.1f} "
            f"{mean_err:10.5f} {max_err:10.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
