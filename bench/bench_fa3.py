import torch
import inspect
try:
    from flash_attn.utils.benchmark import benchmark_forward
except ModuleNotFoundError:
    import torch.utils.benchmark as torch_benchmark

    def benchmark_forward(
        fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
    ):
        if verbose:
            print(desc, "- Forward pass")

        def amp_wrapper(*inputs, **kwinputs):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                fn(*inputs, **kwinputs)

        t = torch_benchmark.Timer(
            stmt="fn_amp(*inputs, **kwinputs)",
            globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
            num_threads=torch.get_num_threads(),
        )
        m = t.timeit(repeats)
        if verbose:
            print(m)
        return t, m

from flash_attn_interface import flash_attn_func as flash_attn_func_v3

import argparse

parser = argparse.ArgumentParser(description='Benchmark FlashAttention3')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'], help='Input dtype')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16

_fa3_param_names = set(inspect.signature(flash_attn_func_v3).parameters.keys())


def run_fa3(q, k, v, is_causal, headdim):
    kwargs = {}
    if "causal" in _fa3_param_names:
        kwargs["causal"] = is_causal
    if "softmax_scale" in _fa3_param_names:
        kwargs["softmax_scale"] = 1 / headdim**0.5
    out = flash_attn_func_v3(q, k, v, **kwargs)
    return out[0] if isinstance(out, tuple) else out

print(f"FlashAttention3 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, dtype: {args.dtype}")

is_causal = False
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    k = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    for i in range(5):
        run_fa3(q, k, v, is_causal, headdim)
    torch.cuda.synchronize()
    _, time = benchmark_forward(
        run_fa3, q, k, v, is_causal, headdim, repeats=100, verbose=False, desc="Triton"
    )
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')

is_causal = True
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    k = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
    for i in range(5):
        run_fa3(q, k, v, is_causal, headdim)
    torch.cuda.synchronize()
    _, time = benchmark_forward(
        run_fa3, q, k, v, is_causal, headdim, repeats=100, verbose=False, desc="Triton"
    )
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')
