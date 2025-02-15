from torch.nn.functional import scaled_dot_product_attention as sdpa
import torch
from flash_attn.utils.benchmark import benchmark_forward

import argparse

parser = argparse.ArgumentParser(description='Benchmark Baseline')
parser.add_argument('--method', type=str, default='fa2', choices=['fa2', 'torch', 'xformers'])
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

assert args.method in ['fa2', 'torch', 'xformers']

# only one of the following is True
torch.backends.cuda.enable_flash_sdp(args.method == 'fa2')   # use FA2
torch.backends.cuda.enable_math_sdp(args.method == 'torch')  # use Torch
torch.backends.cuda.enable_mem_efficient_sdp(args.method == 'xformers')  # use xformers

print(f"Baseline: {args.method}")
print(f"batch: {batch}, head: {head}, headdim: {headdim}")

is_causal = False
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    for i in range(5): sdpa(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')

is_causal = True
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    for i in range(5): sdpa(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')