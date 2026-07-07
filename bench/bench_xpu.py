import argparse
import math
import time

import torch

from sageattention import sageattn

parser = argparse.ArgumentParser(description="Benchmark QK Int8 PV FP16 on XPU")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--num_heads", type=int, default=96, help="Number of heads")
parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads")
parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
parser.add_argument("--seq_len_kv", type=int, default=8192, help="KV sequence length")
args = parser.parse_args()

batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim
num_kv_heads = args.num_kv_heads
seq_len = args.seq_len
seq_len_kv = args.seq_len_kv

if not torch.xpu.is_available():
    print("XPU not available, skipping benchmark.")
    exit(0)

print(f"XPU SageAttention Benchmark")
print(
    f"batch_size: {batch_size}, num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}"
)
print(f"seq_len: {seq_len}, seq_len_kv: {seq_len_kv}")

q = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="xpu"
)
k = torch.randn(
    batch_size, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float16, device="xpu"
)
v = torch.randn(
    batch_size, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float16, device="xpu"
)
scale = 1.0 / math.sqrt(head_dim)

# Warmup
for _ in range(5):
    sageattn(q, k, v, is_causal=True, sm_scale=scale)
torch.xpu.synchronize()

# Benchmark
N = 30
st = time.time()
for _ in range(N):
    sageattn(q, k, v, is_causal=True, sm_scale=scale)
torch.xpu.synchronize()
t = (time.time() - st) / N * 1000

print(f"Latency: {t:.3f} ms")
