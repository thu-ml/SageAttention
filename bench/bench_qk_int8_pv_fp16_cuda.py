import torch
from flash_attn.utils.benchmark import benchmark_forward

import sageattention._qattn_sm80 as qattn

import argparse

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp16', choices=['fp16', 'fp16+fp32', 'fp32'])
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

print(f"CUDA QK Int8 PV FP16 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {args.pv_accum_dtype}")

WARP_Q = 16 if (headdim == 128 and args.pv_accum_dtype == "fp16+fp32") else 32
WARP_K = 64

if args.pv_accum_dtype == 'fp32':
    kernel = qattn.qk_int8_sv_f16_accum_f32_attn # the kernel with fully fp32 accumulator
elif args.pv_accum_dtype == 'fp16+fp32':
    kernel = qattn.qk_int8_sv_f16_accum_f16_attn_inst_buf # the kernel with fp32 longterm buffer and fp16 shortterm accumulator
elif args.pv_accum_dtype == 'fp16':
    kernel = qattn.qk_int8_sv_f16_accum_f16_attn # the kernel with fully fp16 accumulator

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

is_causal = False
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
    
    q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")

    vm = torch.randn(batch, head, headdim, dtype=torch.float16, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")
   
    v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
    o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')

is_causal = True
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
    
    q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")

    vm = torch.randn(batch, head, headdim, dtype=torch.float16, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")
   
    v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
    o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')