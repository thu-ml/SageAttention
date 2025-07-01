import torch
from flash_attn.utils.benchmark import benchmark_forward

import sageattention._qattn_sm89 as qattn

import argparse

import subprocess
import re
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as e:
        print("Failed to get CUDA version:", e)
    return None, None

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP8')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp32+fp16', choices=['fp32', 'fp32+fp32', 'fp32+fp16'])
parser.add_argument('--fused_v', action='store_true', help='Enable fused_v kernel to test pv_accumulator accumulator')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
fused_v = args.fused_v

print(f"CUDA QK Int8 PV FP8 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {args.pv_accum_dtype}, fused_v: {fused_v}")

WARP_Q = 32
WARP_K = 64

#fused_v = False

if fused_v and args.pv_accum_dtype == "fp32":
    raise SystemExit("Error: fused_v must be used with pv_accum_dtype of fp32+fp32 or fp32+fp16")

cuda_major_version, cuda_minor_version = get_cuda_version()
if(cuda_major_version, cuda_minor_version) < (12, 8) and args.pv_accum_dtype == 'fp32+fp16':
    print("=============\n NOTE: cuda version < 12.8, not support pv_accum_dtype fp32+fp16. \n Swith to 'fp32+fp32' automatically\n=============")
    args.pv_accum_dtype = 'fp32+fp32'


if not fused_v:
    if args.pv_accum_dtype == 'fp32':
        kernel = qattn.qk_int8_sv_f8_accum_f32_attn # the kernel with fp32 (actually fp22) accumulator
    elif args.pv_accum_dtype == 'fp32+fp32':
        kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf # the kernel with fp32 longterm buffer and fp32 (actually fp22) shortterm accumulator
    elif args.pv_accum_dtype == 'fp32+fp16':
        kernel = qattn.qk_int8_sv_f8_accum_f16_attn_inst_buf # the kernel with fp32 longterm buffer and fp16 shortterm accumulator
else:
    if args.pv_accum_dtype == 'fp32+fp32':
        kernel = qattn.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    elif args.pv_accum_dtype == 'fp32+fp16':
        kernel = qattn.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

is_causal = False
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

    q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")

    vm = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")
    v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, headdim,head,  seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (headdim ** 0.5)
    if not fused_v:
        for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
        torch.cuda.synchronize()
        _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')
    else:
        for i in range(5): kernel(q, k, v, o, q_scale, k_scale, v_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
        torch.cuda.synchronize()
        _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, v_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')

is_causal = True
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

    q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
    o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")

    vm = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")
    v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, headdim,head,  seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (headdim ** 0.5)
    if not fused_v:
        for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
        torch.cuda.synchronize()
        _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')
    else:
        for i in range(5): kernel(q, k, v, o, q_scale, k_scale, v_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
        torch.cuda.synchronize()
        _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, v_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')