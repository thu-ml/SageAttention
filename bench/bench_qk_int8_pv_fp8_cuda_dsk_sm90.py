import torch
from flash_attn.utils.benchmark import benchmark_forward
import sageattention._qattn_sm90 as qattn
import argparse

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP8 SM90')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
# parser.add_argument('--head_dim_qk', type=int, default=192, help='Head dimension')
# parser.add_argument('--head_dim_v', type=int, default=128, help='Head dimension')
parser.add_argument('--pv_accum_dtype', type=str, default='fp32+fp32', choices=['fp32', 'fp32+fp32'])
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
# head_dim_qk = args.head_dim_qk
# head_dim_v = args.head_dim_v
head_dim_qk = 192
head_dim_v = 128

assert args.pv_accum_dtype == 'fp32+fp32', "pure fp32 accumulator is not supported for now"

print(f"CUDA QK Int8 PV FP8 SM90 Mixed HeadDim=192 Benchmark")
print(f"batch: {batch}, head: {head}, head_dim_qk: {head_dim_qk}, head_dim_v: {head_dim_v}")

WARP_Q = 32
WARP_K = 64

kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf_dsk_sm90

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

is_causal = False
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 2 * head * batch * (head_dim_qk + head_dim_v) * seq_len * seq_len / (2 if is_causal else 1)

    q = torch.randint(-95, 95, (batch, head, seq_len, head_dim_qk), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, head, seq_len, head_dim_qk), dtype=torch.int8, device="cuda")
    q_nope, q_pe = torch.split(q, [128, 64], dim=-1)
    k_nope, k_pe = torch.split(k, [128, 64], dim=-1)
    o = torch.empty(batch, head, seq_len, head_dim_v, dtype=torch.float16, device="cuda")

    v_scale = torch.randn(batch, head, head_dim_v, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4 * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128 * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, head, head_dim_v, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (head_dim_qk ** 0.5)
    for i in range(5): kernel(q_nope, k_nope, q_pe, k_pe, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q_nope, k_nope, q_pe, k_pe, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')


is_causal = True
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 2 * head * batch * (head_dim_qk + head_dim_v) * seq_len * seq_len / (2 if is_causal else 1)

    q = torch.randint(-95, 95, (batch, head, seq_len, head_dim_qk), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, head, seq_len, head_dim_qk), dtype=torch.int8, device="cuda")
    q_nope, q_pe = torch.split(q, [128, 64], dim=-1)
    k_nope, k_pe = torch.split(k, [128, 64], dim=-1)
    o = torch.empty(batch, head, seq_len, head_dim_v, dtype=torch.float16, device="cuda")

    v_scale = torch.randn(batch, head, head_dim_v, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4 * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128 * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, head, head_dim_v, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (head_dim_qk ** 0.5)
    for i in range(5): kernel(q_nope, k_nope, q_pe, k_pe, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q_nope, k_nope, q_pe, k_pe, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')