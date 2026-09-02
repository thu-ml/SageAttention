# Blackwell MXFP8 D128 Attention Path

This branch adds a runnable MXFP8 D128 path to the SageAttention3 Blackwell
package. It is intended as a compact review baseline for the MXFP8 kernel work,
not as a collection of D64, video-generation, or tile-skip experiments.

## Scope

- Target GPU: Blackwell, tested on RTX 5090.
- Target head dimension: D128.
- Quantized Q/K/V format: MXFP8-style E4M3 values with UE8M0 block scales.
- Public API entry: `sageattn3_blackwell(...)` or
  `sageattn3_blackwell(..., quant_format="mxfp8_tc")`.
- This branch is scoped to the MXFP8 D128 path. Use the official SageAttention3
  Blackwell package for the original NVFP4 path.

The MXFP8 tensor-core path currently requires `head_dim == 128`.

## Build

From this directory:

```bash
module load cuda/12.8.1
python setup.py install
```

The build creates two CUDA extensions:

- `fp4attn_cuda`
- `fp4quant_cuda`

## Correctness Check

```bash
python check_mxfp8_small.py
```

This compares the MXFP8 path with PyTorch SDPA on small D128 inputs and prints
mean/max absolute errors.

## Benchmark

```bash
python bench_mxfp8_kernel_only.py --lengths 2048,4096,8192 --check
```

The benchmark reports:

- quantization time
- MXFP8 attention kernel time
- effective attention TOPS
- full Python API time
- optional SDPA-relative error

The TOPS number uses the standard dense attention work estimate:

```text
4 * B * H * L * L * D
```

for non-causal attention.

## Validation Snapshot

Validated on `gpu64` with an RTX 5090, CUDA 12.8.1, PyTorch 2.11.0+cu128.

Small D128 correctness against PyTorch SDPA:

```text
L=128  mxfp8_tc  mean_abs=0.00558  max_abs=0.04492  rel_l2=0.05069  cosine=0.99871
L=256  mxfp8_tc  mean_abs=0.00411  max_abs=0.04785  rel_l2=0.05088  cosine=0.99870
L=512  mxfp8_tc  mean_abs=0.00295  max_abs=0.02637  rel_l2=0.05212  cosine=0.99864
```

D128 non-causal benchmark:

```text
L=2048  quant_ms=0.0307  kernel_ms=0.1043  kernel_TOPS=329.3  api_TOPS=218.0
L=4096  quant_ms=0.0326  kernel_ms=0.3247  kernel_TOPS=423.3  api_TOPS=382.0
L=8192  quant_ms=0.0873  kernel_ms=1.0565  kernel_TOPS=520.4  api_TOPS=467.6
```

## Main Files

- `sageattn3/api.py`: Python API and MXFP8 quantization wrappers.
- `sageattn3/quantization/fp4_quantization_4d.cu`: FP4 and MXFP8 Q/K/V
  quantizers.
- `sageattn3/blackwell/api.cu`: CUDA extension entry points.
- `sageattn3/blackwell/mainloop_tma_ws.h`: online softmax and P-fragment
  packing path used by the MXFP8 tensor-core kernel.
- `sageattn3/blackwell/kernel_traits.h`: MXFP8 type/layout selection.
- `sageattn3/blackwell/blockscaled_layout.h`: block-scale layout mapping.
