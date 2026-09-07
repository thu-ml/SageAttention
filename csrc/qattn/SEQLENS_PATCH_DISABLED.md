# SageAttention seqlens / CUDA-graph patch — ENABLED

The sm89/sm120 device-`seq_lens` decode kernel is active again.

Source:
  `sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_seqlens.cu`

Python API:
  `sageattention.core.sageattn_qk_int8_pv_fp8_cuda_seqlens`

Verify with:
  `python test_seqlens_decode.py`

vLLM `SAGE_ATTN` uses this for pure-decode FULL cudagraphs
(padded KV + device `seq_lens`, FA2-style). Prefill still uses dense
`sageattn()` with exact lengths.
