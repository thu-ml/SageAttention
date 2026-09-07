# Correctness: dense seqlens vs paged-quant + paged-attn (no gather/scatter of KV).
#
# Usage:
#   PYTHONPATH=/path/to/SageAttention python test_paged_k_decode.py
import torch

from sageattention.core import (
    sageattn_qk_int8_pv_fp8_cuda_seqlens,
    sageattn_qk_int8_pv_fp8_cuda_seqlens_paged,
)

torch.manual_seed(0)

DTYPE = torch.bfloat16
B, H, HKV, D = 3, 8, 2, 128
S_CAP = 512
BLOCK = 16


def make_case(seqs, scale=1.0):
    q = torch.randn(B, H, 1, D, dtype=DTYPE, device="cuda") * scale
    k = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    v = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    for i, L in enumerate(seqs):
        k[i, :, L:] = 0
        v[i, :, L:] = 0
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    max_pages = (S_CAP + BLOCK - 1) // BLOCK
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device="cuda")
    for b in range(B):
        for p in range(max_pages):
            block_table[b, p] = b * max_pages + p
    num_blocks = int(block_table.max().item()) + 1
    k_cache = torch.zeros(
        (num_blocks, BLOCK, HKV, D), dtype=DTYPE, device="cuda"
    )
    v_cache = torch.zeros(
        (num_blocks, BLOCK, HKV, D), dtype=DTYPE, device="cuda"
    )
    for b in range(B):
        sl = int(seq_lens[b].item())
        for t in range(sl):
            page = int(block_table[b, t // BLOCK].item())
            off = t % BLOCK
            k_cache[page, off].copy_(k[b, :, t, :])
            v_cache[page, off].copy_(v[b, :, t, :])
    return q, k, v, k_cache, v_cache, seq_lens, block_table


def check(tag, out, ref):
    err = (out.float() - ref.float()).abs()
    rel = err.max().item() / max(ref.float().abs().max().item(), 1e-6)
    print(
        f"[{tag}] max_abs_err={err.max().item():.6f} rel={rel:.6f} "
        f"ref_absmax={ref.float().abs().max().item():.4f}"
    )
    assert rel < 2e-2, f"{tag} mismatch"
    assert not torch.isnan(out.float()).any(), f"{tag} has NaN"


def make_prefill_case(seqs, scale=1.0):
    """Full-prefill: qo_len == seq_len per request (causal)."""
    max_q = max(seqs)
    q = torch.randn(B, H, max_q, D, dtype=DTYPE, device="cuda") * scale
    k = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    v = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    for i, L in enumerate(seqs):
        q[i, :, L:] = 0
        k[i, :, L:] = 0
        v[i, :, L:] = 0
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    max_pages = (S_CAP + BLOCK - 1) // BLOCK
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device="cuda")
    for b in range(B):
        for p in range(max_pages):
            block_table[b, p] = b * max_pages + p
    num_blocks = int(block_table.max().item()) + 1
    k_cache = torch.zeros(
        (num_blocks, BLOCK, HKV, D), dtype=DTYPE, device="cuda"
    )
    v_cache = torch.zeros(
        (num_blocks, BLOCK, HKV, D), dtype=DTYPE, device="cuda"
    )
    for b in range(B):
        sl = int(seq_lens[b].item())
        for t in range(sl):
            page = int(block_table[b, t // BLOCK].item())
            off = t % BLOCK
            k_cache[page, off].copy_(k[b, :, t, :])
            v_cache[page, off].copy_(v[b, :, t, :])
    return q, k, v, k_cache, v_cache, seq_lens, block_table


def main():
    for seqs in ([100, 300, 512], [64, 63, 65], [1, 1, 511], [128, 129, 64]):
        q, k, v, k_cache, v_cache, seq_lens, block_table = make_case(seqs)
        ref = sageattn_qk_int8_pv_fp8_cuda_seqlens(q, k, v, seq_lens)
        out = sageattn_qk_int8_pv_fp8_cuda_seqlens_paged(
            q, k_cache, v_cache, seq_lens, block_table, BLOCK, S_CAP
        )
        check(f"eager decode seqs={seqs}", out, ref)

    # Prefill: qo_len>1 + causal, compare per-request to dense sageattn.
    from sageattention import sageattn

    for seqs in ([128, 64, 256], [32, 33, 31]):
        q, k, v, k_cache, v_cache, seq_lens, block_table = make_prefill_case(
            seqs
        )
        outs = []
        refs = []
        for i, L in enumerate(seqs):
            qi = q[i : i + 1, :, :L, :].contiguous()
            ki = k[i : i + 1, :, :L, :].contiguous()
            vi = v[i : i + 1, :, :L, :].contiguous()
            ref_i = sageattn(
                qi.permute(0, 2, 1, 3),
                ki.permute(0, 2, 1, 3),
                vi.permute(0, 2, 1, 3),
                tensor_layout="NHD",
                is_causal=True,
            ).permute(0, 2, 1, 3)
            out_i = sageattn_qk_int8_pv_fp8_cuda_seqlens_paged(
                qi,
                k_cache,
                v_cache,
                seq_lens[i : i + 1],
                block_table[i : i + 1],
                BLOCK,
                S_CAP,
                is_causal=True,
            )
            refs.append(ref_i)
            outs.append(out_i)
            check(f"prefill causal L={L}", out_i, ref_i)
        print(f"[prefill seqs={seqs}] ALL reqs ok")

    # CUDA Graph: paged-quant outside capture; only paged attn inside (fixed buffers).
    from sageattention import sm89_compile
    from sageattention.quant import (
        mean_k_paged,
        per_warp_int8_q,
        quant_k_paged_int8,
        quant_v_paged_fp8,
    )

    seqs0 = [100, 300, 512]
    q, k, v, k_cache, v_cache, seq_lens, block_table = make_case(seqs0)
    sm = D**-0.5

    def quant_all(q_in, kc, vc, sl, bt):
        km = mean_k_paged(kc, bt, sl, BLOCK).to(dtype=kc.dtype).float()
        q8, qs = per_warp_int8_q(q_in, BLKQ=128, WARPQ=32, tensor_layout="HND")
        k8, ks = quant_k_paged_int8(kc, bt, sl, BLOCK, S_CAP, km=km, BLKK=64)
        v8, vs = quant_v_paged_fp8(vc, bt, sl, BLOCK, S_CAP, scale_max=2.25)
        return q8, qs, k8, ks, v8, vs

    q8, qs, k8, ks, v8, vs = quant_all(q, k_cache, v_cache, seq_lens, block_table)
    o = torch.empty(q.shape, dtype=q.dtype, device="cuda")
    gl = seq_lens.clone()
    gbt = block_table.clone()
    gq8, gqs, gk8, gks, gv8, gvs = (
        q8.clone(), qs.clone(), k8.clone(), ks.clone(), v8.clone(), vs.clone()
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_seqlens_paged(
                gq8, gk8, gv8.view(torch.int8), o, gqs, gks, gvs,
                gl, gbt, BLOCK, S_CAP, 0, 2, sm, 0,
            )
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_seqlens_paged(
            gq8, gk8, gv8.view(torch.int8), o, gqs, gks, gvs,
            gl, gbt, BLOCK, S_CAP, 0, 2, sm, 0,
        )

    graph.replay()
    check(
        "graph replay (same)",
        o,
        sageattn_qk_int8_pv_fp8_cuda_seqlens(q, k, v, seq_lens),
    )

    seqs1 = [200, 400, 256]
    q2, k2, v2, kc2, vc2, seq_lens2, bt2 = make_case(seqs1)
    q8_2, qs_2, k8_2, ks_2, v8_2, vs_2 = quant_all(
        q2, kc2, vc2, seq_lens2, bt2
    )
    gq8.copy_(q8_2)
    gqs.copy_(qs_2)
    gk8.copy_(k8_2)
    gks.copy_(ks_2)
    gv8.copy_(v8_2)
    gvs.copy_(vs_2)
    gl.copy_(seq_lens2)
    gbt.copy_(bt2)
    graph.replay()
    check(
        "graph replay (new lens)",
        o,
        sageattn_qk_int8_pv_fp8_cuda_seqlens(q2, k2, v2, seq_lens2),
    )

    print("ALL PASS")


if __name__ == "__main__":
    main()
