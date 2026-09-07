# Correctness test for sageattn_qk_int8_pv_fp8_cuda_seqlens (sm120 decode path).
#
# Verifies:
# 1. Padded K/V + device seq_lens matches per-request exact-length sageattn().
# 2. A captured CUDA graph replays correctly when seq_lens and K/V contents
#    change afterwards (the whole point of the kernel change).
import torch

from sageattention.core import sageattn, sageattn_qk_int8_pv_fp8_cuda_seqlens

torch.manual_seed(0)

DTYPE = torch.bfloat16
B, H, HKV, D = 3, 8, 2, 128
S_CAP = 512  # fixed padded capacity per request


def make_case(seqs, scale=1.0):
    q = torch.randn(B, H, 1, D, dtype=DTYPE, device="cuda") * scale
    k = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    v = torch.randn(B, HKV, S_CAP, D, dtype=DTYPE, device="cuda") * scale
    for i, L in enumerate(seqs):
        k[i, :, L:] = 0
        v[i, :, L:] = 0
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    return q, k, v, seq_lens


def ref_per_request(q, k, v, seqs):
    outs = []
    for i, L in enumerate(seqs):
        r = sageattn(
            q[i : i + 1],
            k[i : i + 1, :, :L].contiguous(),
            v[i : i + 1, :, :L].contiguous(),
            tensor_layout="HND",
            is_causal=False,
        )
        outs.append(r)
    return torch.cat(outs, dim=0)


def check(tag, out, ref):
    err = (out.float() - ref.float()).abs()
    rel = err.max().item() / ref.float().abs().max().item()
    print(f"[{tag}] max_abs_err={err.max().item():.6f} rel={rel:.6f} "
          f"ref_absmax={ref.float().abs().max().item():.4f}")
    assert rel < 2e-2, f"{tag} mismatch"
    assert not torch.isnan(out.float()).any(), f"{tag} has NaN"


def main():
    # ---- 1. eager correctness across length patterns (block boundary cases) ----
    for seqs in ([100, 300, 512], [64, 63, 65], [1, 1, 511], [128, 129, 64]):
        q, k, v, seq_lens = make_case(seqs)
        out = sageattn_qk_int8_pv_fp8_cuda_seqlens(q, k, v, seq_lens)
        ref = ref_per_request(q, k, v, seqs)
        check(f"eager seqs={seqs}", out, ref)

    # ---- 2. CUDA graph capture + replay with changed lengths/data ----
    seqs0 = [100, 300, 512]
    q, k, v, seq_lens = make_case(seqs0)
    # static graph input buffers
    gq, gk, gv = q.clone(), k.clone(), v.clone()
    gl = seq_lens.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sageattn_qk_int8_pv_fp8_cuda_seqlens(gq, gk, gv, gl)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        go = sageattn_qk_int8_pv_fp8_cuda_seqlens(gq, gk, gv, gl)

    # replay #1: same data
    graph.replay()
    check("graph replay (same)", go, ref_per_request(q, k, v, seqs0))

    # replay #2: new lengths + new data copied into the static buffers
    seqs1 = [200, 33, 511]
    q1, k1, v1, l1 = make_case(seqs1, scale=0.5)
    gq.copy_(q1)
    gk.copy_(k1)
    gv.copy_(v1)
    gl.copy_(l1)
    graph.replay()
    check("graph replay (new lens/data)", go, ref_per_request(q1, k1, v1, seqs1))

    # replay #3: grow past capture-time lengths
    seqs2 = [512, 500, 128]
    q2, k2, v2, l2 = make_case(seqs2)
    gq.copy_(q2)
    gk.copy_(k2)
    gv.copy_(v2)
    gl.copy_(l2)
    graph.replay()
    check("graph replay (grown lens)", go, ref_per_request(q2, k2, v2, seqs2))

    print("ALL PASSED")


if __name__ == "__main__":
    main()
