import torch, sys
from torch.nn.functional import scaled_dot_product_attention as sdpa
from sageattn3 import sageattn3_blackwell

torch.manual_seed(0)
B, H, D = 1, 4, 128
print(f"{'L':>6} {'fmt':>10} {'mean_err':>10} {'max_err':>10} {'rel_l2':>10} {'cosine':>10} {'note'}")
for L in [128, 256, 512]:
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    ref = sdpa(q, k, v, is_causal=False).float()
    for fmt in ["mxfp8_ref", "mxfp8", "mxfp8_tc"]:
        try:
            y = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=False, quant_format=fmt).float()
            d = (y - ref).abs()
            rel_l2 = (y - ref).norm() / ref.norm()
            cosine = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0)
            nan = "  HAS_NAN" if torch.isnan(y).any() else ""
            zero = "  ALL_ZERO" if y.abs().max() == 0 else ""
            print(
                f"{L:>6} {fmt:>10} {d.mean().item():>10.5f} {d.max().item():>10.5f} "
                f"{rel_l2.item():>10.5f} {cosine.item():>10.5f}{nan}{zero}"
            )
        except Exception as e:
            print(f"{L:>6} {fmt:>10}   FAILED: {type(e).__name__}: {str(e)[:150]}")
        sys.stdout.flush()

# Sanity: first row of the first head, first 8 dims -- catches gross layout errors
# that a mean-error number can hide.
L = 128
q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
ref = sdpa(q, k, v, is_causal=False).float()
try:
    y = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=False, quant_format="mxfp8_tc").float()
    print("\nref[0,0,0,:8] =", [round(x, 4) for x in ref[0, 0, 0, :8].tolist()])
    print("tc [0,0,0,:8] =", [round(x, 4) for x in y[0, 0, 0, :8].tolist()])
    print("ref[0,0,64,:8]=", [round(x, 4) for x in ref[0, 0, 64, :8].tolist()])
    print("tc [0,0,64,:8]=", [round(x, 4) for x in y[0, 0, 64, :8].tolist()])
except Exception as e:
    print("spot check failed:", e)
