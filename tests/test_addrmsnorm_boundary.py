"""Find exact M boundary where npu_add_rms_norm becomes non-invariant."""
import torch
import torch_npu

DEVICE = "npu:0"
torch.manual_seed(42)

hidden = 2560
eps = 1e-6
weight = torch.ones(hidden, dtype=torch.bfloat16, device=DEVICE)
x_full = torch.randn(256, hidden, dtype=torch.bfloat16, device=DEVICE)
residual_full = torch.randn(256, hidden, dtype=torch.bfloat16, device=DEVICE)

# Reference: M=1
x_ref = x_full[:1].contiguous()
res_ref = residual_full[:1].contiguous()
ref_out, _, ref_res = torch_npu.npu_add_rms_norm(x_ref, res_ref, weight, eps)

print("=== npu_add_rms_norm: finding M boundary ===")
for M in range(1, 130):
    x_sub = x_full[:M].contiguous()
    res_sub = residual_full[:M].contiguous()
    out, _, res_out = torch_npu.npu_add_rms_norm(x_sub, res_sub, weight, eps)
    diff_out = (out[0] - ref_out[0]).abs().max().item()
    diff_res = (res_out[0] - ref_res[0]).abs().max().item()
    if diff_out > 0 or diff_res > 0:
        print("  M=%3d: out_diff=%.10f res_diff=%.10f  FIRST MISMATCH" % (M, diff_out, diff_res))
        # Show a few more
        for M2 in range(M+1, min(M+5, 130)):
            x_sub2 = x_full[:M2].contiguous()
            res_sub2 = residual_full[:M2].contiguous()
            out2, _, _ = torch_npu.npu_add_rms_norm(x_sub2, res_sub2, weight, eps)
            d2 = (out2[0] - ref_out[0]).abs().max().item()
            print("  M=%3d: out_diff=%.10f" % (M2, d2))
        break
else:
    print("  All M from 1 to 129 are OK")

# Also check: is the issue only for M>=64, or does it depend on the actual data?
print()
print("=== Checking if ALL rows match between different M values ===")
for M in [32, 48, 64, 96, 128]:
    x_sub = x_full[:M].contiguous()
    res_sub = residual_full[:M].contiguous()
    out, _, _ = torch_npu.npu_add_rms_norm(x_sub, res_sub, weight, eps)
    # Compare each row against M=1 single-row processing
    max_diff = 0.0
    mismatched_rows = 0
    for row in range(M):
        x_single = x_full[row:row+1].contiguous()
        res_single = residual_full[row:row+1].contiguous()
        single_out, _, _ = torch_npu.npu_add_rms_norm(x_single, res_single, weight, eps)
        d = (out[row] - single_out[0]).abs().max().item()
        if d > 0:
            mismatched_rows += 1
            max_diff = max(max_diff, d)
    tag = "OK" if mismatched_rows == 0 else "MISMATCH"
    print("  M=%3d: %d/%d rows differ, max_diff=%.10f  %s" % (M, mismatched_rows, M, max_diff, tag))

# Test npu_rms_norm (without add) more thoroughly
print()
print("=== npu_rms_norm: check all rows ===")
for M in [32, 48, 64, 96, 128]:
    x_sub = x_full[:M].contiguous()
    out, _ = torch_npu.npu_rms_norm(x_sub, weight, eps)
    max_diff = 0.0
    mismatched_rows = 0
    for row in range(M):
        x_single = x_full[row:row+1].contiguous()
        single_out, _ = torch_npu.npu_rms_norm(x_single, weight, eps)
        d = (out[row] - single_out[0]).abs().max().item()
        if d > 0:
            mismatched_rows += 1
            max_diff = max(max_diff, d)
    tag = "OK" if mismatched_rows == 0 else "MISMATCH"
    print("  M=%3d: %d/%d rows differ, max_diff=%.10f  %s" % (M, mismatched_rows, M, max_diff, tag))
