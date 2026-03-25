"""Test M-dependence of NPU operators: npu_rms_norm, npu_add_rms_norm, SiLU, mul."""
import torch
import torch_npu

DEVICE = "npu:0"
torch.manual_seed(42)

hidden = 2560
eps = 1e-6
weight = torch.ones(hidden, dtype=torch.bfloat16, device=DEVICE)

x_full = torch.randn(128, hidden, dtype=torch.bfloat16, device=DEVICE)

print("=== npu_rms_norm M-dependence ===")
results = {}
for M in [1, 4, 16, 32, 64, 128]:
    x_sub = x_full[:M].contiguous()
    out, _ = torch_npu.npu_rms_norm(x_sub, weight, eps)
    results[M] = out[0].clone()

for M in [4, 16, 32, 64, 128]:
    diff = (results[M] - results[1]).abs().max().item()
    tag = "OK" if diff == 0.0 else "MISMATCH"
    print("  M=%3d vs M=1: max_diff=%.10f  %s" % (M, diff, tag))

print()
print("=== npu_add_rms_norm M-dependence ===")
residual_full = torch.randn(128, hidden, dtype=torch.bfloat16, device=DEVICE)
results2 = {}
for M in [1, 4, 16, 32, 64, 128]:
    x_sub = x_full[:M].contiguous()
    res_sub = residual_full[:M].contiguous()
    out, _, res_out = torch_npu.npu_add_rms_norm(x_sub, res_sub, weight, eps)
    results2[M] = (out[0].clone(), res_out[0].clone())

for M in [4, 16, 32, 64, 128]:
    diff_out = (results2[M][0] - results2[1][0]).abs().max().item()
    diff_res = (results2[M][1] - results2[1][1]).abs().max().item()
    tag = "OK" if diff_out == 0.0 and diff_res == 0.0 else "MISMATCH"
    print("  M=%3d vs M=1: out=%.10f res=%.10f  %s" % (M, diff_out, diff_res, tag))

print()
print("=== SiLU M-dependence ===")
silu = torch.nn.SiLU()
x_silu = torch.randn(128, 9728, dtype=torch.bfloat16, device=DEVICE)
ref_silu = silu(x_silu[:1].clone())
for M in [4, 16, 32, 64, 128]:
    out = silu(x_silu[:M].clone())
    diff = (out[0] - ref_silu[0]).abs().max().item()
    tag = "OK" if diff == 0.0 else "MISMATCH"
    print("  M=%3d vs M=1: diff=%.10f  %s" % (M, diff, tag))

print()
print("=== torch.mul M-dependence ===")
a = torch.randn(128, 9728, dtype=torch.bfloat16, device=DEVICE)
b = torch.randn(128, 9728, dtype=torch.bfloat16, device=DEVICE)
ref_mul = a[:1] * b[:1]
for M in [4, 16, 32, 64, 128]:
    out = a[:M] * b[:M]
    diff = (out[0] - ref_mul[0]).abs().max().item()
    tag = "OK" if diff == 0.0 else "MISMATCH"
    print("  M=%3d vs M=1: diff=%.10f  %s" % (M, diff, tag))

print()
print("=== F.linear (native, no padding) M-dependence ===")
import torch.nn.functional as F
w = torch.randn(9728, 2560, dtype=torch.bfloat16, device=DEVICE)
x_lin = torch.randn(128, 2560, dtype=torch.bfloat16, device=DEVICE)
ref_lin = F.linear(x_lin[:1], w)
for M in [4, 16, 32, 64, 128]:
    out = F.linear(x_lin[:M], w)
    diff = (out[0] - ref_lin[0]).abs().max().item()
    tag = "OK" if diff == 0.0 else "MISMATCH"
    print("  M=%3d vs M=1: diff=%.10f  %s" % (M, diff, tag))
