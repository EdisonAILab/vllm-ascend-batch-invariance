"""
Test M-invariance of F.linear with TP=4 specific weight dimensions.
Qwen3-4B with TP=4:
  QKV:     [M, 2560] -> [M, 1536]  (ColumnParallel)
  O_proj:  [M, 1024] -> [M, 2560]  (RowParallel + allreduce)
  Gate+Up: [M, 2560] -> [M, 4864]  (ColumnParallel)
  Down:    [M, 2432] -> [M, 2560]  (RowParallel + allreduce)
  LM_head: [M, 2560] -> [M, 151936] (full, not TP-split)
"""
import os
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
import torch
import torch_npu
import torch.nn.functional as F

DEVICE = "npu:0"
torch.manual_seed(42)

# TP=4 linear dimensions for Qwen3-4B
LINEARS = [
    ("QKV (col)",   2560, 1536),
    ("O_proj (row)", 1024, 2560),
    ("GateUp (col)", 2560, 4864),
    ("Down (row)",   2432, 2560),
    ("LM_head",      2560, 37984),  # 151936/4 per rank for lm_head
]

def test_native_invariance(name, in_dim, out_dim):
    """Test raw F.linear without padding."""
    w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)
    x = torch.randn(512, in_dim, dtype=torch.bfloat16, device=DEVICE)

    ref = F.linear(x[:1], w)
    diffs = {}
    for M in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        out = F.linear(x[:M], w)
        d = (out[0] - ref[0]).abs().max().item()
        diffs[M] = d

    first_fail = None
    for M, d in diffs.items():
        if d > 0 and first_fail is None:
            first_fail = M

    return diffs, first_fail

def test_padded_invariance(name, in_dim, out_dim, pad_to=128):
    """Test F.linear with M-padding to fixed size."""
    w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)
    x = torch.randn(512, in_dim, dtype=torch.bfloat16, device=DEVICE)

    # Reference: single row padded to pad_to
    x_ref = torch.cat([x[:1], x.new_zeros(pad_to - 1, in_dim)], dim=0)
    ref = F.linear(x_ref, w)[0]

    diffs = {}
    for M in [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
        # Pad M to multiple of pad_to
        M_padded = ((M + pad_to - 1) // pad_to) * pad_to
        x_pad = torch.cat([x[:M], x.new_zeros(M_padded - M, in_dim)], dim=0)
        out = F.linear(x_pad, w)
        d = (out[0] - ref).abs().max().item()
        diffs[M] = d

    first_fail = None
    for M, d in diffs.items():
        if d > 0 and first_fail is None:
            first_fail = M

    return diffs, first_fail

def test_chunked_invariance(name, in_dim, out_dim, chunk=128):
    """Test F.linear with our chunking approach."""
    w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)
    x = torch.randn(512, in_dim, dtype=torch.bfloat16, device=DEVICE)

    # Reference: single row in a chunk of size chunk
    x_ref = torch.cat([x[:1], x.new_zeros(chunk - 1, in_dim)], dim=0)
    ref = F.linear(x_ref, w)[0]

    diffs = {}
    for M in [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
        # Process in chunks of `chunk`
        out = x.new_empty(M, out_dim)
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            n = e - s
            c = x[s:e]
            if n < chunk:
                c = torch.cat([c, x.new_zeros(chunk - n, in_dim)], dim=0)
            out[s:e] = F.linear(c, w)[:n]
        d = (out[0] - ref).abs().max().item()
        diffs[M] = d

    first_fail = None
    for M, d in diffs.items():
        if d > 0 and first_fail is None:
            first_fail = M

    return diffs, first_fail

print("=" * 80)
print("CANN F.linear M-invariance test with TP=4 dimensions")
print("=" * 80)

for name, in_dim, out_dim in LINEARS:
    print("\n--- %s: [M, %d] -> [M, %d] ---" % (name, in_dim, out_dim))

    # Native (no fix)
    diffs_native, fail_native = test_native_invariance(name, in_dim, out_dim)
    print("  Native:  first_fail=M=%s" % (fail_native if fail_native else "none"))
    for M, d in diffs_native.items():
        if d > 0:
            print("    M=%3d: diff=%.10f" % (M, d))

    # Padded to 128
    diffs_padded, fail_padded = test_padded_invariance(name, in_dim, out_dim, 128)
    print("  Padded(128): first_fail=M=%s" % (fail_padded if fail_padded else "none"))
    for M, d in diffs_padded.items():
        if d > 0:
            print("    M=%3d: diff=%.10f" % (M, d))

    # Chunked to 128
    diffs_chunked, fail_chunked = test_chunked_invariance(name, in_dim, out_dim, 128)
    print("  Chunked(128): first_fail=M=%s" % (fail_chunked if fail_chunked else "none"))
    for M, d in diffs_chunked.items():
        if d > 0:
            print("    M=%3d: diff=%.10f" % (M, d))

# Also test: does padding to a LARGER size help?
print("\n" + "=" * 80)
print("Testing larger pad sizes for O_proj [M, 1024] -> [M, 2560]")
print("=" * 80)
in_dim, out_dim = 1024, 2560
w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)
x = torch.randn(1024, in_dim, dtype=torch.bfloat16, device=DEVICE)

for pad_to in [128, 256, 512, 1024]:
    x_ref = torch.cat([x[:1], x.new_zeros(pad_to - 1, in_dim)], dim=0)
    ref = F.linear(x_ref, w)[0]

    fails = 0
    max_d = 0.0
    for M in [128, 256, 384, 512, 640, 768, 896, 1024]:
        M_padded = ((M + pad_to - 1) // pad_to) * pad_to
        x_pad = torch.cat([x[:M], x.new_zeros(M_padded - M, in_dim)], dim=0)
        out = F.linear(x_pad, w)
        d = (out[0] - ref).abs().max().item()
        if d > 0:
            fails += 1
            max_d = max(max_d, d)

    tag = "OK" if fails == 0 else "FAIL(%d)" % fails
    print("  pad_to=%4d: %s  max_diff=%.10f" % (pad_to, tag, max_d))
