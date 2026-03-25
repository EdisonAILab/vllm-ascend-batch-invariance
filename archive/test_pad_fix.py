"""
Test M-dimension padding as a fix for batch invariance.
Idea: Pad M to a consistent alignment (e.g., 128) so NPU always uses the same matmul algorithm.
"""
import os
import torch
import torch_npu

DEVICE = "npu:0"
SEQ_LEN = 32
BATCH_SIZE = 4

print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'not set')}")

# ========================================
# Test 1: Does padding M make torch.mm invariant?
# ========================================
print("\n=== Test 1: Padded torch.mm (pad M to 128) ===")

PAD_M = 128

for K, N in [(2560, 1024), (2560, 640), (2560, 512), (2560, 2560), (2560, 4096), (9728, 2560)]:
    torch.manual_seed(42)
    W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)

    M_single = SEQ_LEN  # 32
    M_batch = BATCH_SIZE * SEQ_LEN  # 128

    x_single_rows = [torch.randn(M_single, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(BATCH_SIZE)]
    x_batch = torch.cat(x_single_rows, dim=0)  # [128, K]

    with torch.no_grad():
        # Batch: M=128, already aligned to 128
        out_batch = torch.mm(x_batch, W)  # [128, N]

        # Singles: pad M from 32 to 128, compute, take first 32 rows
        diffs = []
        for i in range(BATCH_SIZE):
            x_padded = torch.zeros(PAD_M, K, dtype=torch.bfloat16, device=DEVICE)
            x_padded[:M_single] = x_single_rows[i]
            out_padded = torch.mm(x_padded, W)  # [128, N]
            out_single = out_padded[:M_single]  # [32, N]
            diff = (out_single - out_batch[i * M_single:(i + 1) * M_single]).abs().max().item()
            diffs.append(diff)

    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  padded mm ({K}x{N}): max_diff={max_diff:.8f}  {status}")

# ========================================
# Test 2: Does padding M to different alignments work?
# ========================================
print("\n=== Test 2: Different pad alignments for mm(2560, 1024) ===")

K, N = 2560, 1024
torch.manual_seed(42)
W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
x32 = torch.randn(32, K, dtype=torch.bfloat16, device=DEVICE)
x128 = torch.zeros(128, K, dtype=torch.bfloat16, device=DEVICE)
x128[:32] = x32

with torch.no_grad():
    ref = torch.mm(x128, W)[:32]  # M=128, take first 32 rows

    for pad_to in [64, 128, 256, 512]:
        x_pad = torch.zeros(pad_to, K, dtype=torch.bfloat16, device=DEVICE)
        x_pad[:32] = x32
        out = torch.mm(x_pad, W)[:32]
        diff = (out - ref).abs().max().item()
        status = "OK" if diff == 0.0 else "MISMATCH"
        print(f"  pad_to={pad_to} vs 128: max_diff={diff:.8f}  {status}")

# ========================================
# Test 3: What if we iterate per-sample (always M=32)?
# ========================================
print("\n=== Test 3: Per-sample iteration (always M=32) vs batch (M=128) ===")

for K, N in [(2560, 1024), (2560, 640), (2560, 512), (2560, 2560)]:
    torch.manual_seed(42)
    W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
    x_singles = [torch.randn(32, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)]
    x_batch = torch.cat(x_singles, dim=0)  # [128, K]

    with torch.no_grad():
        # Per-sample: each with M=32
        outs_single = [torch.mm(x_singles[i], W) for i in range(4)]
        outs_single_cat = torch.cat(outs_single, dim=0)  # [128, N]

        # Batch: M=128
        out_batch = torch.mm(x_batch, W)

        # Compare
        diff = (outs_single_cat - out_batch).abs().max().item()
        status = "OK" if diff == 0.0 else "MISMATCH"
        print(f"  per-sample(M=32) vs batch(M=128) ({K}x{N}): max_diff={diff:.8f}  {status}")

# ========================================
# Test 4: F.linear — patch to iterate per-sample
# ========================================
print("\n=== Test 4: Per-sample F.linear (force consistent M per sample) ===")

def linear_per_sample(input, weight, bias=None):
    """Process each batch element individually through F.linear to ensure same M dimension."""
    if input.dim() <= 2:
        return torch.nn.functional.linear(input, weight, bias)
    # input is [B, S, H] — process each [1, S, H] individually
    results = []
    for i in range(input.shape[0]):
        out_i = torch.nn.functional.linear(input[i:i+1], weight, bias)
        results.append(out_i)
    return torch.cat(results, dim=0)

lib = torch.library.Library("aten", "IMPL")
lib.impl("aten::linear", linear_per_sample, "PrivateUse1")

for K, N in [(2560, 1024), (2560, 640), (2560, 512), (2560, 2560), (2560, 4096)]:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
    x_singles = [torch.randn(1, 32, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)]
    x_batch = torch.cat(x_singles, dim=0)

    with torch.no_grad():
        out_batch = torch.nn.functional.linear(x_batch, W, b)
        diffs = []
        for i in range(4):
            out_single = torch.nn.functional.linear(x_singles[i], W, b)
            diff = (out_single[0] - out_batch[i]).abs().max().item()
            diffs.append(diff)

    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  per-sample F.linear ({K}->{N}): max_diff={max_diff:.8f}  {status}")

lib._destroy()
