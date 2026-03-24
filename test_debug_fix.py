"""
Debug: verify the aten::linear patch is intercepted, and test torch.mm at actual M dimensions.
"""
import os
import torch
import torch_npu

DEVICE = "npu:0"
SEQ_LEN = 32
BATCH_SIZE = 4

print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'not set')}")

# ========================================
# Test 1: torch.mm at actual M dimensions (M=32 vs M=128)
# ========================================
print("\n=== Test 1: torch.mm at actual M dimensions ===")
for K, N in [(2560, 1024), (2560, 640), (2560, 512), (2560, 2560), (2560, 4096), (2560, 9728), (9728, 2560)]:
    torch.manual_seed(42)
    W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)

    # M=32 (single: 1 seq * 32 tokens) vs M=128 (batch: 4 seqs * 32 tokens)
    M_single = SEQ_LEN  # 32
    M_batch = BATCH_SIZE * SEQ_LEN  # 128

    x_single_rows = [torch.randn(M_single, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(BATCH_SIZE)]
    x_batch = torch.cat(x_single_rows, dim=0)  # [128, K]

    with torch.no_grad():
        out_batch = torch.mm(x_batch, W)  # [128, N]
        diffs = []
        for i in range(BATCH_SIZE):
            out_single = torch.mm(x_single_rows[i], W)  # [32, N]
            # Compare rows i*32..(i+1)*32 of batch output vs single output
            diff = (out_single - out_batch[i * M_single:(i + 1) * M_single]).abs().max().item()
            diffs.append(diff)

    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  mm M=32vs128 ({K}x{N}): max_diff={max_diff:.8f}  {status}")

# ========================================
# Test 2: F.linear at actual 3D shapes
# ========================================
print("\n=== Test 2: F.linear at actual 3D shapes [B,32,K] ===")
for K, N in [(2560, 1024), (2560, 640), (2560, 512), (2560, 2560), (2560, 4096), (2560, 9728)]:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    x_singles = [torch.randn(1, SEQ_LEN, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(BATCH_SIZE)]
    x_batch = torch.cat(x_singles, dim=0)  # [4, 32, K]

    with torch.no_grad():
        out_batch = torch.nn.functional.linear(x_batch, W, b)
        diffs = []
        for i in range(BATCH_SIZE):
            out_single = torch.nn.functional.linear(x_singles[i], W, b)
            diff = (out_single[0] - out_batch[i]).abs().max().item()
            diffs.append(diff)

    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  F.linear [4,32,{K}]->{N}: max_diff={max_diff:.8f}  {status}")

# ========================================
# Test 3: Verify aten::linear dispatch interception
# ========================================
print("\n=== Test 3: Verify aten::linear dispatch interception ===")

_intercept_count = 0

def _debug_linear(input, weight, bias=None):
    global _intercept_count
    _intercept_count += 1
    orig_shape = input.shape
    flat = input.reshape(-1, orig_shape[-1])
    out = torch.mm(flat, weight.t())
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], weight.shape[0])

lib = torch.library.Library("aten", "IMPL")
lib.impl("aten::linear", _debug_linear, "PrivateUse1")

torch.manual_seed(42)
x = torch.randn(4, 32, 2560, dtype=torch.bfloat16, device=DEVICE)
W = torch.randn(1024, 2560, dtype=torch.bfloat16, device=DEVICE)
b = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)

with torch.no_grad():
    out = torch.nn.functional.linear(x, W, b)

print(f"  F.linear intercepted: {_intercept_count > 0} (count={_intercept_count})")

# Test if the intercepted version is batch-invariant
x_singles = [torch.randn(1, 32, 2560, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)]
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
print(f"  Intercepted F.linear (2560->1024) batch invariance: max_diff={max_diff:.8f}  {status}")

lib._destroy()

# ========================================
# Test 4: torch.mm with nn.Linear weights directly
# ========================================
print("\n=== Test 4: Manual mm-based linear at actual model dimensions ===")
for K, N in [(2560, 1024), (2560, 640), (2560, 512)]:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)  # [out, in] like nn.Linear
    b = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    x_singles = [torch.randn(1, SEQ_LEN, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(BATCH_SIZE)]
    x_batch = torch.cat(x_singles, dim=0)

    with torch.no_grad():
        # Manual: flatten, mm, bias, reshape
        flat_batch = x_batch.reshape(-1, K)  # [128, K]
        out_batch = torch.mm(flat_batch, W.t()) + b  # [128, N]
        out_batch = out_batch.reshape(BATCH_SIZE, SEQ_LEN, N)

        diffs = []
        for i in range(BATCH_SIZE):
            flat_single = x_singles[i].reshape(-1, K)  # [32, K]
            out_single = torch.mm(flat_single, W.t()) + b  # [32, N]
            out_single = out_single.reshape(1, SEQ_LEN, N)
            diff = (out_single[0] - out_batch[i]).abs().max().item()
            diffs.append(diff)

    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  manual mm ({K}->{N}): max_diff={max_diff:.8f}  {status}")
