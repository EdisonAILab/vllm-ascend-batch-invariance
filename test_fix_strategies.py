"""
Test fix strategies for M-dependent operators in vLLM on NPU.

Strategy 1: Row-by-row processing (M=1 per call)
Strategy 2: Fixed chunk size (pad M to power of 2)
Strategy 3: Process all as M=1 via torch.mm row loop
"""
import os
import sys
import time

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch
import torch_npu

DEVICE = "npu:0"
torch.manual_seed(42)

# Simulate the non-invariant operators
in_dim = 2560
out_dims = {
    "o_proj": 2560,
    "up_proj": 9728,
    "down_proj_in": 9728,  # down_proj input dim
}

# Test with realistic M values from vLLM (total tokens in a batch)
M_single = 65   # single prompt
M_batch = 119   # batch of 4 prompts

weight_2560 = torch.randn(2560, 2560, dtype=torch.bfloat16, device=DEVICE)
weight_9728 = torch.randn(9728, 2560, dtype=torch.bfloat16, device=DEVICE)
weight_down = torch.randn(2560, 9728, dtype=torch.bfloat16, device=DEVICE)

x_2560 = torch.randn(M_batch, 2560, dtype=torch.bfloat16, device=DEVICE)
x_9728 = torch.randn(M_batch, 9728, dtype=torch.bfloat16, device=DEVICE)

print("=" * 70)
print("Strategy 1: Row-by-row torch.mm (M=1 per call)")
print("=" * 70)

def linear_row_by_row(x, weight, bias=None):
    """Process each row independently."""
    M = x.shape[0]
    results = []
    for i in range(M):
        row = x[i:i+1]  # [1, K]
        out = torch.mm(row, weight.t())  # [1, N]
        results.append(out)
    result = torch.cat(results, dim=0)
    if bias is not None:
        result = result + bias
    return result

# Verify row-by-row is M-invariant
y_single = linear_row_by_row(x_2560[:M_single], weight_9728)
y_batch = linear_row_by_row(x_2560[:M_batch], weight_9728)
diff = (y_single[:M_single] - y_batch[:M_single]).abs().max().item()
print(f"  up_proj row-by-row: M={M_single} vs M={M_batch} first {M_single} rows diff={diff:.8f}")

# Benchmark
t0 = time.time()
for _ in range(10):
    _ = linear_row_by_row(x_2560, weight_9728)
torch_npu.npu.synchronize()
t_rowbyrow = (time.time() - t0) / 10
print(f"  Time (M={M_batch}, out=9728): {t_rowbyrow*1000:.1f}ms per call")

t0 = time.time()
for _ in range(10):
    _ = torch.mm(x_2560, weight_9728.t())
torch_npu.npu.synchronize()
t_native = (time.time() - t0) / 10
print(f"  Native torch.mm time: {t_native*1000:.1f}ms per call")
print(f"  Overhead: {t_rowbyrow/t_native:.1f}x")

print("\n" + "=" * 70)
print("Strategy 2: Chunk to fixed size (pad M)")
print("=" * 70)

def linear_chunked(x, weight, bias=None, chunk_size=16):
    """Process in fixed-size chunks, padding if needed."""
    M = x.shape[0]
    if M <= chunk_size:
        # Pad to chunk_size
        if M < chunk_size:
            pad = torch.zeros(chunk_size - M, x.shape[1], dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, pad], dim=0)
        else:
            x_padded = x
        out = torch.mm(x_padded, weight.t())
        if bias is not None:
            out = out + bias
        return out[:M]
    else:
        # Process in chunks
        results = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            chunk = x[start:end]
            actual_len = chunk.shape[0]
            if actual_len < chunk_size:
                pad = torch.zeros(chunk_size - actual_len, x.shape[1], dtype=x.dtype, device=x.device)
                chunk = torch.cat([chunk, pad], dim=0)
            out = torch.mm(chunk, weight.t())
            if bias is not None:
                out = out + bias
            results.append(out[:actual_len])
        return torch.cat(results, dim=0)

# Test chunk sizes
for chunk_size in [8, 16, 32, 64]:
    y_single = linear_chunked(x_2560[:M_single], weight_9728, chunk_size=chunk_size)
    y_batch = linear_chunked(x_2560[:M_batch], weight_9728, chunk_size=chunk_size)
    diff = (y_single[:M_single] - y_batch[:M_single]).abs().max().item()

    t0 = time.time()
    for _ in range(10):
        _ = linear_chunked(x_2560, weight_9728, chunk_size=chunk_size)
    torch_npu.npu.synchronize()
    t_chunk = (time.time() - t0) / 10
    overhead = t_chunk / t_native

    status = "OK" if diff == 0.0 else f"MISMATCH diff={diff:.8f}"
    print(f"  chunk_size={chunk_size:3d}: {status}  time={t_chunk*1000:.1f}ms ({overhead:.1f}x)")

print("\n" + "=" * 70)
print("Strategy 3: Pad M to next power of 2")
print("=" * 70)

def next_power_of_2(n):
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p

def linear_padded_pow2(x, weight, bias=None):
    """Pad M to next power of 2."""
    M = x.shape[0]
    target_M = next_power_of_2(M)
    if M < target_M:
        pad = torch.zeros(target_M - M, x.shape[1], dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=0)
    out = torch.mm(x, weight.t())
    if bias is not None:
        out = out + bias
    return out[:M]

y_single = linear_padded_pow2(x_2560[:M_single], weight_9728)
y_batch = linear_padded_pow2(x_2560[:M_batch], weight_9728)
diff = (y_single[:M_single] - y_batch[:M_single]).abs().max().item()
print(f"  Padded pow2: M=65->128 vs M=119->128: diff={diff:.8f} {'OK' if diff == 0.0 else 'MISMATCH'}")

# This only works if both pad to the SAME target
y_s2 = linear_padded_pow2(x_2560[:26], weight_9728)  # 26->32
y_b2 = linear_padded_pow2(x_2560[:57], weight_9728)  # 57->64
# Compare first 26 rows
diff2 = (y_s2[:26] - y_b2[:26]).abs().max().item()
print(f"  Padded pow2: M=26->32 vs M=57->64: diff={diff2:.8f} {'OK' if diff2 == 0.0 else 'MISMATCH (different targets)'}")

# Strategy: pad all to a fixed large value
print("\n" + "=" * 70)
print("Strategy 4: Pad M to fixed value (e.g., 512)")
print("=" * 70)

def linear_fixed_pad(x, weight, bias=None, pad_to=512):
    """Pad M to a fixed value."""
    M = x.shape[0]
    if M > pad_to:
        # Process in chunks of pad_to
        results = []
        for start in range(0, M, pad_to):
            end = min(start + pad_to, M)
            chunk = x[start:end]
            out = linear_fixed_pad(chunk, weight, bias, pad_to)
            results.append(out)
        return torch.cat(results, dim=0)
    if M < pad_to:
        pad = torch.zeros(pad_to - M, x.shape[1], dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=0)
    out = torch.mm(x, weight.t())
    if bias is not None:
        out = out + bias
    return out[:M]

for pad_to in [128, 256, 512]:
    y_single = linear_fixed_pad(x_2560[:M_single], weight_9728, pad_to=pad_to)
    y_batch = linear_fixed_pad(x_2560[:M_batch], weight_9728, pad_to=pad_to)
    diff = (y_single[:M_single] - y_batch[:M_single]).abs().max().item()

    t0 = time.time()
    for _ in range(10):
        _ = linear_fixed_pad(x_2560, weight_9728, pad_to=pad_to)
    torch_npu.npu.synchronize()
    t_pad = (time.time() - t0) / 10
    overhead = t_pad / t_native

    status = "OK" if diff == 0.0 else f"MISMATCH diff={diff:.8f}"
    print(f"  pad_to={pad_to:4d}: {status}  time={t_pad*1000:.1f}ms ({overhead:.1f}x)")

torch_npu.npu.empty_cache()
