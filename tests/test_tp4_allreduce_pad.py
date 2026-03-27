"""Test: pad allreduce tensor to fixed size instead of chunking.
Run with: torchrun --nproc_per_node=4 tests/test_tp4_allreduce_pad.py
"""
import os
os.environ.setdefault("HCCL_DETERMINISTIC", "true")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch
import torch.distributed as dist
import torch_npu

def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    device = torch.device("npu", rank)
    torch.npu.set_device(device)

    hidden = 2560
    torch.manual_seed(200 + rank)
    x = torch.randn(1024, hidden, dtype=torch.bfloat16, device=device)

    # Reference: M=1, padded to PAD_SIZE
    if rank == 0:
        print("=== Padded allreduce invariance test ===\n")

    for PAD_TO in [128, 256, 384, 512, 1024]:
        # Reference: single row padded to PAD_TO
        ref_padded = torch.zeros(PAD_TO, hidden, dtype=torch.bfloat16, device=device)
        ref_padded[0] = x[0]
        dist.all_reduce(ref_padded)
        ref = ref_padded[0].clone()

        fails = 0
        for M in [2, 4, 8, 16, 64, 128, 256, 384, 412, 500, 512, 768, 1024]:
            if M > PAD_TO:
                continue
            y = torch.zeros(PAD_TO, hidden, dtype=torch.bfloat16, device=device)
            y[:M] = x[:M]
            dist.all_reduce(y)
            d = (y[0] - ref).abs().max().item()
            if d > 0:
                fails += 1
                if rank == 0:
                    print("  PAD_TO=%4d M=%4d: diff=%.10f MISMATCH" % (PAD_TO, M, d))

        if rank == 0:
            if fails == 0:
                print("  PAD_TO=%4d: ALL OK (tested M up to %d)" % (PAD_TO, min(1024, PAD_TO)))
            else:
                print("  PAD_TO=%4d: %d failures" % (PAD_TO, fails))

    # Compare performance: chunked vs padded allreduce
    if rank == 0:
        print("\n=== Performance: chunked(384) vs padded(512) vs padded(1024) ===")

    import time
    M = 600
    data = torch.randn(M, hidden, dtype=torch.bfloat16, device=device)
    ITERS = 100

    # Warmup
    for _ in range(10):
        tmp = data.clone()
        dist.all_reduce(tmp)

    # Native allreduce (no fix)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        tmp = data.clone()
        dist.all_reduce(tmp)
    torch.npu.synchronize()
    t_native = (time.time() - t0) / ITERS

    # Chunked allreduce (384)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        tmp = data.clone()
        for s in range(0, M, 384):
            e = min(s + 384, M)
            dist.all_reduce(tmp[s:e])
    torch.npu.synchronize()
    t_chunked = (time.time() - t0) / ITERS

    # Padded allreduce (1024)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        tmp_pad = torch.zeros(1024, hidden, dtype=torch.bfloat16, device=device)
        tmp_pad[:M] = data
        dist.all_reduce(tmp_pad)
    torch.npu.synchronize()
    t_padded_1024 = (time.time() - t0) / ITERS

    # Padded allreduce (512)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        tmp_pad = torch.zeros(512, hidden, dtype=torch.bfloat16, device=device)
        tmp_pad[:min(M, 512)] = data[:min(M, 512)]
        dist.all_reduce(tmp_pad)
        # Need second call for remainder
        if M > 512:
            tmp_pad2 = torch.zeros(512, hidden, dtype=torch.bfloat16, device=device)
            tmp_pad2[:M-512] = data[512:]
            dist.all_reduce(tmp_pad2)
    torch.npu.synchronize()
    t_padded_512 = (time.time() - t0) / ITERS

    if rank == 0:
        print("  M=%d, hidden=%d" % (M, hidden))
        print("  Native:        %.4f ms (baseline)" % (t_native * 1000))
        print("  Chunked(384):  %.4f ms (%.1fx)" % (t_chunked * 1000, t_chunked / t_native))
        print("  Padded(1024):  %.4f ms (%.1fx)" % (t_padded_1024 * 1000, t_padded_1024 / t_native))
        print("  Padded(512):   %.4f ms (%.1fx)" % (t_padded_512 * 1000, t_padded_512 / t_native))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
