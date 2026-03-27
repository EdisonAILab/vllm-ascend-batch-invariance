"""Find exact allreduce M boundary on Ascend NPU with TP=4."""
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
    x = torch.randn(600, hidden, dtype=torch.bfloat16, device=device)

    ref = x[:1].clone()
    dist.all_reduce(ref)

    if rank == 0:
        print("=== allreduce M boundary search ===")

    for M in range(380, 460, 4):
        y = x[:M].clone()
        dist.all_reduce(y)
        d = (y[0] - ref[0]).abs().max().item()
        if rank == 0:
            tag = "OK" if d == 0 else "MISMATCH"
            print("  M=%3d: diff=%.10f  %s" % (M, d, tag))

    # Also test: does chunking allreduce to ≤384 fix it?
    if rank == 0:
        print("\n=== Chunked allreduce (chunk=384) ===")

    for M in [384, 448, 512, 600]:
        y2 = x[:M].clone()
        # Chunked allreduce
        chunk = 384
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            dist.all_reduce(y2[s:e])
        d2 = (y2[0] - ref[0]).abs().max().item()
        if rank == 0:
            tag = "OK" if d2 == 0 else "MISMATCH"
            print("  M=%3d: diff=%.10f  %s" % (M, d2, tag))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
