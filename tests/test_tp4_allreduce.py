"""Test if HCCL allreduce is M-dependent on Ascend NPU.
Run with: torchrun --nproc_per_node=4 tests/test_tp4_allreduce.py
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
    world_size = dist.get_world_size()
    device = torch.device("npu", rank)
    torch.npu.set_device(device)

    hidden = 2560
    torch.manual_seed(42 + rank)

    if rank == 0:
        print("=== HCCL allreduce M-dependence test ===")
        print("HCCL_DETERMINISTIC=%s" % os.environ.get("HCCL_DETERMINISTIC"))
        print("world_size=%d" % world_size)

    # Create test data with different M values
    x_full = torch.randn(256, hidden, dtype=torch.bfloat16, device=device)

    # Reference: allreduce with M=1
    ref_row = x_full[0:1].clone()
    dist.all_reduce(ref_row)
    ref = ref_row[0].clone()

    if rank == 0:
        print()
    for M in [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]:
        x = x_full[:M].clone()
        dist.all_reduce(x)
        diff = (x[0] - ref).abs().max().item()
        if rank == 0:
            tag = "OK" if diff == 0.0 else "MISMATCH"
            print("  M=%3d: first_row_diff=%.10f  %s" % (M, diff, tag))

    # Also test: does allreduce of the SAME data give different results
    # depending on what other data is in the tensor?
    if rank == 0:
        print()
        print("=== allreduce with different surrounding data ===")

    torch.manual_seed(100 + rank)
    base_row = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)

    results = {}
    for M in [1, 8, 32, 64, 128]:
        # Put base_row as first row, fill rest with different random data
        if M == 1:
            x = base_row.clone()
        else:
            torch.manual_seed(200 + rank + M)  # different padding per M
            padding = torch.randn(M - 1, hidden, dtype=torch.bfloat16, device=device)
            x = torch.cat([base_row.clone(), padding], dim=0)
        dist.all_reduce(x)
        results[M] = x[0].clone()

    for M in [8, 32, 64, 128]:
        diff = (results[M] - results[1]).abs().max().item()
        if rank == 0:
            tag = "OK" if diff == 0.0 else "MISMATCH"
            print("  M=%3d vs M=1 (same first row): diff=%.10f  %s" % (M, diff, tag))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
