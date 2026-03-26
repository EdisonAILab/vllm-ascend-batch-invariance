"""Test if HCCL reduce_scatter and all_gather are M-dependent.
Run with: torchrun --nproc_per_node=4 tests/test_tp4_reducescatter.py
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
        print("=== HCCL reduce_scatter M-dependence test ===")
        print("world_size=%d" % world_size)

    # Test reduce_scatter: input [M, hidden*world_size] -> output [M, hidden]
    # Each rank contributes [M, hidden*world_size], output is scattered
    x_full = torch.randn(256, hidden * world_size, dtype=torch.bfloat16, device=device)

    results_rs = {}
    for M in [1, 4, 16, 32, 64, 128]:
        x = x_full[:M].clone()
        output = torch.empty(M, hidden, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(output, x)
        results_rs[M] = output[0].clone()

    if rank == 0:
        print("\nreduce_scatter first row comparison:")
        for M in [4, 16, 32, 64, 128]:
            diff = (results_rs[M] - results_rs[1]).abs().max().item()
            tag = "OK" if diff == 0.0 else "MISMATCH"
            print("  M=%3d vs M=1: diff=%.10f  %s" % (M, diff, tag))

    # Test all_gather: input [M, hidden] -> output [M, hidden*world_size]
    y_full = torch.randn(256, hidden, dtype=torch.bfloat16, device=device)

    results_ag = {}
    for M in [1, 4, 16, 32, 64, 128]:
        y = y_full[:M].clone()
        output = torch.empty(M, hidden * world_size, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(output, y)
        results_ag[M] = output[0].clone()

    if rank == 0:
        print("\nall_gather first row comparison:")
        for M in [4, 16, 32, 64, 128]:
            diff = (results_ag[M] - results_ag[1]).abs().max().item()
            tag = "OK" if diff == 0.0 else "MISMATCH"
            print("  M=%3d vs M=1: diff=%.10f  %s" % (M, diff, tag))

    # Test with vLLM's group comm (reduce_scatter along dim=0)
    # This is what MLPRowParallelOp actually does:
    # output = self.comm_group.reduce_scatter(output_parallel, 0)
    if rank == 0:
        print("\n=== Simulating MLPRowParallelOp reduce_scatter ===")

    # Each rank has [M, output_dim]. reduce_scatter along dim=0 means
    # each rank gets M/world_size rows of the reduced result.
    # But vLLM's reduce_scatter(tensor, dim=0) splits tensor along dim=0
    for M_total in [4, 8, 16, 32, 64, 128]:
        if M_total % world_size != 0:
            continue
        M_per_rank = M_total // world_size
        x = torch.randn(M_total, hidden, dtype=torch.bfloat16, device=device)
        # For consistent test: use same seed
        torch.manual_seed(42 + rank)
        x = torch.randn(M_total, hidden, dtype=torch.bfloat16, device=device)
        output = torch.empty(M_per_rank, hidden, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(output, x)

        # Compare first row with M_total=4 case
        if M_total == 4:
            ref = output[0].clone()
        else:
            diff = (output[0] - ref).abs().max().item()
            if rank == 0:
                tag = "OK" if diff == 0.0 else "MISMATCH"
                print("  M_total=%3d (M_per_rank=%d): diff=%.10f  %s" % (M_total, M_per_rank, diff, tag))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
