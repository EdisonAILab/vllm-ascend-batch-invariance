"""Pinpoint which step in chunked_linear+allreduce fails at M=512 with TP=4."""
import os
os.environ.setdefault("HCCL_DETERMINISTIC", "true")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch
import torch.distributed as dist
import torch_npu

def chunked_linear(x, w, chunk=128):
    M = x.shape[0]
    N = w.shape[0]
    if M <= chunk:
        if M < chunk:
            x = torch.cat([x, x.new_zeros(chunk - M, x.shape[1])], dim=0)
        return torch.nn.functional.linear(x, w)[:M]
    out = x.new_empty(M, N)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        n = e - s
        c = x[s:e]
        if n < chunk:
            c = torch.cat([c, x.new_zeros(chunk - n, x.shape[1])], dim=0)
        out[s:e] = torch.nn.functional.linear(c, w)[:n]
    return out

def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("npu", rank)
    torch.npu.set_device(device)

    # TP=4 down_proj dimensions
    torch.manual_seed(42 + rank)
    w = torch.randn(2560, 2432, dtype=torch.bfloat16, device=device)

    torch.manual_seed(100)
    x = torch.randn(600, 2432, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print("=== Pinpointing M=512 failure ===")
        print("Step 1: chunked_linear alone (no allreduce)")

    # Test chunked_linear alone
    ref = chunked_linear(x[:1], w)
    for M in [384, 448, 480, 496, 512]:
        out = chunked_linear(x[:M], w)
        d = (out[0] - ref[0]).abs().max().item()
        if rank == 0:
            print("  M=%3d: diff=%.10f %s" % (M, d, "OK" if d == 0 else "MISMATCH"))

    if rank == 0:
        print("\nStep 2: chunked_linear + allreduce")

    ref_ar = chunked_linear(x[:1], w)
    dist.all_reduce(ref_ar)
    for M in [384, 448, 480, 496, 512]:
        out_ar = chunked_linear(x[:M], w)
        dist.all_reduce(out_ar)
        d = (out_ar[0] - ref_ar[0]).abs().max().item()
        if rank == 0:
            print("  M=%3d: diff=%.10f %s" % (M, d, "OK" if d == 0 else "MISMATCH"))

    if rank == 0:
        print("\nStep 3: allreduce alone (same data, different M)")

    torch.manual_seed(200 + rank)
    y = torch.randn(600, 2560, dtype=torch.bfloat16, device=device)
    ref_y = y[:1].clone()
    dist.all_reduce(ref_y)
    for M in [384, 448, 480, 496, 512]:
        y_m = y[:M].clone()
        dist.all_reduce(y_m)
        d = (y_m[0] - ref_y[0]).abs().max().item()
        if rank == 0:
            print("  M=%3d: diff=%.10f %s" % (M, d, "OK" if d == 0 else "MISMATCH"))

    if rank == 0:
        print("\nStep 4: Row-by-row linear (no chunking, each row independently)")

    ref_rbr = torch.nn.functional.linear(
        torch.cat([x[:1], x.new_zeros(127, x.shape[1])], dim=0), w)[:1]
    dist.all_reduce(ref_rbr)
    for M in [384, 448, 480, 496, 512]:
        out_rows = []
        for i in range(M):
            row_padded = torch.cat([x[i:i+1], x.new_zeros(127, x.shape[1])], dim=0)
            row_out = torch.nn.functional.linear(row_padded, w)[:1]
            out_rows.append(row_out)
        out_rbr = torch.cat(out_rows, dim=0)
        dist.all_reduce(out_rbr)
        d = (out_rbr[0] - ref_rbr[0]).abs().max().item()
        if rank == 0:
            print("  M=%3d: diff=%.10f %s" % (M, d, "OK" if d == 0 else "MISMATCH"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
