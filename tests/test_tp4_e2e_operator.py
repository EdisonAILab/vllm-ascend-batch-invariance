"""
End-to-end TP=4 operator test: F.linear + allreduce M-invariance.
Run with: torchrun --nproc_per_node=4 tests/test_tp4_e2e_operator.py
Simulates one transformer layer's RowParallel path with different M values.
"""
import os
os.environ.setdefault("HCCL_DETERMINISTIC", "true")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch
import torch.distributed as dist
import torch_npu

def chunked_linear(x, w, chunk=128):
    """Our batch-invariant chunked F.linear."""
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

def decomposed_add_rms_norm(x, residual, weight, eps=1e-6):
    """Our batch-invariant decomposed add + rms_norm."""
    x = x + residual
    residual = x.clone()
    x, _ = torch_npu.npu_rms_norm(x, weight, eps)
    return x, residual

def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("npu", rank)
    torch.npu.set_device(device)

    hidden = 2560
    intermediate = 9728
    # TP=4 dimensions
    hidden_per_tp = hidden  # RowParallel output is full hidden
    intermediate_per_tp = intermediate // world_size  # 2432

    torch.manual_seed(42)
    # Shared weights (same across ranks in real TP, but split)
    # For this test, each rank has its own shard
    torch.manual_seed(42 + rank)
    w_down = torch.randn(hidden, intermediate_per_tp, dtype=torch.bfloat16, device=device)
    rms_weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)

    # Shared input data across ranks
    torch.manual_seed(100)
    x_full = torch.randn(600, intermediate_per_tp, dtype=torch.bfloat16, device=device)
    residual_full = torch.randn(600, hidden, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print("=== TP=4 E2E operator test: chunked_linear + allreduce + add_rms_norm ===")

    # Reference: M=1
    x_ref = chunked_linear(x_full[:1], w_down)
    dist.all_reduce(x_ref)
    x_ref, res_ref = decomposed_add_rms_norm(x_ref, residual_full[:1], rms_weight)

    for M in [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
        x_m = chunked_linear(x_full[:M], w_down)
        dist.all_reduce(x_m)
        x_m, res_m = decomposed_add_rms_norm(x_m, residual_full[:M], rms_weight)

        diff = (x_m[0] - x_ref[0]).abs().max().item()
        diff_res = (res_m[0] - res_ref[0]).abs().max().item()
        if rank == 0:
            tag = "OK" if diff == 0 and diff_res == 0 else "MISMATCH"
            print("  M=%3d: norm_diff=%.10f res_diff=%.10f  %s" % (M, diff, diff_res, tag))

    # Now test: full transformer layer simulation (2 linear + allreduce + rms_norm)
    if rank == 0:
        print("\n=== Full layer simulation: gate_up -> silu*gate -> down -> allreduce -> add_rms ===")

    torch.manual_seed(42 + rank)
    w_gate_up = torch.randn(intermediate_per_tp * 2, hidden, dtype=torch.bfloat16, device=device)
    w_down2 = torch.randn(hidden, intermediate_per_tp, dtype=torch.bfloat16, device=device)

    torch.manual_seed(200)
    input_full = torch.randn(600, hidden, dtype=torch.bfloat16, device=device)
    residual2 = torch.randn(600, hidden, dtype=torch.bfloat16, device=device)

    def mlp_forward(x, M):
        """Simulate MLP: gate_up -> silu*gate -> down -> allreduce -> add_rms"""
        h = chunked_linear(x[:M], w_gate_up)
        gate = h[:, :intermediate_per_tp]
        up = h[:, intermediate_per_tp:]
        h = torch.nn.functional.silu(gate) * up
        out = chunked_linear(h, w_down2)
        dist.all_reduce(out)
        out, res = decomposed_add_rms_norm(out, residual2[:M], rms_weight)
        return out

    ref_mlp = mlp_forward(input_full, 1)
    for M in [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
        out_mlp = mlp_forward(input_full, M)
        diff = (out_mlp[0] - ref_mlp[0]).abs().max().item()
        if rank == 0:
            tag = "OK" if diff == 0 else "MISMATCH"
            print("  M=%3d: diff=%.10f  %s" % (M, diff, tag))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
