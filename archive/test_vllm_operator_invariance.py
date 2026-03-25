"""
Test which linear operators in Qwen3-4B are M-dependent on NPU.

In vLLM, tokens are packed as [total_tokens, hidden_size]. We test each
linear projection with different M values (total_tokens) to see which
produce different results for the same input row.
"""
import os
import sys
import json

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
DEVICE = "npu:0"

config = AutoConfig.from_pretrained(MODEL_PATH)
print(f"hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}")
print(f"num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}")
print(f"head_dim={config.hidden_size // config.num_attention_heads}")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE).eval()
layer0 = model.model.layers[0]

# Get all linear projections from layer 0
projections = {
    "q_proj": layer0.self_attn.q_proj,       # 2560 -> 4096
    "k_proj": layer0.self_attn.k_proj,       # 2560 -> 1024
    "v_proj": layer0.self_attn.v_proj,       # 2560 -> 1024
    "o_proj": layer0.self_attn.o_proj,       # 4096 -> 2560
    "gate_proj": layer0.mlp.gate_proj,       # 2560 -> 9728
    "up_proj": layer0.mlp.up_proj,           # 2560 -> 9728
    "down_proj": layer0.mlp.down_proj,       # 9728 -> 2560
}

print(f"\nProjection shapes:")
for name, proj in projections.items():
    print(f"  {name}: {proj.in_features} -> {proj.out_features}")

# Test: for each projection, run with M=1 vs M=N (same input row replicated)
# If the output for the same row differs, the operator is M-dependent.
print(f"\n{'=' * 70}")
print(f"Test: M-dependence of each linear projection")
print(f"M values tested: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512")
print(f"{'=' * 70}")

M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

for name, proj in projections.items():
    torch.manual_seed(42)
    # Create a single input row
    x_single = torch.randn(1, proj.in_features, dtype=torch.bfloat16, device=DEVICE)

    # Run with M=1 (baseline)
    with torch.no_grad():
        y_baseline = proj(x_single)  # [1, out_features]

    mismatches = []
    for M in M_values:
        if M == 1:
            continue
        # Create M rows, all identical to x_single
        x_batch = x_single.expand(M, -1).contiguous()  # [M, in_features]
        with torch.no_grad():
            y_batch = proj(x_batch)  # [M, out_features]

        # Compare first row of batch with baseline
        diff = (y_baseline[0] - y_batch[0]).abs().max().item()
        if diff > 0:
            mismatches.append((M, diff))

    if mismatches:
        print(f"  {name:12s} ({proj.in_features}->{proj.out_features}): "
              f"NON-INVARIANT  M={[m for m,_ in mismatches]}, "
              f"max_diff={max(d for _,d in mismatches):.8f}")
    else:
        print(f"  {name:12s} ({proj.in_features}->{proj.out_features}): OK (invariant)")

# Also test torch.mm directly with different weight shapes
print(f"\n{'=' * 70}")
print(f"Test: torch.mm M-dependence by output dimension")
print(f"{'=' * 70}")

# Test various output dimensions to find the threshold
in_dim = config.hidden_size  # 2560
test_out_dims = [512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 4096, 9728]

torch.manual_seed(42)
x_single = torch.randn(1, in_dim, dtype=torch.bfloat16, device=DEVICE)

for out_dim in test_out_dims:
    torch.manual_seed(100 + out_dim)
    weight = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)

    y_base = torch.mm(x_single, weight.t())

    mismatches = []
    for M in [2, 4, 8, 16, 32, 64, 128, 256]:
        x_batch = x_single.expand(M, -1).contiguous()
        y_batch = torch.mm(x_batch, weight.t())
        diff = (y_base[0] - y_batch[0]).abs().max().item()
        if diff > 0:
            mismatches.append((M, diff))

    if mismatches:
        print(f"  out_dim={out_dim:5d}: NON-INVARIANT  "
              f"first_fail_M={mismatches[0][0]}, max_diff={max(d for _,d in mismatches):.8f}")
    else:
        print(f"  out_dim={out_dim:5d}: OK")

# Test with F.linear too
print(f"\n{'=' * 70}")
print(f"Test: F.linear M-dependence by output dimension")
print(f"{'=' * 70}")

import torch.nn.functional as F

for out_dim in test_out_dims:
    torch.manual_seed(100 + out_dim)
    weight = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)

    y_base = F.linear(x_single, weight)

    mismatches = []
    for M in [2, 4, 8, 16, 32, 64, 128, 256]:
        x_batch = x_single.expand(M, -1).contiguous()
        y_batch = F.linear(x_batch, weight)
        diff = (y_base[0] - y_batch[0]).abs().max().item()
        if diff > 0:
            mismatches.append((M, diff))

    if mismatches:
        print(f"  out_dim={out_dim:5d}: NON-INVARIANT  "
              f"first_fail_M={mismatches[0][0]}, max_diff={max(d for _,d in mismatches):.8f}")
    else:
        print(f"  out_dim={out_dim:5d}: OK")

del model
torch_npu.npu.empty_cache()
