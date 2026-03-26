"""
Dequantize Qwen3-4B-FP8 to bfloat16 safetensors.
Reads FP8 weights + weight_scale_inv, dequantizes to bf16, saves as standard checkpoint.
"""
import os
import json
import torch
from safetensors.torch import load_file, save_file

FP8_DIR = "/home/bruceli/models/Qwen/Qwen3-4B-FP8"
OUT_DIR = "/home/bruceli/models/Qwen/Qwen3-4B-FP8-dequant"
os.makedirs(OUT_DIR, exist_ok=True)

# Load index
with open(os.path.join(FP8_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)

# Group weights by shard file
shard_to_keys = {}
for key, shard in index["weight_map"].items():
    shard_to_keys.setdefault(shard, []).append(key)

new_weight_map = {}
total_size = 0

for shard_name, keys in sorted(shard_to_keys.items()):
    print("Processing %s (%d keys)..." % (shard_name, len(keys)))
    shard_path = os.path.join(FP8_DIR, shard_name)
    tensors = load_file(shard_path)

    new_tensors = {}
    for key in sorted(keys):
        if key.endswith(".weight_scale_inv"):
            # Skip scale tensors - they'll be consumed during dequantization
            continue

        t = tensors[key]
        scale_key = key.replace(".weight", ".weight_scale_inv")

        if t.dtype == torch.float8_e4m3fn and scale_key in tensors:
            # Dequantize: weight_bf16 = weight_fp8.to(bf16) * scale_inv
            scale_inv = tensors[scale_key]
            # Block-wise dequantization (128x128 blocks)
            w_bf16 = t.to(torch.bfloat16)

            # weight_scale_inv shape tells us the block structure
            if scale_inv.dim() == 2:
                # Per-block scales: scale_inv shape [out_blocks, in_blocks]
                block_out = t.shape[0] // scale_inv.shape[0]
                block_in = t.shape[1] // scale_inv.shape[1]
                # Repeat scale to match weight shape
                scale_expanded = scale_inv.repeat_interleave(block_out, dim=0).repeat_interleave(block_in, dim=1)
                w_bf16 = w_bf16 * scale_expanded.to(torch.bfloat16)
            elif scale_inv.dim() == 0 or (scale_inv.dim() == 1 and scale_inv.shape[0] == 1):
                # Per-tensor scale
                w_bf16 = w_bf16 * scale_inv.to(torch.bfloat16)
            else:
                print("  WARNING: unexpected scale shape %s for %s, using per-tensor" % (scale_inv.shape, key))
                w_bf16 = w_bf16 * scale_inv.flatten()[0].to(torch.bfloat16)

            new_tensors[key] = w_bf16
            print("  Dequantized: %s %s -> bf16" % (key, t.shape))
        else:
            new_tensors[key] = t.to(torch.bfloat16) if t.is_floating_point() else t
            new_weight_map[key] = shard_name

        new_weight_map[key] = shard_name

    # Save dequantized shard
    out_path = os.path.join(OUT_DIR, shard_name)
    save_file(new_tensors, out_path)
    shard_size = os.path.getsize(out_path)
    total_size += shard_size
    print("  Saved %s (%.1f MB)" % (out_path, shard_size / 1e6))

# Write new index (without weight_scale_inv entries)
new_index = {
    "metadata": {"total_size": total_size},
    "weight_map": new_weight_map
}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(new_index, f, indent=2)

# Copy config and tokenizer files (modify config to remove quantization_config)
with open(os.path.join(FP8_DIR, "config.json")) as f:
    config = json.load(f)
config.pop("quantization_config", None)
config["torch_dtype"] = "bfloat16"
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "special_tokens_map.json"]:
    src = os.path.join(FP8_DIR, fname)
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, os.path.join(OUT_DIR, fname))

print("\nDone! Dequantized model saved to %s" % OUT_DIR)
print("Total size: %.1f GB" % (total_size / 1e9))
