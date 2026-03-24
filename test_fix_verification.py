"""
Verify the lightweight batch-invariance fix for Qwen3-4B on Ascend NPU.
Tests: mode OFF (native, non-invariant), lightweight fix (mm-based), full Triton fix.
"""
import os
import time
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM
from fix_batch_invariance_npu import npu_batch_invariant_mode

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
DEVICE = "npu:0"
SEQ_LEN = 32

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is Berlin. The two cities are",
    "Once upon a time in a land far away, there lived a princess who dreamed of becoming a",
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the",
    "Artificial intelligence has transformed many industries, including healthcare, finance, and",
]


def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def run_test(model, tokenizer, mode_label):
    print(f"\n=== {mode_label} ===")
    ids_list = []
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if len(ids) < SEQ_LEN:
            pad = torch.full((SEQ_LEN - len(ids),), tokenizer.eos_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:SEQ_LEN]
        ids_list.append(ids)

    batch_ids = torch.stack(ids_list).to(DEVICE)

    t0 = time.time()
    logits_batch = get_logits(model, batch_ids)

    all_invariant = True
    max_diff_overall = 0.0
    for i in range(len(PROMPTS)):
        logits_single = get_logits(model, batch_ids[i:i + 1])
        diff = (logits_single[0] - logits_batch[i]).abs().max().item()
        invariant = diff == 0.0
        all_invariant = all_invariant and invariant
        max_diff_overall = max(max_diff_overall, diff)
        print(f"  Prompt {i}: max_diff={diff:.8f}  {'OK (invariant)' if invariant else 'MISMATCH'}")

    elapsed = time.time() - t0
    print(f"  --> Batch-invariant: {all_invariant}  (overall max_diff={max_diff_overall:.8f})  time={elapsed:.1f}s")
    return all_invariant, max_diff_overall


def main():
    print(f"Loading {MODEL_PATH} on {DEVICE} ...")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'not set')}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"dtype=bfloat16, seq_len={SEQ_LEN}, batch_size={len(PROMPTS)}")

    # Test 1: Native (no fix)
    run_test(model, tokenizer, "Mode OFF: Native NPU ops (no fix)")

    # Test 2: Lightweight mm-based fix
    with npu_batch_invariant_mode():
        run_test(model, tokenizer, "Lightweight Fix: aten::linear patched to use torch.mm")

    # Test 3: Also verify individual k_proj/v_proj with the fix
    print("\n=== Per-operator verification with lightweight fix ===")
    with torch.no_grad():
        ids_list = []
        for p in PROMPTS:
            ids = tokenizer(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            if len(ids) < SEQ_LEN:
                pad = torch.full((SEQ_LEN - len(ids),), tokenizer.eos_token_id, dtype=ids.dtype)
                ids = torch.cat([ids, pad])
            else:
                ids = ids[:SEQ_LEN]
            ids_list.append(ids)
        batch_ids = torch.stack(ids_list).to(DEVICE)
        batch_embeds = model.model.embed_tokens(batch_ids)
        single_embeds = [model.model.embed_tokens(batch_ids[i:i + 1]) for i in range(4)]

        layer = model.model.layers[0]
        normed_batch = layer.input_layernorm(batch_embeds)
        normed_singles = [layer.input_layernorm(single_embeds[i]) for i in range(4)]

        # Without fix
        for proj_name in ["k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            out_batch = proj(normed_batch)
            diffs = []
            for i in range(4):
                out_single = proj(normed_singles[i])
                diff = (out_single[0] - out_batch[i]).abs().max().item()
                diffs.append(diff)
            print(f"  {proj_name} (no fix):    max_diff={max(diffs):.8f}  {'OK' if max(diffs) == 0 else 'MISMATCH'}")

        # With fix
        with npu_batch_invariant_mode():
            for proj_name in ["k_proj", "v_proj"]:
                proj = getattr(layer.self_attn, proj_name)
                out_batch = proj(normed_batch)
                diffs = []
                for i in range(4):
                    out_single = proj(normed_singles[i])
                    diff = (out_single[0] - out_batch[i]).abs().max().item()
                    diffs.append(diff)
                print(f"  {proj_name} (mm fix):    max_diff={max(diffs):.8f}  {'OK' if max(diffs) == 0 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
