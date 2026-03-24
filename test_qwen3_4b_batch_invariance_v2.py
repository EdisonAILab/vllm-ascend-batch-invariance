"""
Batch invariance test for Qwen3-4B on Ascend NPU.
Tests three modes:
  1. Native NPU ops (no fix) — shows non-invariance
  2. Lightweight per-sample fix (patches aten::linear) — achieves bit-exact invariance
  3. batch_invariant_ops Triton kernel (for comparison)
"""
import os
import time
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM
from fix_batch_invariance_npu import npu_batch_invariant_mode
from batch_invariant_ops import set_batch_invariant_mode

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
    print(f"hidden_size={model.config.hidden_size}, num_kv_heads={model.config.num_key_value_heads}")

    # Mode 1: Native (no fix)
    run_test(model, tokenizer, "Mode 1: Native NPU ops (no fix)")

    # Mode 2: Lightweight per-sample fix
    with npu_batch_invariant_mode():
        run_test(model, tokenizer, "Mode 2: Lightweight per-sample fix (aten::linear -> per-sample torch.mm)")

    # Mode 3: batch_invariant_ops Triton kernel
    with set_batch_invariant_mode(True):
        run_test(model, tokenizer, "Mode 3: batch_invariant_ops Triton persistent kernel")


if __name__ == "__main__":
    main()
