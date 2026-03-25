"""
Batch invariance test using GSM8K dataset on Qwen3-4B / Ascend NPU.

Loads GSM8K test questions, tokenizes them, and compares logits from
batched vs single forward passes under:
  1. Native NPU ops (no fix)
  2. Lightweight per-sample fix (aten::linear -> per-sample torch.mm)
  3. (Optional) batch_invariant_ops Triton persistent kernel

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python3 test_gsm8k_batch_invariance.py [--num_prompts 16] [--batch_size 8] [--triton]
"""
import argparse
import os
import time
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
DEVICE = "npu:0"


def load_gsm8k_prompts(num_prompts):
    """Load GSM8K test set questions from local JSONL file."""
    import json
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_prompts:
                break
            item = json.loads(line)
            prompts.append(item["question"])
    print(f"Loaded {len(prompts)} GSM8K test prompts from {jsonl_path}")
    return prompts


def tokenize_prompts(prompts, tokenizer, max_len):
    """Tokenize and pad/truncate all prompts to the same length."""
    ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if len(ids) < max_len:
            pad = torch.full((max_len - len(ids),), tokenizer.eos_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:max_len]
        ids_list.append(ids)
    return ids_list


def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def run_batch_invariance_test(model, ids_list, batch_size, mode_label):
    """Test batch invariance: compare batched logits vs single-item logits."""
    print(f"\n{'=' * 70}")
    print(f"=== {mode_label} ===")
    print(f"{'=' * 70}")

    num_prompts = len(ids_list)
    all_diffs = []
    num_mismatches = 0
    total_tested = 0

    t0 = time.time()

    # Process in batches
    for batch_start in range(0, num_prompts, batch_size):
        batch_end = min(batch_start + batch_size, num_prompts)
        batch_ids = torch.stack(ids_list[batch_start:batch_end]).to(DEVICE)
        actual_bs = batch_ids.shape[0]

        if actual_bs <= 1:
            continue

        logits_batch = get_logits(model, batch_ids)

        for i in range(actual_bs):
            logits_single = get_logits(model, batch_ids[i:i + 1])
            diff = (logits_single[0] - logits_batch[i]).abs().max().item()
            all_diffs.append(diff)
            total_tested += 1
            if diff > 0:
                num_mismatches += 1

            status = "OK" if diff == 0.0 else "MISMATCH"
            print(f"  Prompt {batch_start + i}: max_diff={diff:.8f}  {status}")

    elapsed = time.time() - t0
    overall_max = max(all_diffs) if all_diffs else 0.0
    avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0.0
    invariant = num_mismatches == 0

    print(f"\n  --- Summary ---")
    print(f"  Total prompts tested: {total_tested}")
    print(f"  Mismatches: {num_mismatches}/{total_tested}")
    print(f"  Max diff: {overall_max:.8f}")
    print(f"  Avg diff: {avg_diff:.8f}")
    print(f"  Batch-invariant: {invariant}")
    print(f"  Time: {elapsed:.1f}s")

    return invariant, overall_max, num_mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=16, help="Number of GSM8K prompts to test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batched forward pass")
    parser.add_argument("--max_len", type=int, default=64, help="Max token length (pad/truncate)")
    parser.add_argument("--triton", action="store_true", help="Also test Triton persistent kernel")
    args = parser.parse_args()

    print(f"Loading {MODEL_PATH} on {DEVICE} ...")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'not set')}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"dtype=bfloat16, max_len={args.max_len}, num_prompts={args.num_prompts}, batch_size={args.batch_size}")
    print(f"hidden_size={model.config.hidden_size}, num_kv_heads={model.config.num_key_value_heads}")

    # Load GSM8K
    prompts = load_gsm8k_prompts(args.num_prompts)
    ids_list = tokenize_prompts(prompts, tokenizer, args.max_len)
    print(f"Tokenized {len(ids_list)} prompts to length {args.max_len}")

    # Mode 1: Native
    run_batch_invariance_test(model, ids_list, args.batch_size,
                              "Mode 1: Native NPU ops (no fix)")

    # Mode 2: Lightweight fix
    from fix_batch_invariance_npu import npu_batch_invariant_mode
    with npu_batch_invariant_mode():
        run_batch_invariance_test(model, ids_list, args.batch_size,
                                  "Mode 2: Lightweight per-sample fix")

    # Mode 3: Triton (optional)
    if args.triton:
        from batch_invariant_ops import set_batch_invariant_mode
        with set_batch_invariant_mode(True):
            run_batch_invariance_test(model, ids_list, args.batch_size,
                                      "Mode 3: batch_invariant_ops Triton persistent kernel")


if __name__ == "__main__":
    main()
