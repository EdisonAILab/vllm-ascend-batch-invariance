"""
vLLM batch invariance test using GSM8K dataset on Qwen3-4B / Ascend NPU.

Compares generated tokens and logprobs between single-prompt and batched runs.
Tests:
  1. Native NPU ops (no fix)
  2. Lightweight per-sample fix (aten::linear -> per-sample torch.mm)
  3. (Optional) batch_invariant_ops Triton persistent kernel

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python3 test_vllm_gsm8k_batch_invariance.py [--num_prompts 16] [--max_tokens 32] [--triton]
"""
import argparse
import json
import os
import sys
import time

# Must set before importing torch_npu/vllm
if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch_npu
from vllm import LLM, SamplingParams


MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"


def load_gsm8k_prompts(num_prompts):
    """Load GSM8K test set questions from local JSONL file."""
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_prompts:
                break
            item = json.loads(line)
            prompts.append(item["question"])
    print(f"Loaded {len(prompts)} GSM8K test prompts")
    return prompts


def run_batch_invariance_test(llm, prompts, max_tokens, label):
    """Compare single-prompt runs vs batched run."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  prompts={len(prompts)}, max_tokens={max_tokens}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)

    # --- Singles ---
    print("\n  Running singles...")
    sys.stdout.flush()
    t0 = time.time()
    singles = []
    for i, p in enumerate(prompts):
        out = llm.generate([p], sp)
        singles.append(out[0].outputs[0])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Single {i} done ({time.time() - t0:.1f}s)")
            sys.stdout.flush()
    t_single = time.time() - t0
    print(f"  Singles done: {t_single:.1f}s")
    sys.stdout.flush()

    # --- Batch ---
    print("  Running batch...")
    sys.stdout.flush()
    t0 = time.time()
    batched = llm.generate(prompts, sp)
    t_batch = time.time() - t0
    print(f"  Batch done: {t_batch:.1f}s")
    sys.stdout.flush()

    # --- Compare ---
    all_tok_ok = True
    all_lp_ok = True
    num_tok_fail = 0
    num_lp_fail = 0
    max_lp_diff_overall = 0.0

    for i, (s, b_out) in enumerate(zip(singles, batched)):
        b = b_out.outputs[0]
        tok_ok = s.token_ids == b.token_ids

        max_lp = 0.0
        for ts, tb in zip(s.logprobs, b.logprobs):
            if ts and tb:
                lp_s = next(iter(ts.values())).logprob
                lp_b = next(iter(tb.values())).logprob
                max_lp = max(max_lp, abs(lp_s - lp_b))
        lp_ok = max_lp == 0.0

        all_tok_ok = all_tok_ok and tok_ok
        all_lp_ok = all_lp_ok and lp_ok
        max_lp_diff_overall = max(max_lp_diff_overall, max_lp)

        if not tok_ok:
            num_tok_fail += 1
        if not lp_ok:
            num_lp_fail += 1

        status = "OK" if (tok_ok and lp_ok) else "MISMATCH"
        print(f"  Prompt {i:3d}: tokens={'OK  ' if tok_ok else 'FAIL'}  logprob_diff={max_lp:.8f}  {status}")

        if not tok_ok:
            n_diff = sum(1 for a, b_tok in zip(s.token_ids, b.token_ids) if a != b_tok)
            print(f"             {n_diff}/{len(s.token_ids)} tokens differ")

    print(f"\n  --- Summary ---")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Token failures: {num_tok_fail}/{len(prompts)}")
    print(f"  Logprob failures: {num_lp_fail}/{len(prompts)}")
    print(f"  Max logprob diff: {max_lp_diff_overall:.8f}")
    print(f"  Tokens invariant: {all_tok_ok}")
    print(f"  Logprobs invariant: {all_lp_ok}")
    print(f"  Time: singles={t_single:.1f}s  batch={t_batch:.1f}s")
    sys.stdout.flush()
    return all_tok_ok and all_lp_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=16, help="Number of GSM8K prompts")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max generated tokens per prompt")
    parser.add_argument("--triton", action="store_true", help="Also test Triton persistent kernel")
    args = parser.parse_args()

    print(f"Model: {MODEL_PATH}")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    print(f"num_prompts={args.num_prompts}, max_tokens={args.max_tokens}")
    sys.stdout.flush()

    prompts = load_gsm8k_prompts(args.num_prompts)

    # --- Mode 1: Native ---
    print("\n\nLoading vLLM (Native, TP=1)...")
    sys.stdout.flush()
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9)
    native_ok = run_batch_invariance_test(llm, prompts, args.max_tokens,
                                          "Mode 1: Native NPU ops (no fix)")
    del llm
    torch_npu.npu.empty_cache()

    # --- Mode 2: Lightweight fix ---
    print("\n\nLoading vLLM (Lightweight fix, TP=1)...")
    sys.stdout.flush()
    from fix_batch_invariance_npu import enable_npu_batch_invariant_linear, disable_npu_batch_invariant_linear
    enable_npu_batch_invariant_linear()
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9)
    fix_ok = run_batch_invariance_test(llm, prompts, args.max_tokens,
                                       "Mode 2: Lightweight per-sample fix")
    del llm
    disable_npu_batch_invariant_linear()
    torch_npu.npu.empty_cache()

    # --- Mode 3: Triton (optional) ---
    triton_ok = None
    if args.triton:
        print("\n\nLoading vLLM (Triton, TP=1)...")
        print("NOTE: Triton JIT compilation will be slow on first run")
        sys.stdout.flush()
        from batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode
        enable_batch_invariant_mode()
        llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9)
        triton_ok = run_batch_invariance_test(llm, prompts, args.max_tokens,
                                              "Mode 3: Triton persistent kernel")
        del llm
        disable_batch_invariant_mode()
        torch_npu.npu.empty_cache()

    # --- Final ---
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"  Native:          {'PASS' if native_ok else 'FAIL'}")
    print(f"  Lightweight fix:  {'PASS' if fix_ok else 'FAIL'}")
    if triton_ok is not None:
        print(f"  Triton:          {'PASS' if triton_ok else 'FAIL'}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
