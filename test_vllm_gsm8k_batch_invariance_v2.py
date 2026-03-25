"""
vLLM batch invariance test using GSM8K dataset on Qwen3-4B / Ascend NPU.

Compares generated tokens and logprobs between single-prompt and batched runs.
Also disables prefix caching and tests with enforce_eager=True.

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python3 test_vllm_gsm8k_batch_invariance_v2.py [--num_prompts 16] [--max_tokens 32]
"""
import argparse
import json
import os
import sys
import time

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch_npu
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"


def load_gsm8k_prompts(num_prompts):
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


def run_test(llm, prompts, max_tokens, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  prompts={len(prompts)}, max_tokens={max_tokens}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)

    # Singles
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

    # Batch
    print("  Running batch...")
    sys.stdout.flush()
    t0 = time.time()
    batched = llm.generate(prompts, sp)
    t_batch = time.time() - t0

    # Compare
    num_tok_fail = 0
    num_lp_fail = 0
    max_lp_overall = 0.0

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
        max_lp_overall = max(max_lp_overall, max_lp)
        if not tok_ok:
            num_tok_fail += 1
        if not lp_ok:
            num_lp_fail += 1

        status = "OK" if (tok_ok and lp_ok) else "MISMATCH"
        extra = ""
        if not tok_ok:
            n_diff = sum(1 for a, bt in zip(s.token_ids, b.token_ids) if a != bt)
            extra = f"  ({n_diff}/{len(s.token_ids)} tokens differ)"
        print(f"  Prompt {i:3d}: tokens={'OK  ' if tok_ok else 'FAIL'}  logprob_diff={max_lp:.8f}  {status}{extra}")

    all_ok = (num_tok_fail == 0 and num_lp_fail == 0)
    print(f"\n  --- Summary ---")
    print(f"  Token failures:  {num_tok_fail}/{len(prompts)}")
    print(f"  Logprob failures: {num_lp_fail}/{len(prompts)}")
    print(f"  Max logprob diff: {max_lp_overall:.8f}")
    print(f"  Batch-invariant: {all_ok}")
    print(f"  Time: singles={t_single:.1f}s  batch={t_batch:.1f}s")
    sys.stdout.flush()
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"Model: {MODEL_PATH}")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    sys.stdout.flush()

    prompts = load_gsm8k_prompts(args.num_prompts)

    results = {}

    # --- Test 1: Native, default settings ---
    print("\n\n>>> Loading vLLM (Native, default settings)...")
    sys.stdout.flush()
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9)
    results["Native (default)"] = run_test(llm, prompts, args.max_tokens,
                                           "Mode 1: Native, default settings")
    del llm
    torch_npu.npu.empty_cache()

    # --- Test 2: Native, prefix caching OFF ---
    print("\n\n>>> Loading vLLM (Native, prefix_caching=False)...")
    sys.stdout.flush()
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False)
    results["Native (no prefix cache)"] = run_test(llm, prompts, args.max_tokens,
                                                    "Mode 2: Native, prefix_caching=False")
    del llm
    torch_npu.npu.empty_cache()

    # --- Test 3: Lightweight fix, prefix caching OFF ---
    print("\n\n>>> Loading vLLM (Lightweight fix, prefix_caching=False)...")
    sys.stdout.flush()
    from fix_batch_invariance_npu import enable_npu_batch_invariant_linear, disable_npu_batch_invariant_linear
    enable_npu_batch_invariant_linear()
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False)
    results["Fix + no prefix cache"] = run_test(llm, prompts, args.max_tokens,
                                                 "Mode 3: Lightweight fix + prefix_caching=False")
    del llm
    disable_npu_batch_invariant_linear()
    torch_npu.npu.empty_cache()

    # --- Final ---
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    for name, ok in results.items():
        print(f"  {name:30s}: {'PASS' if ok else 'FAIL'}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
