"""
vLLM batch invariance test v4 — attention fix (on-disk) + linear fix (runtime).

Tests:
1. Default (attention fix only, from on-disk patch)
2. Attention fix + aten::linear per-sample fix
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
            prompts.append(json.loads(line)["question"])
    print(f"Loaded {len(prompts)} GSM8K test prompts")
    return prompts


def run_singles(prompts, max_tokens, **llm_kwargs):
    print("\n  Loading LLM for singles...", flush=True)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9, **llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    t0 = time.time()
    results = []
    for i, p in enumerate(prompts):
        out = llm.generate([p], sp)
        results.append(out[0].outputs[0])
    t_single = time.time() - t0
    print(f"  Singles done: {t_single:.1f}s", flush=True)
    del llm
    torch_npu.npu.empty_cache()
    return results, t_single


def run_batch(prompts, max_tokens, **llm_kwargs):
    print("\n  Loading LLM for batch...", flush=True)
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9, **llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    t0 = time.time()
    out = llm.generate(prompts, sp)
    t_batch = time.time() - t0
    print(f"  Batch done: {t_batch:.1f}s", flush=True)
    results = [o.outputs[0] for o in out]
    del llm
    torch_npu.npu.empty_cache()
    return results, t_batch


def compare(singles, batched, prompts):
    num_tok_fail = 0
    num_lp_fail = 0
    max_lp_overall = 0.0
    for i, (s, b) in enumerate(zip(singles, batched)):
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
    sys.stdout.flush()
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    prompts = load_gsm8k_prompts(args.num_prompts)
    results = {}

    # --- Test 1: Attention fix only (on-disk patch) ---
    print(f"\n{'=' * 70}")
    print(f"  Test 1: Attention fix only (on-disk)")
    print(f"{'=' * 70}")
    singles1, t_s1 = run_singles(prompts, args.max_tokens)
    batched1, t_b1 = run_batch(prompts, args.max_tokens)
    print(f"  Time: singles={t_s1:.1f}s  batch={t_b1:.1f}s")
    results["Attention fix only"] = compare(singles1, batched1, prompts)

    # --- Test 2: Attention fix + aten::linear fix ---
    print(f"\n{'=' * 70}")
    print(f"  Test 2: Attention fix + aten::linear per-sample fix")
    print(f"{'=' * 70}")
    from fix_batch_invariance_npu import enable_npu_batch_invariant_linear
    enable_npu_batch_invariant_linear()
    print("  [aten::linear fix enabled]")
    singles2, t_s2 = run_singles(prompts, args.max_tokens)
    batched2, t_b2 = run_batch(prompts, args.max_tokens)
    print(f"  Time: singles={t_s2:.1f}s  batch={t_b2:.1f}s")
    results["Attention + Linear fix"] = compare(singles2, batched2, prompts)

    # --- Final ---
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    for name, ok in results.items():
        print(f"  {name:40s}: {'PASS' if ok else 'FAIL'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
