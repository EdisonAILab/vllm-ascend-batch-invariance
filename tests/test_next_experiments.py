"""
Run multiple experiments from NEXT_EXPERIMENTS.md:
1. Larger batch (64 prompts)
2. Non-greedy sampling (temp=0.6, seed=42)
3. Prefix caching enabled
4. Chunked prefill enabled
All with TP=1, max_tokens=2048, operator-level fixes.
"""
import json, os, sys, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_gsm8k_prompts(n):
    with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
        return [json.loads(line)["question"] for i, line in enumerate(f) if i < n]

def run_experiment(name, prompts, sp, llm_kwargs, output_dir):
    """Run single-vs-batch comparison and save results."""
    os.makedirs(output_dir, exist_ok=True)
    num = len(prompts)
    print("\n" + "=" * 70)
    print("Experiment: %s (%d prompts, max_tokens=%d)" % (name, num, sp.max_tokens))
    print("=" * 70)

    # Singles
    print("  Running singles...")
    llm = LLM(**llm_kwargs)
    t0 = time.time()
    singles = []
    for i, p in enumerate(prompts):
        out = llm.generate([p], sp)
        singles.append(out[0].outputs[0])
        if (i + 1) % 10 == 0 or i == num - 1:
            print("    %d/%d done" % (i + 1, num), flush=True)
    t_single = time.time() - t0
    del llm; torch_npu.npu.empty_cache()

    # Batch
    print("  Running batch...")
    llm = LLM(**llm_kwargs)
    t0 = time.time()
    batched_out = llm.generate(prompts, sp)
    t_batch = time.time() - t0
    batched = [o.outputs[0] for o in batched_out]
    del llm; torch_npu.npu.empty_cache()

    # Compare
    num_tok_fail = 0
    num_lp_fail = 0
    max_lp = 0.0
    total_tokens = 0
    comparison = []
    for i in range(num):
        s, b = singles[i], batched[i]
        tok_ok = s.token_ids == b.token_ids
        lp_max = 0.0
        for ts, tb in zip(s.logprobs or [], b.logprobs or []):
            if ts and tb:
                d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
                lp_max = max(lp_max, d)
        lp_ok = lp_max == 0.0
        max_lp = max(max_lp, lp_max)
        if not tok_ok: num_tok_fail += 1
        if not lp_ok: num_lp_fail += 1
        total_tokens += len(s.token_ids)
        comparison.append({"prompt_idx": i, "tokens_match": tok_ok,
                          "logprob_max_diff": lp_max, "num_tokens": len(s.token_ids),
                          "status": "OK" if tok_ok and lp_ok else "MISMATCH"})

    summary = {
        "experiment": name, "model": "Qwen3-4B", "num_prompts": num,
        "max_tokens": sp.max_tokens, "temperature": sp.temperature,
        "total_generated_tokens": total_tokens,
        "token_failures": num_tok_fail, "logprob_failures": num_lp_fail,
        "max_logprob_diff": max_lp,
        "batch_invariant": num_tok_fail == 0 and num_lp_fail == 0,
        "time_singles_s": round(t_single, 1), "time_batch_s": round(t_batch, 1),
        "speedup": round(t_single / t_batch, 1) if t_batch > 0 else 0,
        "per_prompt": comparison,
    }
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tag = "PASS" if summary["batch_invariant"] else "FAIL"
    print("  Result: %s  tok_fail=%d/%d  lp_fail=%d/%d  max_lp=%.8f" % (
        tag, num_tok_fail, num, num_lp_fail, num, max_lp))
    print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
        t_single, t_batch, summary["speedup"]))
    return summary

BASE_KWARGS = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9)

results = {}

# Experiment 1: Larger batch (64 prompts, 256 tokens to keep time reasonable)
prompts_64 = load_gsm8k_prompts(64)
sp1 = SamplingParams(temperature=0.0, max_tokens=256, logprobs=1)
kwargs1 = dict(**BASE_KWARGS, enable_prefix_caching=False, enable_chunked_prefill=False)
results["batch64"] = run_experiment("Batch-64 (256 tokens)", prompts_64, sp1, kwargs1,
                                     os.path.join(PROJECT_DIR, "results", "batch64"))

# Experiment 2: Non-greedy sampling with fixed seed
prompts_16 = load_gsm8k_prompts(16)
sp2 = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, logprobs=1, seed=42)
kwargs2 = dict(**BASE_KWARGS, enable_prefix_caching=False, enable_chunked_prefill=False)
results["sampling"] = run_experiment("Sampling (temp=0.6, seed=42)", prompts_16, sp2, kwargs2,
                                      os.path.join(PROJECT_DIR, "results", "sampling_temp06"))

# Experiment 3: Prefix caching enabled
sp3 = SamplingParams(temperature=0.0, max_tokens=256, logprobs=1)
kwargs3 = dict(**BASE_KWARGS, enable_prefix_caching=True, enable_chunked_prefill=False)
results["prefix_cache"] = run_experiment("Prefix caching", prompts_16, sp3, kwargs3,
                                          os.path.join(PROJECT_DIR, "results", "prefix_caching"))

# Experiment 4: Chunked prefill enabled
sp4 = SamplingParams(temperature=0.0, max_tokens=256, logprobs=1)
kwargs4 = dict(**BASE_KWARGS, enable_prefix_caching=False, enable_chunked_prefill=True)
results["chunked_prefill"] = run_experiment("Chunked prefill", prompts_16, sp4, kwargs4,
                                             os.path.join(PROJECT_DIR, "results", "chunked_prefill"))

# Summary
print("\n" + "=" * 70)
print("ALL EXPERIMENTS SUMMARY")
print("=" * 70)
for name, r in results.items():
    tag = "PASS" if r["batch_invariant"] else "FAIL"
    print("  %-20s: %s  tok=%d/%d  lp=%d/%d  max_lp=%.8f" % (
        name, tag, r["token_failures"], r["num_prompts"],
        r["logprob_failures"], r["num_prompts"], r["max_logprob_diff"]))
