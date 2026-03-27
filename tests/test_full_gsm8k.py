"""
Experiment #8: Full GSM8K (all 1319 prompts) batch invariance test.
Uses max_tokens=256 to keep runtime reasonable while testing all prompts.
"""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "full_gsm8k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for line in f]

NUM_PROMPTS = len(prompts)
MAX_TOKENS = 256
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)
kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

print("=" * 70)
print("Full GSM8K: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)

# Singles (process in batches of 1 but using same LLM)
print("\nRunning singles...")
llm = LLM(**kwargs)
t0 = time.time()
singles = []
for i, p in enumerate(prompts):
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
    if (i + 1) % 100 == 0:
        print("  %d/%d done (%.0fs)" % (i + 1, NUM_PROMPTS, time.time() - t0), flush=True)
t_single = time.time() - t0
print("  Singles done in %.1fs" % t_single)
del llm; torch_npu.npu.empty_cache()

# Batch
print("\nRunning batch...")
llm = LLM(**kwargs)
t0 = time.time()
batched_out = llm.generate(prompts, sp)
t_batch = time.time() - t0
batched = [o.outputs[0] for o in batched_out]
print("  Batch done in %.1fs" % t_batch)
del llm; torch_npu.npu.empty_cache()

# Compare
num_tok_fail = 0
num_lp_fail = 0
max_lp_overall = 0.0
total_tokens = 0
comparison = []
for i in range(NUM_PROMPTS):
    s, b = singles[i], batched[i]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
            max_lp = max(max_lp, d)
    lp_ok = max_lp == 0.0
    max_lp_overall = max(max_lp_overall, max_lp)
    if not tok_ok: num_tok_fail += 1
    if not lp_ok: num_lp_fail += 1
    total_tokens += len(s.token_ids)
    comparison.append({"prompt_idx": i, "tokens_match": tok_ok,
                      "logprob_max_diff": max_lp, "num_tokens": len(s.token_ids),
                      "status": "OK" if tok_ok and lp_ok else "MISMATCH"})

# Print failures if any
if num_tok_fail > 0 or num_lp_fail > 0:
    print("\nFailed prompts:")
    for c in comparison:
        if c["status"] != "OK":
            print("  Prompt %d: tokens=%s lp=%.8f" % (
                c["prompt_idx"], c["tokens_match"], c["logprob_max_diff"]))

summary = {"test": "Full GSM8K", "model": "Qwen3-4B",
           "num_prompts": NUM_PROMPTS, "max_tokens": MAX_TOKENS,
           "total_generated_tokens": total_tokens,
           "token_failures": num_tok_fail, "logprob_failures": num_lp_fail,
           "max_logprob_diff": max_lp_overall,
           "batch_invariant": num_tok_fail == 0 and num_lp_fail == 0,
           "time_singles_s": round(t_single, 1), "time_batch_s": round(t_batch, 1),
           "speedup": round(t_single / t_batch, 1) if t_batch > 0 else 0,
           "per_prompt": comparison}

with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 70)
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp_overall)
print("  Total tokens:    %d" % total_tokens)
print("  Batch-invariant: %s" % summary["batch_invariant"])
print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
    t_single, t_batch, summary["speedup"]))
