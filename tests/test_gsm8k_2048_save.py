"""
GSM8K batch invariance test with max_tokens=2048.
Saves all generated responses (single + batch) to JSON for reproducibility.
"""
import json
import os
import sys
import time

os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch_npu
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_gsm8k_prompts(n):
    jsonl_path = os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            prompts.append(json.loads(line)["question"])
    return prompts

def output_to_dict(output, prompt_idx, prompt_text):
    """Convert vLLM output to serializable dict."""
    o = output.outputs[0]
    logprobs_list = []
    for lp in o.logprobs:
        if lp:
            top = next(iter(lp.values()))
            logprobs_list.append({
                "token_id": top.decoded_token if hasattr(top, "decoded_token") else str(list(lp.keys())[0]),
                "logprob": top.logprob
            })
        else:
            logprobs_list.append(None)
    return {
        "prompt_idx": prompt_idx,
        "prompt": prompt_text,
        "token_ids": list(o.token_ids),
        "text": o.text,
        "num_tokens": len(o.token_ids),
        "finish_reason": o.finish_reason,
        "logprobs": logprobs_list,
    }

NUM_PROMPTS = 16
MAX_TOKENS = 2048
prompts = load_gsm8k_prompts(NUM_PROMPTS)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

LLM_KWARGS = dict(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=False,
                   enable_chunked_prefill=False)

# Singles
print("Running %d singles (max_tokens=%d)..." % (NUM_PROMPTS, MAX_TOKENS))
llm_s = LLM(**LLM_KWARGS)
t0 = time.time()
singles_raw = []
singles_data = []
for i, p in enumerate(prompts):
    out = llm_s.generate([p], sp)
    singles_raw.append(out[0])
    singles_data.append(output_to_dict(out[0], i, p))
    n_toks = len(out[0].outputs[0].token_ids)
    print("  Single %d done (%d tokens)" % (i, n_toks), flush=True)
t_single = time.time() - t0
print("  Singles done in %.1fs" % t_single)
del llm_s
torch_npu.npu.empty_cache()

# Save singles
with open(os.path.join(OUTPUT_DIR, "singles_responses.json"), "w") as f:
    json.dump(singles_data, f, indent=2, ensure_ascii=False)
print("  Saved singles to results/singles_responses.json")

# Batch
print("Running batch of %d (max_tokens=%d)..." % (NUM_PROMPTS, MAX_TOKENS))
llm_b = LLM(**LLM_KWARGS)
t0 = time.time()
batched_out = llm_b.generate(prompts, sp)
t_batch = time.time() - t0
batched_data = [output_to_dict(o, i, prompts[i]) for i, o in enumerate(batched_out)]
print("  Batch done in %.1fs" % t_batch)
del llm_b
torch_npu.npu.empty_cache()

# Save batch
with open(os.path.join(OUTPUT_DIR, "batch_responses.json"), "w") as f:
    json.dump(batched_data, f, indent=2, ensure_ascii=False)
print("  Saved batch to results/batch_responses.json")

# Compare
print()
print("=" * 70)
print("GSM8K Results: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)
comparison = []
num_tok_fail = 0
num_lp_fail = 0
max_lp_overall = 0.0
total_tokens = 0
for i in range(NUM_PROMPTS):
    s = singles_raw[i].outputs[0]
    b = batched_out[i].outputs[0]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    first_diff_pos = -1
    for j, (ts, tb) in enumerate(zip(s.logprobs, b.logprobs)):
        if ts and tb:
            lp_s = next(iter(ts.values())).logprob
            lp_b = next(iter(tb.values())).logprob
            d = abs(lp_s - lp_b)
            if d > max_lp:
                max_lp = d
            if d > 0 and first_diff_pos < 0:
                first_diff_pos = j
    lp_ok = max_lp == 0.0
    max_lp_overall = max(max_lp_overall, max_lp)
    if not tok_ok:
        num_tok_fail += 1
    if not lp_ok:
        num_lp_fail += 1
    status = "OK" if (tok_ok and lp_ok) else "MISMATCH"
    n_s = len(s.token_ids)
    total_tokens += n_s
    extra = ""
    if not tok_ok:
        min_len = min(n_s, len(b.token_ids))
        n_diff = sum(1 for a, bt in zip(s.token_ids[:min_len], b.token_ids[:min_len]) if a != bt)
        extra = "  (%d tokens differ)" % n_diff
    result_line = "Prompt %2d: tokens=%s  logprob_diff=%.8f  gen=%4d toks  %s%s" % (
        i, "OK  " if tok_ok else "FAIL", max_lp, n_s, status, extra)
    print("  " + result_line)
    comparison.append({
        "prompt_idx": i,
        "tokens_match": tok_ok,
        "logprob_max_diff": max_lp,
        "num_tokens": n_s,
        "status": status,
    })

summary = {
    "test": "GSM8K batch invariance",
    "model": "Qwen3-4B",
    "num_prompts": NUM_PROMPTS,
    "max_tokens": MAX_TOKENS,
    "total_generated_tokens": total_tokens,
    "token_failures": num_tok_fail,
    "logprob_failures": num_lp_fail,
    "max_logprob_diff": max_lp_overall,
    "batch_invariant": num_tok_fail == 0 and num_lp_fail == 0,
    "time_singles_s": round(t_single, 1),
    "time_batch_s": round(t_batch, 1),
    "speedup": round(t_single / t_batch, 1) if t_batch > 0 else 0,
    "per_prompt": comparison,
}

with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print()
print("  --- Summary ---")
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp_overall)
print("  Total tokens generated: %d" % total_tokens)
print("  Batch-invariant: %s" % summary["batch_invariant"])
print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
    t_single, t_batch, summary["speedup"]))
print("  Results saved to: %s" % OUTPUT_DIR)
