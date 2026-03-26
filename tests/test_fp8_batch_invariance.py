"""
FP8 quantized Qwen3-4B batch invariance test on Ascend NPU.
Tests with quantization="fp8" to check if FP8 matmul introduces
new batch-dependent behavior.
"""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B-FP8-dequant"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "fp8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 16]

NUM_PROMPTS = len(prompts)
MAX_TOKENS = 2048
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              )

def output_to_dict(output, idx, prompt):
    o = output.outputs[0]
    lps = []
    for lp in o.logprobs:
        if lp:
            top = next(iter(lp.values()))
            lps.append({"logprob": top.logprob})
        else:
            lps.append(None)
    return {"prompt_idx": idx, "prompt": prompt, "token_ids": list(o.token_ids),
            "text": o.text, "num_tokens": len(o.token_ids),
            "finish_reason": o.finish_reason, "logprobs": lps}

print("=" * 70)
print("FP8 Batch Invariance Test: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("quantization=fp8, TP=1")
print("=" * 70)

# Singles
print("\nRunning singles (FP8)...")
llm = LLM(**kwargs)
t0 = time.time()
singles = []
singles_data = []
for i, p in enumerate(prompts):
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
    singles_data.append(output_to_dict(out[0], i, p))
    print("  Single %d done (%d tokens)" % (i, len(out[0].outputs[0].token_ids)), flush=True)
t_single = time.time() - t0
del llm; torch_npu.npu.empty_cache()

with open(os.path.join(OUTPUT_DIR, "singles_responses.json"), "w") as f:
    json.dump(singles_data, f, indent=2, ensure_ascii=False)

# Batch
print("\nRunning batch (FP8)...")
llm = LLM(**kwargs)
t0 = time.time()
batched_out = llm.generate(prompts, sp)
t_batch = time.time() - t0
batched = [o.outputs[0] for o in batched_out]
batched_data = [output_to_dict(o, i, prompts[i]) for i, o in enumerate(batched_out)]
del llm; torch_npu.npu.empty_cache()

with open(os.path.join(OUTPUT_DIR, "batch_responses.json"), "w") as f:
    json.dump(batched_data, f, indent=2, ensure_ascii=False)

# Compare
print()
print("=" * 70)
print("FP8 Results: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)
comparison = []
num_tok_fail = 0
num_lp_fail = 0
max_lp_overall = 0.0
total_tokens = 0
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
    n_s = len(s.token_ids)
    total_tokens += n_s
    status = "OK" if tok_ok and lp_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        min_len = min(n_s, len(b.token_ids))
        n_diff = sum(1 for a, bt in zip(s.token_ids[:min_len], b.token_ids[:min_len]) if a != bt)
        if n_s != len(b.token_ids):
            extra = "  (len %d vs %d)" % (n_s, len(b.token_ids))
        else:
            extra = "  (%d/%d differ)" % (n_diff, n_s)
    print("  Prompt %2d: tokens=%s  lp_diff=%.8f  gen=%4d  %s%s" % (
        i, "OK  " if tok_ok else "FAIL", max_lp, n_s, status, extra))
    comparison.append({"prompt_idx": i, "tokens_match": tok_ok,
                       "logprob_max_diff": max_lp, "num_tokens": n_s, "status": status})

summary = {"test": "FP8 batch invariance", "model": "Qwen3-4B", "quantization": "fp8",
           "tensor_parallel_size": 1, "num_prompts": NUM_PROMPTS, "max_tokens": MAX_TOKENS,
           "total_generated_tokens": total_tokens, "token_failures": num_tok_fail,
           "logprob_failures": num_lp_fail, "max_logprob_diff": max_lp_overall,
           "batch_invariant": num_tok_fail == 0 and num_lp_fail == 0,
           "time_singles_s": round(t_single, 1), "time_batch_s": round(t_batch, 1),
           "speedup": round(t_single / t_batch, 1) if t_batch > 0 else 0,
           "per_prompt": comparison}

with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print()
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp_overall)
print("  Total tokens:    %d" % total_tokens)
print("  Batch-invariant: %s" % summary["batch_invariant"])
print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
    t_single, t_batch, summary["speedup"]))
