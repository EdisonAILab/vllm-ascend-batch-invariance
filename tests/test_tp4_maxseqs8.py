"""TP=4 with max_num_seqs=8: test if limiting to 8 concurrent seqs achieves invariance."""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "tp4_maxseqs8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 16]

NUM_PROMPTS = len(prompts)
MAX_TOKENS = 2048
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)
kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_num_seqs=8)

print("=" * 70)
print("TP=4 max_num_seqs=8: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)

# Singles
print("\nRunning singles...")
llm = LLM(**kwargs)
t0 = time.time()
singles = []
for i, p in enumerate(prompts):
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
    print("  Single %d done (%d tokens)" % (i, len(out[0].outputs[0].token_ids)), flush=True)
t_single = time.time() - t0
del llm; torch_npu.npu.empty_cache()

# Batch
print("\nRunning batch...")
llm = LLM(**kwargs)
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
for i in range(NUM_PROMPTS):
    s, b = singles[i], batched[i]
    tok_ok = s.token_ids == b.token_ids
    lp_max = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
            lp_max = max(lp_max, d)
    lp_ok = lp_max == 0.0
    max_lp = max(max_lp, lp_max)
    if not tok_ok: num_tok_fail += 1
    if not lp_ok: num_lp_fail += 1
    n_s = len(s.token_ids)
    total_tokens += n_s
    status = "OK" if tok_ok and lp_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        min_len = min(n_s, len(b.token_ids))
        n_diff = sum(1 for a, bt in zip(s.token_ids[:min_len], b.token_ids[:min_len]) if a != bt)
        extra = "  (%d diffs)" % n_diff
    print("  Prompt %2d: tokens=%s  lp=%.8f  gen=%4d  %s%s" % (
        i, "OK  " if tok_ok else "FAIL", lp_max, n_s, status, extra))
    comparison.append({"prompt_idx": i, "tokens_match": tok_ok,
                      "logprob_max_diff": lp_max, "num_tokens": n_s, "status": status})

summary = {"test": "TP=4 max_num_seqs=8", "model": "Qwen3-4B",
           "tensor_parallel_size": 4, "max_num_seqs": 8,
           "num_prompts": NUM_PROMPTS, "max_tokens": MAX_TOKENS,
           "total_generated_tokens": total_tokens,
           "token_failures": num_tok_fail, "logprob_failures": num_lp_fail,
           "max_logprob_diff": max_lp,
           "batch_invariant": num_tok_fail == 0 and num_lp_fail == 0,
           "time_singles_s": round(t_single, 1), "time_batch_s": round(t_batch, 1),
           "speedup": round(t_single / t_batch, 1) if t_batch > 0 else 0,
           "per_prompt": comparison}

with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print()
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp)
print("  Batch-invariant: %s" % summary["batch_invariant"])
print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
    t_single, t_batch, summary["speedup"]))
