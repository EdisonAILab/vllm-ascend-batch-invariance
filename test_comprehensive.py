"""
Comprehensive batch invariance test: 16 prompts, 32 tokens.
Tests with all fixes: matmul padding + add_rms_norm decomposition + per-sequence attention.
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

def load_gsm8k_prompts(n):
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            prompts.append(json.loads(line)["question"])
    return prompts

NUM_PROMPTS = 16
MAX_TOKENS = 32
prompts = load_gsm8k_prompts(NUM_PROMPTS)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

LLM_KWARGS = dict(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=False,
                   enable_chunked_prefill=False)

# Singles
print("Running %d singles..." % NUM_PROMPTS)
llm_s = LLM(**LLM_KWARGS)
t0 = time.time()
singles = []
for i, p in enumerate(prompts):
    out = llm_s.generate([p], sp)
    singles.append(out[0].outputs[0])
t_single = time.time() - t0
print("  Singles done in %.1fs" % t_single)
del llm_s
torch_npu.npu.empty_cache()

# Batch
print("Running batch of %d..." % NUM_PROMPTS)
llm_b = LLM(**LLM_KWARGS)
t0 = time.time()
batched_out = llm_b.generate(prompts, sp)
t_batch = time.time() - t0
batched = [o.outputs[0] for o in batched_out]
print("  Batch done in %.1fs" % t_batch)
del llm_b
torch_npu.npu.empty_cache()

# Compare
print()
print("=" * 70)
print("Results: %d prompts, %d tokens, matmul+rmsnorm+attention fixes" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)
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
        extra = "  (%d/%d tokens differ)" % (n_diff, len(s.token_ids))
    print("  Prompt %2d: tokens=%s  logprob_diff=%.8f  %s%s" % (
        i, "OK  " if tok_ok else "FAIL", max_lp, status, extra))

print()
print("  --- Summary ---")
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp_overall)
print("  Batch-invariant: %s" % (num_tok_fail == 0 and num_lp_fail == 0))
print("  Time: singles=%.1fs  batch=%.1fs  speedup=%.1fx" % (
    t_single, t_batch, t_single / t_batch if t_batch > 0 else 0))
