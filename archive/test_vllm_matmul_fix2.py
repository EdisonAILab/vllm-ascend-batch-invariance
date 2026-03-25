"""
Test vLLM batch invariance with matmul fix — using separate LLM per prompt.

Each single prompt gets its own LLM instance to eliminate any KV cache
pollution between prompts.
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

LLM_KWARGS = dict(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=False,
                   enable_chunked_prefill=False)

NUM_PROMPTS = 4
MAX_TOKENS = 8
prompts = load_gsm8k_prompts(NUM_PROMPTS)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

# Singles — SEPARATE LLM instance per prompt
print(f"Running {NUM_PROMPTS} singles (separate LLM each)...")
singles = []
t0 = time.time()
for i, p in enumerate(prompts):
    llm = LLM(**LLM_KWARGS)
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
    del llm
    torch_npu.npu.empty_cache()
    print(f"  Single {i} done", flush=True)
t_single = time.time() - t0

# Batch
print(f"Running batch of {NUM_PROMPTS}...")
llm_b = LLM(**LLM_KWARGS)
t0 = time.time()
batched_out = llm_b.generate(prompts, sp)
t_batch = time.time() - t0
batched = [o.outputs[0] for o in batched_out]
del llm_b
torch_npu.npu.empty_cache()

# Compare
print(f"\n{'=' * 70}")
print(f"Results ({NUM_PROMPTS} prompts, {MAX_TOKENS} tokens)")
print(f"{'=' * 70}")
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
    # Show first few tokens for comparison
    s_toks = list(s.token_ids[:5])
    b_toks = list(b.token_ids[:5])
    print(f"  Prompt {i}: tokens={'OK  ' if tok_ok else 'FAIL'}  "
          f"logprob_diff={max_lp:.8f}  {status}{extra}")
    if not tok_ok:
        print(f"    single_toks={s_toks}  batch_toks={b_toks}")

print(f"\n  Token failures:  {num_tok_fail}/{NUM_PROMPTS}")
print(f"  Logprob failures: {num_lp_fail}/{NUM_PROMPTS}")
print(f"  Max logprob diff: {max_lp_overall:.8f}")
print(f"  Batch-invariant: {num_tok_fail == 0 and num_lp_fail == 0}")
print(f"  Time: singles={t_single:.1f}s  batch={t_batch:.1f}s")
