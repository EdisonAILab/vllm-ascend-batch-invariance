"""
Test batch invariance with max_num_seqs=1 — forces vLLM to process
one sequence at a time, ensuring same M dimension in matmul.
"""
import json
import os
import sys
import time

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

prompts = load_gsm8k_prompts(4)
sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=1)

# Test with max_num_seqs=1 — force single-sequence processing
print("Loading LLM with max_num_seqs=1 for singles...")
llm_s = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
            tensor_parallel_size=1, max_model_len=4096,
            gpu_memory_utilization=0.9, max_num_seqs=1)
singles = []
for p in prompts:
    out = llm_s.generate([p], sp)
    singles.append(out[0].outputs[0])
del llm_s
torch_npu.npu.empty_cache()

print("\nLoading LLM with max_num_seqs=1 for batch...")
llm_b = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
            tensor_parallel_size=1, max_model_len=4096,
            gpu_memory_utilization=0.9, max_num_seqs=1)
batched_out = llm_b.generate(prompts, sp)
batched = [o.outputs[0] for o in batched_out]
del llm_b
torch_npu.npu.empty_cache()

# Compare
print("\n" + "=" * 70)
print("Results: max_num_seqs=1")
print("=" * 70)
num_fail = 0
for i, (s, b) in enumerate(zip(singles, batched)):
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            lp_s = next(iter(ts.values())).logprob
            lp_b = next(iter(tb.values())).logprob
            max_lp = max(max_lp, abs(lp_s - lp_b))
    ok = tok_ok and max_lp == 0.0
    if not ok:
        num_fail += 1
    status = "OK" if ok else "MISMATCH"
    print(f"  Prompt {i}: tokens={'OK  ' if tok_ok else 'FAIL'}  logprob_diff={max_lp:.8f}  {status}")

print(f"\nBatch-invariant: {num_fail == 0}")
