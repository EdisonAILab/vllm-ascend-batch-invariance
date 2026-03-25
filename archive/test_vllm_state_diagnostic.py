"""
Diagnostic: Log which attention state vLLM selects for single vs batched runs.
"""
import os
import sys
import json

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch_npu

def load_gsm8k_prompts(n):
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            prompts.append(json.loads(line)["question"])
    return prompts

# Monkey-patch _build_attn_state to log which state is selected
import vllm_ascend.worker.model_runner_v1 as mr
original_build = mr.NPUModelRunner._build_attn_state

_attn_state_log = []

def logging_build(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
    state = original_build(self, num_reqs, num_scheduled_tokens, num_valid_tokens)
    entry = f"num_reqs={num_reqs}, sched_tokens={num_scheduled_tokens[:num_reqs].tolist()}, state={state.name}"
    _attn_state_log.append(entry)
    return state

mr.NPUModelRunner._build_attn_state = logging_build

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
prompts = load_gsm8k_prompts(4)
sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=1)

print("Loading vLLM...")
sys.stdout.flush()
llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
          tensor_parallel_size=1, max_model_len=4096,
          gpu_memory_utilization=0.9)

# --- Single prompt runs ---
print("\n" + "=" * 70)
print("Single prompt runs")
print("=" * 70)
for i, p in enumerate(prompts):
    _attn_state_log.clear()
    out = llm.generate([p], sp)
    toks = list(out[0].outputs[0].token_ids)
    print(f"\nPrompt {i}: tokens={toks}")
    # Show first few state transitions
    for j, entry in enumerate(_attn_state_log[:5]):
        print(f"  step {j}: {entry}")
    if len(_attn_state_log) > 5:
        print(f"  ... ({len(_attn_state_log)} total steps)")
    sys.stdout.flush()

# --- Batched run ---
print("\n" + "=" * 70)
print(f"Batched run ({len(prompts)} prompts together)")
print("=" * 70)
_attn_state_log.clear()
out = llm.generate(prompts, sp)
for i, o in enumerate(out):
    toks = list(o.outputs[0].token_ids)
    print(f"Prompt {i}: tokens={toks}")

print(f"\nAttention state log ({len(_attn_state_log)} steps):")
for j, entry in enumerate(_attn_state_log[:15]):
    print(f"  step {j}: {entry}")
if len(_attn_state_log) > 15:
    print(f"  ... ({len(_attn_state_log)} total steps)")

# --- Compare tokens ---
print("\n" + "=" * 70)
print("Token comparison: single vs batch")
print("=" * 70)

single_tokens = []
for p in prompts:
    o = llm.generate([p], sp)
    single_tokens.append(list(o[0].outputs[0].token_ids))

batch_out = llm.generate(prompts, sp)
batch_tokens = [list(o.outputs[0].token_ids) for o in batch_out]

for i in range(len(prompts)):
    match = single_tokens[i] == batch_tokens[i]
    print(f"Prompt {i}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  single: {single_tokens[i]}")
        print(f"  batch:  {batch_tokens[i]}")

del llm
torch_npu.npu.empty_cache()
mr.NPUModelRunner._build_attn_state = original_build
