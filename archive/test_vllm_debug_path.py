"""
Debug: Trace which attention code paths are hit in vLLM single vs batch.
Adds logging to the forward() method to see attn_state and num_seqs.
"""
import os
import sys
import json

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch_npu

# Patch the attention forward to log state info
# We must patch the SOURCE FILE since vLLM forks a subprocess
import vllm_ascend.attention.attention_v1 as attn_mod
import inspect

# Get original forward source to check current state
src = inspect.getsource(attn_mod.AscendAttentionBackendImpl._forward_prefill_no_cache)
has_loop = "for i in range(num_seqs)" in src
print(f"Patch check: _forward_prefill_no_cache has per-seq loop: {has_loop}")

# Add logging wrapper to forward()
_orig_forward = attn_mod.AscendAttentionBackendImpl.forward
_call_count = [0]

def _logging_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, trace_flag=True):
    _call_count[0] += 1
    if _call_count[0] <= 50:  # Only log first 50 calls
        state_name = attn_metadata.attn_state.name if attn_metadata else "None"
        num_seqs = attn_metadata.seq_lens.shape[0] if attn_metadata and attn_metadata.seq_lens is not None else 0
        seq_lens_list = attn_metadata.seq_lens_list if attn_metadata else []
        print(f"  [FWD #{_call_count[0]}] layer={layer.layer_name[:30]:30s} state={state_name:20s} num_seqs={num_seqs} seq_lens={seq_lens_list[:8]}", flush=True)
    return _orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, trace_flag)

attn_mod.AscendAttentionBackendImpl.forward = _logging_forward

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
sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)

print("\nLoading vLLM...")
llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
          tensor_parallel_size=1, max_model_len=4096,
          gpu_memory_utilization=0.9)

print("\n" + "=" * 70)
print("Single prompt run (prompt 0)")
print("=" * 70)
_call_count[0] = 0
out = llm.generate([prompts[0]], sp)
print(f"tokens={list(out[0].outputs[0].token_ids)}")

print("\n" + "=" * 70)
print(f"Batch run ({len(prompts)} prompts)")
print("=" * 70)
_call_count[0] = 0
out = llm.generate(prompts, sp)
for i, o in enumerate(out):
    print(f"  Prompt {i}: tokens={list(o.outputs[0].token_ids)}")

del llm
torch_npu.npu.empty_cache()
