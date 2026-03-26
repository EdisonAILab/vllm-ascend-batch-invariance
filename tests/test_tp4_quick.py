"""Quick TP=4 diagnosis: 4 prompts, 8 tokens. Check if issue is allreduce non-determinism."""
import os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
prompts = [
    "What is 2+2?",
    "The capital of France is",
    "Explain gravity in one sentence.",
    "Write a haiku about mountains.",
]
sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=1)
kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

# Run singles
print("Singles (TP=4)...")
llm = LLM(**kwargs)
singles = []
for p in prompts:
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
del llm; torch_npu.npu.empty_cache()

# Run batch
print("Batch (TP=4)...")
llm = LLM(**kwargs)
batched = [o.outputs[0] for o in llm.generate(prompts, sp)]
del llm; torch_npu.npu.empty_cache()

# Run singles AGAIN with new LLM to check self-consistency
print("Singles again (TP=4)...")
llm = LLM(**kwargs)
singles2 = []
for p in prompts:
    out = llm.generate([p], sp)
    singles2.append(out[0].outputs[0])
del llm; torch_npu.npu.empty_cache()

print("\n=== Single vs Batch ===")
for i in range(len(prompts)):
    s, b = singles[i], batched[i]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
            max_lp = max(max_lp, d)
    tag = "OK" if tok_ok and max_lp == 0.0 else "MISMATCH"
    print("  Prompt %d: tokens=%s lp_diff=%.8f %s" % (i, "OK" if tok_ok else "FAIL", max_lp, tag))
    if not tok_ok:
        print("    single: %s" % list(s.token_ids))
        print("    batch:  %s" % list(b.token_ids))

print("\n=== Single Run1 vs Single Run2 (self-consistency) ===")
for i in range(len(prompts)):
    s1, s2 = singles[i], singles2[i]
    tok_ok = s1.token_ids == s2.token_ids
    max_lp = 0.0
    for ts1, ts2 in zip(s1.logprobs, s2.logprobs):
        if ts1 and ts2:
            d = abs(next(iter(ts1.values())).logprob - next(iter(ts2.values())).logprob)
            max_lp = max(max_lp, d)
    tag = "OK" if tok_ok and max_lp == 0.0 else "MISMATCH"
    print("  Prompt %d: tokens=%s lp_diff=%.8f %s" % (i, "OK" if tok_ok else "FAIL", max_lp, tag))
    if not tok_ok:
        print("    run1: %s" % list(s1.token_ids))
        print("    run2: %s" % list(s2.token_ids))
