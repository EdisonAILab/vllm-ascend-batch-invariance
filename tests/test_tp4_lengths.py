"""Test TP=4 batch invariance at different generation lengths to find where it breaks."""
import os
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

kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

llm = LLM(**kwargs)

for max_tokens in [8, 16, 32, 64, 128, 256, 512]:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)

    # Singles
    singles = []
    for p in prompts:
        out = llm.generate([p], sp)
        singles.append(out[0].outputs[0])

    # Batch
    batched = [o.outputs[0] for o in llm.generate(prompts, sp)]

    # Compare
    failures = 0
    max_lp = 0.0
    for i in range(len(prompts)):
        s, b = singles[i], batched[i]
        if s.token_ids != b.token_ids:
            failures += 1
        for ts, tb in zip(s.logprobs, b.logprobs):
            if ts and tb:
                d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
                max_lp = max(max_lp, d)

    tag = "OK" if failures == 0 and max_lp == 0.0 else "FAIL"
    print("max_tokens=%4d: %d/%d failures, max_lp_diff=%.8f  %s" % (
        max_tokens, failures, len(prompts), max_lp, tag))

del llm
