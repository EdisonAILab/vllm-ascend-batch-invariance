"""Narrow down TP=4 failure boundary between 8-16 prompts."""
import json, os
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    all_prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 16]

kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

llm = LLM(**kwargs)

# Test 256 tokens for each prompt count 8-16
print("=== TP=4 Narrowing: prompts 8-16, max_tokens=256 ===\n")
sp = SamplingParams(temperature=0.0, max_tokens=256, logprobs=1)

for n in range(8, 17):
    prompts = all_prompts[:n]
    singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
    batched = [o.outputs[0] for o in llm.generate(prompts, sp)]

    failures = 0
    max_lp = 0.0
    for s, b in zip(singles, batched):
        if s.token_ids != b.token_ids:
            failures += 1
        for ts, tb in zip(s.logprobs, b.logprobs):
            if ts and tb:
                d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
                max_lp = max(max_lp, d)

    tag = "OK" if failures == 0 and max_lp == 0.0 else "FAIL"

    # Also print total prefill tokens for the batch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    total_prefill = sum(len(tok.encode(p)) for p in prompts)

    print("  prompts=%2d  total_prefill_tokens=%4d: %d/%d fail  max_lp=%.8f  %s" % (
        n, total_prefill, failures, n, max_lp, tag))

del llm
