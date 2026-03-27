"""Bisect TP=4 batch invariance failure: vary prompts and tokens to find the boundary."""
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

print("=== TP=4 Bisect: varying num_prompts and max_tokens ===")
print("Using SAME LLM instance for all tests\n")

for num_prompts in [2, 4, 8, 16]:
    prompts = all_prompts[:num_prompts]
    for max_tokens in [32, 128, 512, 1024]:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)

        # Singles
        singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
        # Batch
        batched = [o.outputs[0] for o in llm.generate(prompts, sp)]

        # Compare
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
        print("  prompts=%2d  max_tokens=%4d: %d/%d fail  max_lp=%.8f  %s" % (
            num_prompts, max_tokens, failures, num_prompts, max_lp, tag))
    print()

del llm
