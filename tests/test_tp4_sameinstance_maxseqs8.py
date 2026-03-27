"""TP=4 max_num_seqs=8, SAME LLM instance — check if cross-batch state causes the issue."""
import json, os
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 16]

sp = SamplingParams(temperature=0.0, max_tokens=512, logprobs=1)
kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_num_seqs=8)

llm = LLM(**kwargs)

print("=== Same instance, max_num_seqs=8, 16 prompts ===")
singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
batched = [o.outputs[0] for o in llm.generate(prompts, sp)]

for i in range(16):
    s, b = singles[i], batched[i]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            max_lp = max(max_lp, abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob))
    tag = "OK" if tok_ok and max_lp == 0.0 else "FAIL"
    extra = ""
    if not tok_ok:
        d = sum(1 for a, bt in zip(s.token_ids, b.token_ids) if a != bt)
        extra = " (%d diffs)" % d
    print("  Prompt %2d: tokens=%s  lp=%.8f  %s%s" % (i, "OK" if tok_ok else "FAIL", max_lp, tag, extra))

# Also test: submit only prompts 8-15 as a batch (to see if they work alone)
print("\n=== Same instance, prompts 8-15 only as batch ===")
batched_8_15 = [o.outputs[0] for o in llm.generate(prompts[8:], sp)]
for i in range(8):
    s, b = singles[8+i], batched_8_15[i]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            max_lp = max(max_lp, abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob))
    tag = "OK" if tok_ok and max_lp == 0.0 else "FAIL"
    print("  Prompt %2d: %s  lp=%.8f" % (8+i, tag, max_lp))

del llm
