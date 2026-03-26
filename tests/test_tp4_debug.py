"""Debug TP=4: test with max_num_seqs=1 and also check self-consistency at 2048 tokens."""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 4]

sp = SamplingParams(temperature=0.0, max_tokens=2048, logprobs=1)
kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

# Test 1: self-consistency (same prompt, two runs, same LLM)
print("=== Test 1: Self-consistency (2048 tokens, TP=4) ===")
llm = LLM(**kwargs)
run1 = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
run2 = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
for i in range(len(prompts)):
    tok_ok = run1[i].token_ids == run2[i].token_ids
    n = len(run1[i].token_ids)
    tag = "OK" if tok_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        diffs = sum(1 for a, b in zip(run1[i].token_ids, run2[i].token_ids) if a != b)
        extra = " (%d/%d differ)" % (diffs, n)
    print("  Prompt %d: %d toks  %s%s" % (i, n, tag, extra))
del llm; torch_npu.npu.empty_cache()

# Test 2: batch invariance with max_num_seqs=1 (sequential scheduling)
print("\n=== Test 2: max_num_seqs=1, TP=4, 2048 tokens ===")
kwargs2 = dict(**kwargs, max_num_seqs=1)
llm = LLM(**kwargs2)
singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
batched = [o.outputs[0] for o in llm.generate(prompts, sp)]
for i in range(len(prompts)):
    tok_ok = singles[i].token_ids == batched[i].token_ids
    n = len(singles[i].token_ids)
    tag = "OK" if tok_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        diffs = sum(1 for a, b in zip(singles[i].token_ids, batched[i].token_ids) if a != b)
        extra = " (%d/%d differ)" % (diffs, n)
    print("  Prompt %d: %d toks  %s%s" % (i, n, tag, extra))
del llm; torch_npu.npu.empty_cache()

# Test 3: batch invariance WITHOUT our patches (env var off)
print("\n=== Test 3: TP=4, 2048 tokens, patches OFF ===")
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "0"
llm = LLM(**kwargs)
singles_off = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
batched_off = [o.outputs[0] for o in llm.generate(prompts, sp)]
for i in range(len(prompts)):
    tok_ok = singles_off[i].token_ids == batched_off[i].token_ids
    n = len(singles_off[i].token_ids)
    tag = "OK" if tok_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        diffs = sum(1 for a, b in zip(singles_off[i].token_ids, batched_off[i].token_ids) if a != b)
        extra = " (%d/%d differ)" % (diffs, n)
    print("  Prompt %d: %d toks  %s%s" % (i, n, tag, extra))
del llm
