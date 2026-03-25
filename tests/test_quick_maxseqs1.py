"""Quick test: max_num_seqs=1 with matmul fix — should be 0 failures."""
import os, json, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
prompts = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")) as f:
    for i, line in enumerate(f):
        if i >= 4: break
        prompts.append(json.loads(line)["question"])

sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=1)
KWARGS = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.9, max_num_seqs=1)

# Singles
llm = LLM(**KWARGS)
singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
del llm; torch_npu.npu.empty_cache()

# Batch
llm = LLM(**KWARGS)
batched = [o.outputs[0] for o in llm.generate(prompts, sp)]
del llm; torch_npu.npu.empty_cache()

# Compare
for i, (s, b) in enumerate(zip(singles, batched)):
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            lp_s = next(iter(ts.values())).logprob
            lp_b = next(iter(tb.values())).logprob
            max_lp = max(max_lp, abs(lp_s - lp_b))
    ok = tok_ok and max_lp == 0.0
    print(f"Prompt {i}: tokens={'OK  ' if tok_ok else 'FAIL'}  logprob_diff={max_lp:.8f}  {'OK' if ok else 'MISMATCH'}")
