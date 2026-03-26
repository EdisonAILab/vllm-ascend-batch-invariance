"""TP=4 test using SAME LLM instance for singles and batch (2048 tokens)."""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_gsm8k_prompts(n):
    with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
        return [json.loads(line)["question"] for i, line in enumerate(f) if i < n]

NUM_PROMPTS = 16
MAX_TOKENS = 2048
prompts = load_gsm8k_prompts(NUM_PROMPTS)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

print("=" * 70)
print("TP=4 Single-Instance Test: %d prompts, max_tokens=%d" % (NUM_PROMPTS, MAX_TOKENS))
print("Using SAME LLM instance for singles and batch")
print("=" * 70)

llm = LLM(**kwargs)

# Singles
print("\nRunning singles...")
t0 = time.time()
singles = []
for i, p in enumerate(prompts):
    out = llm.generate([p], sp)
    singles.append(out[0].outputs[0])
    print("  Single %d done (%d tokens)" % (i, len(out[0].outputs[0].token_ids)), flush=True)
t_single = time.time() - t0

# Batch
print("\nRunning batch...")
t0 = time.time()
batched = [o.outputs[0] for o in llm.generate(prompts, sp)]
t_batch = time.time() - t0

del llm

# Compare
print()
print("=" * 70)
print("Results: TP=4, %d prompts, max_tokens=%d (single instance)" % (NUM_PROMPTS, MAX_TOKENS))
print("=" * 70)
num_tok_fail = 0
num_lp_fail = 0
max_lp_overall = 0.0
total_tokens = 0
for i in range(NUM_PROMPTS):
    s, b = singles[i], batched[i]
    tok_ok = s.token_ids == b.token_ids
    max_lp = 0.0
    for ts, tb in zip(s.logprobs, b.logprobs):
        if ts and tb:
            d = abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob)
            max_lp = max(max_lp, d)
    lp_ok = max_lp == 0.0
    max_lp_overall = max(max_lp_overall, max_lp)
    if not tok_ok: num_tok_fail += 1
    if not lp_ok: num_lp_fail += 1
    n_s = len(s.token_ids)
    total_tokens += n_s
    status = "OK" if tok_ok and lp_ok else "MISMATCH"
    extra = ""
    if not tok_ok:
        min_len = min(n_s, len(b.token_ids))
        n_diff = sum(1 for a, bt in zip(s.token_ids[:min_len], b.token_ids[:min_len]) if a != bt)
        extra = "  (%d diffs)" % n_diff
    print("  Prompt %2d: tokens=%s  lp_diff=%.8f  gen=%4d  %s%s" % (
        i, "OK  " if tok_ok else "FAIL", max_lp, n_s, status, extra))

print()
print("  Token failures:  %d/%d" % (num_tok_fail, NUM_PROMPTS))
print("  Logprob failures: %d/%d" % (num_lp_fail, NUM_PROMPTS))
print("  Max logprob diff: %.8f" % max_lp_overall)
print("  Total tokens:    %d" % total_tokens)
print("  Batch-invariant: %s" % (num_tok_fail == 0 and num_lp_fail == 0))
print("  Time: singles=%.1fs  batch=%.1fs" % (t_single, t_batch))
