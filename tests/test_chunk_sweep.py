"""Experiment #7: Matmul chunk size sweep. Test correctness and performance for different chunk sizes."""
import json, os, time
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch_npu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "chunk_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 8]

sp = SamplingParams(temperature=0.0, max_tokens=256, logprobs=1)
base_kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=False, enable_chunked_prefill=False)

results = []
for chunk_size in [32, 64, 128, 256, 512]:
    os.environ["MATMUL_CHUNK_SIZE"] = str(chunk_size)
    print("\n=== Chunk size: %d ===" % chunk_size)

    llm = LLM(**base_kwargs)

    # Singles
    t0 = time.time()
    singles = [llm.generate([p], sp)[0].outputs[0] for p in prompts]
    t_single = time.time() - t0

    # Batch
    t0 = time.time()
    batched = [o.outputs[0] for o in llm.generate(prompts, sp)]
    t_batch = time.time() - t0

    del llm; torch_npu.npu.empty_cache()

    # Compare
    tok_fail = 0
    lp_max = 0.0
    for s, b in zip(singles, batched):
        if s.token_ids != b.token_ids:
            tok_fail += 1
        for ts, tb in zip(s.logprobs, b.logprobs):
            if ts and tb:
                lp_max = max(lp_max, abs(next(iter(ts.values())).logprob - next(iter(tb.values())).logprob))

    invariant = tok_fail == 0 and lp_max == 0.0
    tag = "PASS" if invariant else "FAIL"
    print("  %s  tok_fail=%d/%d  max_lp=%.8f  t_single=%.1fs  t_batch=%.1fs" % (
        tag, tok_fail, len(prompts), lp_max, t_single, t_batch))

    results.append({
        "chunk_size": chunk_size, "invariant": invariant,
        "token_failures": tok_fail, "max_logprob_diff": lp_max,
        "time_singles_s": round(t_single, 1), "time_batch_s": round(t_batch, 1),
    })

with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
for r in results:
    tag = "PASS" if r["invariant"] else "FAIL"
    print("  chunk=%3d: %s  batch=%.1fs" % (r["chunk_size"], tag, r["time_batch_s"]))
