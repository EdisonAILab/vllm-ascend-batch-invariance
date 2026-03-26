# Next Experiments for Batch Invariance

Current state: TP=1 achieves 16/16 bit-exact match on GSM8K (16 prompts, 2048 tokens). TP=4 experiment is in progress.

All experiments run on the Ascend 910 NPU server (`root@7.150.11.210`, container `verl-npu-bruceli`). Apply all three patches before running, and set `VLLM_NPU_BATCH_INVARIANT_MATMUL=1`.

---

## 1. Larger Batch Sizes

Test batch=32, 64, 128 to verify invariance holds at higher utilization. The matmul chunk size is 128, so batches with M > 128 exercise the chunking logic more heavily.

```python
# In test script, change:
NUM_PROMPTS = 64  # or 32, 128
# Keep everything else the same as test_gsm8k_2048_save.py
```

Save results to `results/batch64/`, `results/batch128/` etc.

## 2. Non-Greedy Sampling with Fixed Seed

Current tests use `temperature=0.0` (greedy). Test with temperature > 0 and a fixed seed to verify the sampling RNG path is also batch-invariant.

```python
sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, logprobs=1, seed=42)
```

Compare single vs batch outputs. With a fixed seed, token IDs should match if the underlying logits are identical. Save results to `results/sampling_temp06/`.

## 3. Prefix Caching Enabled

Currently disabled with `enable_prefix_caching=False`. Enable it to check if KV cache sharing introduces new batch-dependent code paths.

```python
LLM_KWARGS = dict(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=True,   # <-- changed
                   enable_chunked_prefill=False)
```

Save results to `results/prefix_caching/`.

## 4. Chunked Prefill Enabled

Currently disabled. The attention patch explicitly handles chunked prefill, so this tests that code path.

```python
LLM_KWARGS = dict(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=False,
                   enable_chunked_prefill=True)  # <-- changed
```

Save results to `results/chunked_prefill/`.

## 5. Larger Model (Qwen3-8B)

Test whether the same three patches are sufficient for a larger model, or if new operators become M-dependent at larger hidden dimensions.

```python
MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-8B"  # download if not present
# May need TP=2 or TP=4 depending on memory
```

Save results to `results/qwen3_8b/`.

## 6. Different Model Architecture (LLaMA-based)

Test a non-Qwen model to confirm the fixes are architecture-agnostic on Ascend. Pick whatever LLaMA-family model is available on the server.

```bash
# Check what models are available
ls /home/bruceli/models/
```

Save results to `results/<model_name>/`.

## 7. Matmul Chunk Size Sweep

The current chunk size is 128. Benchmark different values to find the performance-optimal setting.

Modify `patches/patch_matmul_invariance.py` to accept the chunk size from an env var, then test:

```bash
for CHUNK in 64 128 256 512; do
  VLLM_NPU_BATCH_INVARIANT_MATMUL=1 MATMUL_CHUNK_SIZE=$CHUNK python tests/test_comprehensive.py
done
```

Record both correctness (invariance) and wall-clock time for each chunk size. Save results to `results/chunk_sweep/`.

## 8. Full GSM8K Evaluation (All 1319 Prompts)

Run all 1319 prompts from `gsm8k_test.jsonl` and compare actual GSM8K accuracy (extract numerical answers, check against ground truth) between single and batch modes. This validates the patches have no accuracy impact at scale.

```python
NUM_PROMPTS = 1319
MAX_TOKENS = 2048
```

This will take significantly longer. Save results to `results/full_gsm8k/`.

## 9. Longer Sequences (max_tokens=4096)

Push generation length to stress the attention patch across more autoregressive decode steps.

```python
MAX_TOKENS = 4096
LLM_KWARGS = dict(..., max_model_len=8192, ...)
```

Save results to `results/max4096/`.

## 10. TP=2 and TP=8

Fill in the tensor parallelism scaling picture. TP=2 is useful to test the minimal multi-card case.

```python
# TP=2
TP_SIZE = 2
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

# TP=8 (if 8 NPUs available)
TP_SIZE = 8
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
```

Save results to `results/tp2/` and `results/tp8/`.

---

## Priority Order

1. **Larger batch sizes** (#1) — most likely to break the chunk=128 assumption
2. **Non-greedy sampling** (#2) — different code path, quick to test
3. **Chunked prefill enabled** (#4) — directly tests patched attention code
4. **Prefix caching** (#3) — quick config change
5. **Full GSM8K eval** (#8) — strongest correctness validation
6. Everything else as time allows
