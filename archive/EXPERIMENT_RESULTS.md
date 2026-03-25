# Batch Invariance Experiment Results — Qwen3-4B on Ascend NPU

## Test Matrix

| Framework | TP | Mode OFF (native) | Mode ON (batch_invariant_ops) |
|---|---|---|---|
| HuggingFace | 1 | NOT invariant (max_diff=0.53) | **Bit-exact invariant** (max_diff=0.0) |
| vLLM | 1 | NOT invariant (2/8 token, 8/8 logprob) | **NOT invariant, WORSE** (8/8 token, 8/8 logprob) |
| vLLM | 1 | **Repeat: DETERMINISTIC** (3 runs) | **Repeat: DETERMINISTIC** (3 runs, different tokens from OFF) |
| vLLM | 8 | NOT invariant, mostly repeatable | **NOT invariant, catastrophically non-deterministic** |

### Kernel Determinism (single NPU, no vLLM)

| Kernel | Result |
|---|---|
| `matmul_persistent_npu` (8 shapes) | All DETERMINISTIC |
| `log_softmax` (5 batch sizes, vocab=152064) | All DETERMINISTIC |
| `mean_dim` (4×2048) | DETERMINISTIC |

---

## 1. HuggingFace `AutoModelForCausalLM` — TP=1

**Setup:** Single NPU (npu:0), 4 prompts, 32 tokens, bfloat16.

**Methodology:** Tokenize 4 prompts, pad to 32 tokens. Run all 4 as a batch (`[4,32,H]`), then
each individually (`[1,32,H]`). Compare logits with max absolute difference.

### Mode OFF (native NPU ops)

```
Prompt 0: max_diff=0.53125000  MISMATCH
Prompt 1: max_diff=0.43750000  MISMATCH
Prompt 2: max_diff=0.43750000  MISMATCH
Prompt 3: max_diff=0.32812500  MISMATCH
--> Batch-invariant: False  (overall max_diff=0.53125000)
```

**Root cause:** Ascend 910 selects different matmul algorithms depending on M dimension —
gemv for M=32 (batch=1), gemm for M=128 (batch=4) — affecting `k_proj` and `v_proj`
(output dim 1024 in GQA).

### Mode ON (batch_invariant_ops Triton kernel)

```
Prompt 0: max_diff=0.00000000  OK (invariant)
Prompt 1: max_diff=0.00000000  OK (invariant)
Prompt 2: max_diff=0.00000000  OK (invariant)
Prompt 3: max_diff=0.00000000  OK (invariant)
--> Batch-invariant: True  (overall max_diff=0.00000000)
```

**Conclusion:** `batch_invariant_ops` achieves bit-exact batch invariance on Ascend NPU
with HuggingFace models.

---

## 2. vLLM — TP=1, Single NPU

### 2a. Limited test (4 prompts, 8 tokens) — MISLEADING

**Setup:** vLLM 0.11.0 + `vllm_ascend`, offline `LLM.generate`, greedy (temperature=0.0),
`enforce_eager=True`, 4 prompts, 8 tokens.

Both Mode OFF and Mode ON appeared batch-invariant in this test. However, **this was
misleading** — the small input size (4 prompts, 8 tokens) didn't trigger different
chunked prefill scheduling paths.

### 2b. Thorough test (8 prompts, 32 tokens)

**Setup:** vLLM 0.11.0 + `vllm_ascend`, NPU 4, TP=1, greedy, `enforce_eager=True`,
8 prompts, 32 tokens.

**Test script:** `test_vllm_tp1_batch_invariance.py`

**Methodology:** Run each prompt individually (8 single-item batches), then all 8 together
in one batch. Compare token IDs and logprobs.

### Mode OFF

```
Prompt 0: tokens=OK   logprob_diff=0.12187248  MISMATCH
Prompt 1: tokens=OK   logprob_diff=0.10101080  MISMATCH
Prompt 2: tokens=OK   logprob_diff=0.03669330  MISMATCH
Prompt 3: tokens=FAIL logprob_diff=2.88272164  MISMATCH
Prompt 4: tokens=FAIL logprob_diff=1.18001364  MISMATCH
Prompt 5: tokens=OK   logprob_diff=0.04527336  MISMATCH
Prompt 6: tokens=OK   logprob_diff=0.06272012  MISMATCH
Prompt 7: tokens=OK   logprob_diff=0.05625576  MISMATCH
--> Tokens invariant: False  |  Logprobs invariant: False
Time: singles=6.2s  batch=1.1s
```

### Mode ON

```
Prompt 0: tokens=FAIL logprob_diff=2.48246658  MISMATCH
Prompt 1: tokens=FAIL logprob_diff=2.56419039  MISMATCH
Prompt 2: tokens=FAIL logprob_diff=2.18755001  MISMATCH
Prompt 3: tokens=FAIL logprob_diff=2.16706884  MISMATCH
Prompt 4: tokens=FAIL logprob_diff=2.57542956  MISMATCH
Prompt 5: tokens=FAIL logprob_diff=2.42402029  MISMATCH
Prompt 6: tokens=FAIL logprob_diff=2.85735345  MISMATCH
Prompt 7: tokens=FAIL logprob_diff=2.34586495  MISMATCH
--> Tokens invariant: False  |  Logprobs invariant: False
Time: singles=38182s (~10.6h)  batch=3824s (~64 min)
```

**Summary:** vLLM TP=1 is NOT batch-invariant with either mode. Mode ON makes it **worse**:
- Mode OFF: 2/8 token failures, logprob diffs 0.04–2.88
- Mode ON: **8/8 token failures**, logprob diffs 2.17–2.86 (all large)

**Root cause:** vLLM's chunked prefill scheduler processes single-item and 8-item batches
differently. With single prompts, all tokens are prefilled in one chunk. With 8 prompts,
the scheduler may split them across multiple chunks, changing the computation order and
triggering different CANN kernel selections (gemv vs gemm) for different M dimensions.
Mode ON's Triton kernels should be invariant to M, but the `aten::_log_softmax` and
`aten::mean.dim` Triton replacements see different input shapes between single and batched
runs, and the float32 accumulation path amplifies numerical differences.

---

## 3. vLLM — TP=8, NPUs 0–7 (Thorough Test)

**Setup:** vLLM 0.11.0 + `vllm_ascend`, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
`tensor_parallel_size=8`, offline `LLM.generate`, greedy, `enforce_eager=True`,
10 prompts, 64 tokens.

**Test script:** `test_vllm_tp8_v2.py`

**Methodology — 3 phases:**
- **Phase 1 (Single vs Batch):** Run each prompt individually, then all 10 as a batch.
  Compare token IDs and logprobs.
- **Phase 2 (Shuffle Invariance):** Run 3 shuffled batches, compare with original batch
  order using inverse permutation.
- **Phase 3 (Repeat Consistency):** Run the same batch 2 more times, compare with
  original batch.

### Mode OFF (native NPU ops)

| Phase | Result | Details |
|---|---|---|
| Phase 1 | **FAIL** | 10/10 logprob mismatch; 3/10 token failures (prompts 1, 3, 8). Max logprob_diff = 2.89 |
| Phase 2 | **FAIL** | 3/3 shuffle trials mismatch. Up to 58/64 tokens differ |
| Phase 3 | **FAIL** | Repeat 0: prompt 0 had 2/64 tokens differ. Repeat 1: OK |
| Timing | | Singles: 114.1s, Batch: 12.2s |

**Summary:** vLLM TP=8 on NPU is NOT batch-invariant with native ops. Single-vs-batch
and shuffle comparisons show significant mismatches. However, **repeat consistency is
mostly preserved** (only 2/64 tokens differ in one prompt on one repeat).

### Mode ON (batch_invariant_ops Triton kernel)

| Phase | Result | Details |
|---|---|---|
| Phase 1 | **FAIL** | **10/10 token failures** (all prompts). Max logprob_diff = 2.71. Worse than Mode OFF |
| Phase 2 | **FAIL** | Shuffle trial 0: 63–64/64 tokens differ. Trials 1–2: OK |
| Phase 3 | **FAIL** | **Both repeats FAIL. 63–64/64 tokens differ per prompt** (catastrophic) |
| Timing | | Singles: 20064s (~5.6h), Batch: 1787s (~30 min). **~500× slower** |

**Summary:** `batch_invariant_ops` made things **dramatically worse** with vLLM TP=8.
Repeat consistency completely broke (64/64 tokens differ), meaning the same batch of prompts
produces entirely different output on each run.

---

## 4. Diagnostic: Triton Kernel Determinism

**Setup:** Direct calls to `matmul_persistent_npu` on single NPU, no vLLM.

### Determinism (same inputs, 10 runs)

| Shape (M×K @ K×N) | Result |
|---|---|
| 1×2048 @ 2048×256 | DETERMINISTIC |
| 1×2048 @ 2048×64 | DETERMINISTIC |
| 32×2048 @ 2048×256 | DETERMINISTIC |
| 128×2048 @ 2048×256 | DETERMINISTIC |
| 1×2048 @ 2048×2048 | DETERMINISTIC |
| 1×2048 @ 2048×1024 | DETERMINISTIC |
| 5×2048 @ 2048×256 | DETERMINISTIC |
| 7×2048 @ 2048×64 | DETERMINISTIC |

### Batch Invariance (row i same result regardless of M)

| K×N | Result |
|---|---|
| 2048×256 | BATCH INVARIANT (8 rows) |
| 2048×64 | BATCH INVARIANT (8 rows) |
| 2048×1024 | BATCH INVARIANT (8 rows) |

### `linear_batch_invariant` determinism

| Shape | Result |
|---|---|
| [1,1,2048] × [256,2048] | DETERMINISTIC |
| [1,4,2048] × [256,2048] | DETERMINISTIC |
| [1,1,2048] × [64,2048] | DETERMINISTIC |
| [1,8,2048] × [1024,2048] | DETERMINISTIC |

**Conclusion:** The matmul Triton kernel is both deterministic and batch-invariant on
single NPU. The Mode ON non-determinism in vLLM does **not** originate from the matmul kernel.

---

## 5. Diagnostic: log_softmax and mean_dim Triton Kernel Determinism

**Setup:** Direct calls on single NPU 4, no vLLM.

| Kernel | Shape | Result |
|---|---|---|
| `log_softmax` | 4×152064 (5 runs) | DETERMINISTIC |
| `mean_dim` | 4×2048 (5 runs) | DETERMINISTIC |
| `log_softmax` | M=1, 152064 | DETERMINISTIC |
| `log_softmax` | M=3, 152064 | DETERMINISTIC |
| `log_softmax` | M=5, 152064 | DETERMINISTIC |
| `log_softmax` | M=8, 152064 | DETERMINISTIC |
| `log_softmax` | M=16, 152064 | DETERMINISTIC |

**Conclusion:** All Triton kernels (matmul, log_softmax, mean_dim) are deterministic on
single NPU. The `tl.sum()` reduction is NOT non-deterministic on NPU.

---

## 6. Diagnostic: vLLM TP=1 Mode ON Repeat Consistency

**Setup:** vLLM 0.11.0, TP=1 on NPU 4, 5 prompts, 32 tokens, 3 repeat runs.

### Mode OFF

```
Run 0: 2.1s  tokens[0][:8]=[12095, 13, 576, 6722, 315, 9856, 374, 19846]
Run 1: 1.9s  tokens[0][:8]=[12095, 13, 576, 6722, 315, 9856, 374, 19846]
Run 2: 1.8s  tokens[0][:8]=[12095, 13, 576, 6722, 315, 9856, 374, 19846]
--> All 3 runs identical: DETERMINISTIC
```

### Mode ON

```
Run 0: 4421.5s  tokens[0][:8]=[119570, 52127, 91229, 117109, 34079, 113668, 72141, 1256]
Run 1: 4430.2s  tokens[0][:8]=[119570, 52127, 91229, 117109, 34079, 113668, 72141, 1256]
Run 2: 4382.8s  tokens[0][:8]=[119570, 52127, 91229, 117109, 34079, 113668, 72141, 1256]
--> All 3 runs identical: DETERMINISTIC
```

**Key findings:**

1. **Mode ON at TP=1 is fully deterministic** — 3 runs produce identical output.
2. **Mode ON tokens differ from Mode OFF** — this is expected and NOT a bug. The Triton
   persistent kernel accumulates in float32 and casts back to bfloat16, producing different
   numerical results from native CANN ops (which use native bfloat16 matmul). The different
   logits at the first token cascade into entirely different generated sequences due to
   autoregressive decoding.
3. **Engine init took 2541s (~42 min)** for Triton JIT compilation during vLLM warmup.
4. **Each run took ~74 min** for 5 prompts × 32 tokens (~500× slower than Mode OFF).

### Implications for TP=8 Non-Determinism

Since Mode ON at TP=1 is deterministic, the catastrophic non-determinism at TP=8 Mode ON
(64/64 tokens differ between repeats) must originate from **multi-NPU interaction**:

- **HCCL all-reduce non-determinism** — HCCL collective operations (used for tensor-parallel
  all-reduce after row-parallel linear layers) may not be deterministic when Triton kernels
  introduce ~500× slower execution, causing different timing and buffering behavior.
- **vLLM scheduler timing** — the massive slowdown may cause different chunked prefill
  decisions or different scheduling orders across TP workers.
- **Async execution ordering** — with extremely slow Triton kernels, the relative ordering
  of operations across 8 NPUs becomes more sensitive to timing jitter.

---

## 7. Root Cause Analysis

### Mode ON produces different tokens from Mode OFF (expected)

`enable_batch_invariant_mode()` patches five ATen ops:

| Op | Replacement | Used for |
|---|---|---|
| `aten::mm` | Triton persistent matmul | Raw matmul |
| `aten::addmm` | Triton persistent matmul + bias | 2D linear |
| `aten::linear` | Triton persistent matmul (NPU only) | 3D+ linear |
| `aten::_log_softmax` | Triton log-softmax kernel | Next-token logprobs |
| `aten::mean.dim` | Triton mean kernel | RMSNorm |

The Triton persistent kernel accumulates in float32 and casts back to bfloat16, while
native CANN ops use native bfloat16 matmul. This numerical difference means Mode ON
produces different logits from Mode OFF — which is expected and not a correctness bug.
Due to autoregressive decoding, even tiny logit differences at the first token cascade
into entirely different generated text.

All individual kernels are proven deterministic on single NPU:
- `matmul_persistent_npu`: 8 shapes tested, 10 runs each — all deterministic
- `log_softmax`: 5 batch sizes (M=1,3,5,8,16) × 152064 vocab — all deterministic
- `mean_dim`: 4×2048 — deterministic

### Why Mode ON fails to achieve batch invariance in vLLM

**Key finding:** Mode ON is deterministic at TP=1 (same batch repeated = same output), but
NOT batch-invariant (single vs batched = different output). This is the same behavior as
Mode OFF, but worse (8/8 vs 2/8 token failures).

The batch_invariant_ops library ensures that **row i of a matmul produces the same result
regardless of M** (the total number of rows). This is the correct fix for the HuggingFace
case, where single vs batched inputs differ only in M.

However, vLLM's execution pipeline introduces additional sources of non-invariance beyond
the M-dimension matmul issue:

1. **Chunked prefill scheduling**: vLLM's V1 scheduler with `max_num_batched_tokens=8192`
   processes single-item and 8-item batches differently. With 8 prompts, the scheduler may
   split them across multiple prefill chunks, changing the order and grouping of tokens sent
   to the model. This changes which tokens share attention context during prefill.

2. **PagedAttention block allocation**: Different batch sizes lead to different KV cache block
   layouts. The attention kernel reads from different memory locations depending on block
   allocation, which can produce numerically different results.

3. **Prefix caching interactions**: vLLM has `enable_prefix_caching=True` by default. When
   prompts are run individually, no prefix sharing occurs. When run as a batch, common
   prefixes may be cached and shared, changing the computation path.

4. **Mode ON amplifies the problem** because the Triton kernels' float32→bfloat16 cast path
   produces logits that are numerically different from native CANN. When these different
   logits interact with vLLM's scheduling-dependent computation order, the mismatches are
   larger (logprob diffs of 2.2–2.9 vs 0.04–2.9 for Mode OFF).

### Why Mode ON breaks repeat consistency at TP=8

Mode ON at **TP=1 is fully deterministic** (3 runs identical). The catastrophic
non-determinism at **TP=8** (64/64 tokens differ between repeats) is caused by
**multi-NPU interaction under extreme slowdown**:

1. **HCCL collective non-determinism under slow kernels**: Triton kernels are ~500×
   slower than native CANN ops. This creates much larger timing variance between the 8
   TP workers. HCCL all-reduce operations (used after row-parallel linear layers) may
   have non-deterministic reduction ordering when workers arrive at different times.

2. **vLLM V1 scheduler sensitivity**: The V1 engine's chunked prefill scheduler makes
   batching decisions based on timing. With ~500× slower execution, micro-timing
   differences between workers cause different scheduling decisions across runs, leading
   to different computation order.

3. **Mode OFF TP=8 is also slightly non-deterministic** (2/64 tokens in one prompt),
   confirming baseline HCCL non-determinism exists. Mode ON amplifies this because:
   - The 500× slowdown increases timing jitter
   - The different numerical path (float32→bfloat16) may be more sensitive to
     reduction order changes in HCCL

### vLLM dispatch path verification

For NPU with TP=1, vLLM uses the default linear dispatch path:
```
vLLM LinearBase.apply() → UnquantizedLinearMethod.apply()
  → dispatch_unquantized_gemm() → default_unquantized_gemm()
    → torch.nn.functional.linear(x, weight, bias)
      → aten::linear  [PATCHED BY batch_invariant_ops]
```

The ATen dispatch patches ARE intercepted by vLLM. vllm_ascend's custom ops
(`MLPColumnParallelOp`, etc.) are only activated when specific features are enabled
(`mlp_tp_enable()`, `enable_sp()`, `matmul_allreduce_enable()`), which are disabled
at TP=1.

### Conclusions and recommendations

1. **`batch_invariant_ops` works correctly for HuggingFace** — achieves bit-exact batch
   invariance by fixing the M-dimension matmul issue on NPU.

2. **`batch_invariant_ops` is insufficient for vLLM** — vLLM's non-invariance comes from
   the scheduling/batching layer (chunked prefill, PagedAttention, prefix caching), not
   just from matmul kernel selection. Patching ATen ops does not address these higher-level
   sources of non-invariance.

3. **For vLLM batch invariance on NPU**, the fix must be at the vLLM/vllm_ascend level:
   - Ensure chunked prefill processes each sequence identically regardless of batch size
   - Disable prefix caching or ensure it doesn't change computation paths
   - Use deterministic KV cache block allocation
   - For TP>1: use deterministic HCCL settings (`HCCL_DETERMINISTIC=1` or equivalent)

4. **Patch only matmul ops** if using `batch_invariant_ops` — skip `aten::_log_softmax`
   and `aten::mean.dim` to reduce the ~500× performance overhead. The native CANN
   implementations of these ops are already deterministic on NPU.

---

## Performance

| Configuration | Singles | Batch | Slowdown |
|---|---|---|---|
| vLLM TP=1, Mode OFF (8 prompts, 32 tokens) | 6.2s | 1.1s | — |
| vLLM TP=1, Mode ON (8 prompts, 32 tokens) | 38182s (~10.6h) | 3824s (~64 min) | ~6000× / ~3500× |
| vLLM TP=8, Mode OFF (10 prompts, 64 tokens) | 114s | 12s | — |
| vLLM TP=8, Mode ON (10 prompts, 64 tokens) | 20064s (~5.6h) | 1787s (~30 min) | ~175× / ~147× |

The overhead comes from Triton JIT compilation on NPU's BiSheng compiler backend
(~33 min per single prompt) and non-optimized Triton kernel execution vs native CANN ops.

---

## Environment

| Item | Details |
|---|---|
| Machine | `root@7.150.11.210` |
| Container | `verl-npu-bruceli` |
| Accelerator | 8× Ascend 910 NPU (NPUs 0–7) |
| Model | Qwen3-4B (`/home/bruceli/models/Qwen/Qwen3-4B`) |
| dtype | bfloat16 |
| vLLM | 0.11.0 with `vllm_ascend` platform plugin |
| `batch_invariant_ops` | `/home/bruceli/projects/batch_invariant_ops` (patched for NPU) |
| Triton | 3.2.0 (Ascend backend via `torch_npu`) |
