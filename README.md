# Batch Invariance for vLLM on Ascend NPU

**Bit-exact batch-invariant inference** for vLLM on Ascend 910 NPU. A given input sequence produces identical output tokens and logprobs regardless of what other sequences are in the same batch.

## Quick Start

```bash
# Apply patches (inside container)
python patches/patch_matmul_invariance.py
python patches/patch_addrmsnorm_invariance.py
python patches/patch_attention_invariance.py

# Run with batch invariance enabled
VLLM_NPU_BATCH_INVARIANT_MATMUL=1 python your_script.py
```

## Results

**GSM8K, 16 prompts, max_tokens=2048, Qwen3-4B bfloat16:**

| Metric | Value |
|---|---|
| Token match | **16/16 (100%)** |
| Logprob match | **16/16 (100%)** |
| Max logprob diff | **0.00000000** |
| Total tokens generated | **30,066** |
| Batch time | 112.4s |
| Sequential time | 1168.0s |
| Batch speedup | **10.4x** |

```
Prompt  0: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  1: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  2: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  3: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  4: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  5: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  6: tokens=OK  logprob_diff=0.00000000  gen=1236 toks  OK
Prompt  7: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  8: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt  9: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt 10: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt 11: tokens=OK  logprob_diff=0.00000000  gen= 158 toks  OK
Prompt 12: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt 13: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt 14: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
Prompt 15: tokens=OK  logprob_diff=0.00000000  gen=2048 toks  OK
```

Full generated responses (single and batch) are saved in [`results/`](results/).

### Additional Experiments (all PASS)

| Experiment | Prompts | Config | Result |
|---|---|---|---|
| **Batch-64** | 64 | greedy, 256 tokens | **0/64 failures** |
| **Non-greedy sampling** | 16 | temp=0.6, top_p=0.95, seed=42 | **0/16 failures** |
| **Prefix caching** | 16 | enable_prefix_caching=True | **0/16 failures** |
| **Chunked prefill** | 16 | enable_chunked_prefill=True | **0/16 failures** |

The operator-level fixes are robust across batch sizes, sampling strategies, and vLLM scheduling modes.

---

## Problem

On Ascend 910 NPU, vLLM produces **different outputs** for the same prompt depending on batch composition. The same prompt processed alone gives different token IDs and logprobs compared to processing it alongside other prompts.

**Root cause:** Three CANN operators internally select different algorithms based on the M dimension (total tokens packed across all sequences in a batch). Different batch compositions produce different M values, triggering different kernels with numerically different results. These small differences compound across transformer layers and autoregressive decoding steps, leading to completely divergent outputs.

---

## Root Cause Analysis

### 1. `F.linear` / matmul --- M-dependent at M >= 16

CANN's matmul selects different algorithms (gemv vs gemm) based on M (number of rows in the input matrix).

```
F.linear native (no padding):
  M=  4 vs M=1: diff=0.0000000000  OK
  M= 16 vs M=1: diff=0.0039062500  MISMATCH
  M=128 vs M=1: diff=0.0039062500  MISMATCH
```

**Fix:** Chunk the M dimension to a fixed size (128) so CANN always selects the same algorithm regardless of actual batch size.

### 2. `npu_add_rms_norm` --- M-dependent at M >= 49

The fused add+RMSNorm operator exhibits M-dependent behavior. Critically, `npu_rms_norm` (without fused add) is fully invariant.

```
npu_add_rms_norm:
  M= 48: 0/48 rows differ   OK
  M= 64: 64/64 rows differ  MISMATCH (max_diff=0.03125)
  M=128: 128/128 rows differ MISMATCH

npu_rms_norm (no fused add): invariant at all M values
```

**Fix:** Decompose `npu_add_rms_norm(x, residual, w, eps)` into separate `x = x + residual` followed by `npu_rms_norm(x, w, eps)`.

### 3. Attention kernels --- different kernels per scheduling state

vLLM uses three different NPU attention kernels depending on scheduling state:

| State | Kernel | When |
|---|---|---|
| PrefillNoCache | `_npu_flash_attention` | Single prompt, first pass |
| DecodeOnly | `_npu_paged_attention` | Single prompt, autoregressive |
| ChunkedPrefill | `npu_fused_infer_attention_score` | Mixed batch (prefill + decode) |

When batched, mixed scheduling triggers ChunkedPrefill (different kernel, different numerics). Additionally, `_npu_flash_attention` shows cross-sequence interference when multiple sequences are packed.

**Fix:** Process each sequence individually through the same kernel it would use in single-prompt mode.

---

## Operator Invariance Summary

| Operator | M-Invariant? | Threshold | Fix |
|---|---|---|---|
| `F.linear` (matmul) | No | M >= 16 | Chunk M to 128 |
| `npu_add_rms_norm` | No | M >= 49 | Decompose to `add` + `rms_norm` |
| `npu_rms_norm` | **Yes** | -- | None needed |
| `SiLU` | **Yes** | -- | None needed |
| `torch.mul` | **Yes** | -- | None needed |
| `_npu_flash_attention` | No | Multi-sequence | Per-sequence calls |
| `_npu_paged_attention` | No | Multi-sequence | Per-sequence calls |
| `npu_fused_infer_attention_score` | No | Different kernel | Replaced by flash/paged per-sequence |
| `HCCL allreduce` (TP only) | No | M >= 412 rows | Chunk to 384 rows |

---

## Patches

| Patch | Target File | What It Does |
|---|---|---|
| [`patches/patch_matmul_invariance.py`](patches/patch_matmul_invariance.py) | `vllm/.../layers/utils.py` | Replaces `dispatch_unquantized_gemm()` with M-chunked version (chunk=128) |
| [`patches/patch_addrmsnorm_invariance.py`](patches/patch_addrmsnorm_invariance.py) | `vllm_ascend/.../ops/layernorm.py` | Decomposes fused `npu_add_rms_norm` into `add` + `npu_rms_norm` |
| [`patches/patch_attention_invariance.py`](patches/patch_attention_invariance.py) | `vllm_ascend/.../attention/attention_v1.py` | Per-sequence attention for prefill, decode, and chunked prefill |
| [`patches/patch_allreduce_invariance.py`](patches/patch_allreduce_invariance.py) | `vllm/.../distributed/communication_op.py` | Chunks allreduce to <=384 rows for TP>1 batch invariance |

Each patch:
- Creates a `.bak` backup before modifying
- Is idempotent (re-running restores from backup first)
- Activated by `VLLM_NPU_BATCH_INVARIANT_MATMUL=1` environment variable

---

## Performance

### TP=1 (single NPU)

| Configuration | Time (16 prompts, 2048 tokens) | Batch Invariant? |
|---|---|---|
| Native vLLM (no fix) | ~112s batch | No (multiple failures) |
| `max_num_seqs=1` (sequential) | ~1168s | Yes, but **10x slower** |
| **Operator-level fixes** | **~112s batch** | **Yes, full speed** |

The operator-level fixes achieve batch invariance with **zero throughput penalty** compared to native batching.

### TP=4 (4 NPUs)

| Configuration | Time (16 prompts, 2048 tokens) | Batch Invariant? |
|---|---|---|
| Native vLLM (no fix) | ~112s batch | No (3/4 failures) |
| Operator fixes (3 patches) | ~129s batch | **No** (16/16 failures) |
| **All 4 patches + `HCCL_DETERMINISTIC=true`** | **~1651s** | **Yes, 16/16 bit-exact** |

**4th patch: HCCL allreduce chunking.** HCCL's allreduce is M-dependent at M>=412 rows (with hidden_size=2560, 4 ranks). Chunking allreduce to <=384 rows eliminates this. Requires `HCCL_DETERMINISTIC=true`.

**Key findings for TP=4:**
- `HCCL_DETERMINISTIC=true` is required (value must be `true`/`false`/`strict`, not `1`/`0`)
- `allreduce` is M-dependent at M>=412 rows (~2 MB tensor threshold)
- Chunked allreduce (384 rows) restores M-invariance
- Self-consistency is perfect (same prompt gives same result across runs)

---

## Repository Structure

```
.
├── README.md                              # This file
├── gsm8k_test.jsonl                       # GSM8K test set (1319 prompts)
├── patches/
│   ├── patch_matmul_invariance.py         # Fix 1: matmul M-chunking
│   ├── patch_addrmsnorm_invariance.py     # Fix 2: RMSNorm decomposition
│   ├── patch_attention_invariance.py      # Fix 3: per-sequence attention
│   └── patch_attention_v1_legacy.py       # Earlier attention patch version
├── tests/
│   ├── test_gsm8k_2048_save.py            # Main test: 16 prompts, 2048 tokens, saves responses
│   ├── test_gsm8k_2048.py                 # Same test without saving
│   ├── test_comprehensive.py              # Quick test: 16 prompts, 32 tokens
│   ├── test_vllm_matmul_fix.py            # 8 prompts, 16 tokens
│   ├── test_quick_maxseqs1.py             # max_num_seqs=1 baseline
│   ├── test_op_invariance.py              # Operator-level M-dependence tests
│   ├── test_addrmsnorm_boundary.py        # npu_add_rms_norm boundary finder
│   └── test_fix_strategies.py             # Matmul fix strategy benchmarks
├── results/
│   ├── singles_responses.json             # TP=1 single-prompt responses
│   ├── batch_responses.json               # TP=1 batched responses
│   ├── comparison_summary.json            # TP=1 per-prompt comparison
│   └── tp4/                               # TP=4 results (max_num_seqs=1)
│       ├── singles_responses.json
│       ├── batch_responses.json
│       └── comparison_summary.json
└── archive/                               # Earlier experiment scripts
```

## Setup

| Item | Details |
|---|---|
| Server | `root@7.150.11.210`, container `verl-npu-bruceli` |
| Accelerator | Ascend 910 NPU (64 GB HBM) |
| Model | Qwen3-4B, bfloat16 |
| vLLM | 0.11.0 + vllm_ascend plugin |
| Inference | `LLM.generate`, greedy (temperature=0.0), enforce_eager=True |
