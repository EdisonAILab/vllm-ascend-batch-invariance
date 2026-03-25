# Batch Invariance for vLLM on Ascend NPU (Qwen3-4B)

## What is Batch Invariance?

**Batch invariance** means a model produces bit-exact identical output for a given input sequence regardless of what other sequences are in the same batch. This matters for reproducible evaluation, deterministic RLHF reward computation, and debugging.

## Problem

On Ascend 910 NPU, vLLM produces **different outputs** for the same prompt depending on batch composition. A prompt processed alone gives different tokens/logprobs than the same prompt processed alongside other prompts. Root cause: three CANN operators select different internal algorithms based on the M dimension (total tokens across all sequences in a batch).

## Solution

Three operator-level patches that eliminate all sources of M-dependent non-determinism, activated by a single environment variable:

```bash
export VLLM_NPU_BATCH_INVARIANT_MATMUL=1
```

### Results

```
16/16 prompts: bit-exact match (tokens + logprobs)
Max logprob diff: 0.00000000
Singles: 20.0s  |  Batch: 1.9s  |  Speedup: 10.3x
```

No `max_num_seqs=1` restriction needed. Full batching performance preserved.

---

## Root Cause Analysis

### 1. `F.linear` / matmul (M-dependent at M >= 16)

CANN's matmul selects different algorithms (gemv vs gemm) based on M (number of rows). When vLLM packs multiple sequences into a single `[total_tokens, hidden_size]` tensor, different batch compositions produce different M values, triggering different CANN kernels with numerically different results.

**Evidence:**
```
F.linear M-dependence (native, no padding):
  M=  4 vs M=1: diff=0.0000000000  OK
  M= 16 vs M=1: diff=0.0039062500  MISMATCH
  M= 32 vs M=1: diff=0.0039062500  MISMATCH
  M= 64 vs M=1: diff=0.0039062500  MISMATCH
  M=128 vs M=1: diff=0.0039062500  MISMATCH
```

**Fix:** Chunk/pad the M dimension to a fixed size (128) so CANN always selects the same algorithm. See [`patch_matmul_invariance.py`](patch_matmul_invariance.py).

### 2. `npu_add_rms_norm` (M-dependent at M >= 49)

The fused add+RMSNorm operator (`torch_npu.npu_add_rms_norm`) also exhibits M-dependent algorithm selection. Notably, `npu_rms_norm` (without the fused add) is invariant.

**Evidence:**
```
npu_add_rms_norm M-dependence:
  M= 48: 0/48 rows differ   OK
  M= 64: 64/64 rows differ  MISMATCH (max_diff=0.03125)
  M=128: 128/128 rows differ MISMATCH

npu_rms_norm (without add): invariant at all M values
```

**Fix:** Decompose `npu_add_rms_norm(x, residual, weight, eps)` into separate `x = x + residual` followed by `npu_rms_norm(x, weight, eps)`. See [`patch_addrmsnorm_invariance.py`](patch_addrmsnorm_invariance.py).

### 3. Attention kernels (different kernels per scheduling state)

vLLM uses three different NPU attention kernels depending on scheduling state:
- **PrefillNoCache**: `_npu_flash_attention`
- **DecodeOnly**: `_npu_paged_attention`
- **ChunkedPrefill**: `npu_fused_infer_attention_score`

When a single prompt runs alone, it goes through PrefillNoCache then DecodeOnly. When batched, mixed scheduling states trigger ChunkedPrefill, which uses a different kernel producing different numerics.

Additionally, `_npu_flash_attention` itself shows cross-sequence interference when multiple sequences are packed together.

**Fix:** Process each sequence individually through the same kernel it would use in single-prompt mode. See [`patch_v3.py`](patch_v3.py).

---

## Patches

| File | Target | Description |
|---|---|---|
| [`patch_matmul_invariance.py`](patch_matmul_invariance.py) | `/vllm/vllm/model_executor/layers/utils.py` | Replaces `dispatch_unquantized_gemm()` with M-chunked version (chunk_size=128) |
| [`patch_addrmsnorm_invariance.py`](patch_addrmsnorm_invariance.py) | `/vllm-ascend/vllm_ascend/ops/layernorm.py` | Decomposes `npu_add_rms_norm` into `add` + `npu_rms_norm` |
| [`patch_v3.py`](patch_v3.py) | `/vllm-ascend/vllm_ascend/attention/attention_v1.py` | Per-sequence attention for prefill, decode, and chunked prefill |

### How to Apply

```bash
# Inside the container
python patch_matmul_invariance.py
python patch_addrmsnorm_invariance.py
python patch_v3.py

# Run with the fix enabled
VLLM_NPU_BATCH_INVARIANT_MATMUL=1 python your_script.py
```

Each patch creates a `.bak` backup and is idempotent (re-running restores from backup first).

---

## Operator Invariance Summary

Tested on Ascend 910 NPU with bfloat16, hidden_size=2560:

| Operator | M-Invariant? | Threshold | Fix |
|---|---|---|---|
| `F.linear` (matmul) | No | M >= 16 | Chunk M to 128 |
| `npu_add_rms_norm` | No | M >= 49 | Decompose to add + rms_norm |
| `npu_rms_norm` | Yes | - | None needed |
| `SiLU` | Yes | - | None needed |
| `torch.mul` | Yes | - | None needed |
| `_npu_flash_attention` | No* | Multi-sequence | Per-sequence processing |
| `_npu_paged_attention` | No* | Multi-sequence | Per-sequence processing |
| `npu_fused_infer_attention_score` | No* | Different kernel | Avoid; use flash/paged |

*Attention kernels are invariant when called with a single sequence but produce different results when multiple sequences are packed together or when different kernels are selected for different scheduling states.

---

## Test Scripts

| Script | Description |
|---|---|
| [`test_comprehensive.py`](test_comprehensive.py) | Full test: 16 prompts, 32 tokens, all fixes |
| [`test_vllm_matmul_fix.py`](test_vllm_matmul_fix.py) | 8 prompts, 16 tokens test |
| [`test_op_invariance.py`](test_op_invariance.py) | Isolates M-dependence of individual NPU operators |
| [`test_addrmsnorm_boundary.py`](test_addrmsnorm_boundary.py) | Finds exact M threshold for npu_add_rms_norm |
| [`test_fix_strategies.py`](test_fix_strategies.py) | Benchmarks matmul fix strategies (row-by-row, chunked, padded) |

---

## Setup

| Item | Details |
|---|---|
| Server | `root@7.150.11.210`, container `verl-npu-bruceli` |
| Accelerator | Ascend 910 NPU |
| Model | Qwen3-4B (`/home/bruceli/models/Qwen/Qwen3-4B`), bfloat16 |
| vLLM | 0.11.0 with `vllm_ascend` platform plugin |
| Inference | Offline (`LLM.generate`), greedy (`temperature=0.0`), `enforce_eager=True` |

---

## Performance

| Configuration | Time (16 prompts, 32 tokens) | Batch Invariant? |
|---|---|---|
| Native (no fix) | ~1.9s batch | No (4-8/16 failures) |
| `max_num_seqs=1` | ~20s (sequential) | Yes |
| **Operator fixes (this repo)** | **~1.9s batch** | **Yes** |

The operator-level fixes achieve batch invariance with **zero performance penalty** compared to native batching. The matmul chunking adds ~1.3x overhead per linear layer, but this is offset by retained batch parallelism.
