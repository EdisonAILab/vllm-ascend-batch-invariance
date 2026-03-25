# GSM8K Batch Invariance Results — Qwen3-4B on Ascend NPU

## Setup

| Item | Details |
|---|---|
| Machine | `root@7.150.11.210` |
| Container | `verl-npu-bruceli` |
| Accelerator | Ascend 910 NPU (NPU 0 via `ASCEND_RT_VISIBLE_DEVICES=0`) |
| Model | Qwen3-4B (`/home/bruceli/models/Qwen/Qwen3-4B`), bfloat16 |
| Dataset | GSM8K test set (1319 math word problems) |
| Sequence length | 64 tokens (pad/truncate) |
| Batch size | 8 |
| Test script | `test_gsm8k_batch_invariance.py` |

## Methodology

1. Load all 1319 GSM8K test questions.
2. Tokenize and pad/truncate each to 64 tokens.
3. For each batch of 8 prompts:
   - Run all 8 together as a batch (`[8, 64, H]`) → batched logits.
   - Run each prompt individually (`[1, 64, H]`) → single logits.
   - Compare `logits_single[0]` vs `logits_batch[i]` using max absolute difference.
4. A diff of exactly `0.0` means bit-exact batch invariance for that prompt.

## Results

| Mode | Mismatches | Max Diff | Avg Diff | Batch Invariant | Time |
|---|---|---|---|---|---|
| Native NPU ops (no fix) | **1319/1319** | 7.84375 | 1.07939 | **No** | 58.1s |
| Lightweight per-sample fix | **0/1319** | 0.00000 | 0.00000 | **Yes** | 78.1s |

### Native NPU ops (no fix)

Every single one of the 1319 prompts shows a mismatch between batched and single
forward passes. The max logit difference reaches **7.84** — significantly larger than
the 0.53 observed with the original 4 short prompts (32 tokens), because GSM8K
questions are longer and numerical differences accumulate through more transformer
layers.

```
Prompt 0:  max_diff=2.03125000  MISMATCH
Prompt 1:  max_diff=1.28125000  MISMATCH
Prompt 2:  max_diff=0.74218750  MISMATCH
...
Prompt 7:  max_diff=3.90625000  MISMATCH
...
--> Mismatches: 1319/1319
--> Max diff: 7.84375000
--> Avg diff: 1.07939401
```

### Lightweight per-sample fix

All 1319 prompts achieve **bit-exact batch invariance** (max_diff = 0.0). The fix
adds ~1.3x time overhead (78s vs 58s), which is negligible compared to the ~500x
overhead of the Triton persistent kernel approach.

```
Prompt 0:  max_diff=0.00000000  OK
Prompt 1:  max_diff=0.00000000  OK
...
Prompt 1318: max_diff=0.00000000  OK
--> Mismatches: 0/1319
--> Max diff: 0.00000000
--> Avg diff: 0.00000000
```

## Root Cause

NPU's CANN matmul selects different algorithms (gemv vs gemm) depending on the M
dimension (total rows = batch_size × seq_len). When `F.linear` is called with
`[1, 64, 2560]` (M=64) vs `[8, 64, 2560]` (M=512), different internal kernels
produce numerically different results for the same row of data.

The affected operators in Qwen3-4B (hidden_size=2560, num_kv_heads=8):

| Operator | Shape | Batch Invariant |
|---|---|---|
| `self_attn.k_proj` | 2560 → 1024 | **No** (every layer) |
| `self_attn.v_proj` | 2560 → 1024 | **No** (every layer) |
| `self_attn.q_proj` | 2560 → 4096 | Yes |
| `mlp.gate_proj` | 2560 → 9728 | Yes |
| `mlp.up_proj` | 2560 → 9728 | Yes |
| `mlp.down_proj` | 9728 → 2560 | Yes |
| `input_layernorm` | RMSNorm | Yes |
| `log_softmax` | — | Yes |
| `mean` | — | Yes |

## Fix: Per-Sample `torch.mm` (`fix_batch_invariance_npu.py`)

The fix patches `aten::linear` on NPU's `PrivateUse1` dispatch key to split the
batch dimension and process each sample independently via `torch.mm`. This ensures
every matmul call sees the same M dimension regardless of batch size.

```python
from fix_batch_invariance_npu import enable_npu_batch_invariant_linear

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("npu:0")
enable_npu_batch_invariant_linear()
# All F.linear calls are now batch-invariant
```

### Performance comparison

| Approach | Batch Invariant | Overhead | Mechanism |
|---|---|---|---|
| Native (no fix) | No | — | NPU selects different algorithms per M |
| **Lightweight fix** | **Yes** | **~1.3x** | Per-sample `torch.mm` via `aten::linear` patch |
| Triton persistent kernel | Yes | ~500x | Triton JIT kernel on NPU BiSheng backend |

The lightweight fix uses native CANN ops (no Triton), making it practical for
production use. The Triton approach is useful for validation but impractical due
to ~500x performance overhead on the Ascend Triton 3.2.0 backend.

---

## vLLM Results (GSM8K, TP=1)

### Setup

| Item | Details |
|---|---|
| vLLM | 0.11.0 with `vllm_ascend` platform plugin |
| Inference | Offline `LLM.generate`, greedy (`temperature=0.0`), `enforce_eager=True` |
| Prompts | 16 GSM8K test questions |
| Max tokens | 32 generated tokens per prompt |
| Test script | `test_vllm_gsm8k_batch_invariance_v2.py` |

### Methodology

1. Run each prompt individually (16 single-item batches) → single outputs.
2. Run all 16 prompts together in one batch → batched outputs.
3. Compare generated token IDs and logprobs between single and batched runs.

### Results

| Mode | Token Failures | Logprob Failures | Max Logprob Diff | Batch Invariant | Time (singles/batch) |
|---|---|---|---|---|---|
| Native (default) | 4/16 | 16/16 | 1.37475676 | **No** | 11.8s / 1.2s |
| Attention per-seq fix (on-disk) | 3/4* | 3/4* | 0.41310936 | **No** | — |
| **`max_num_seqs=1`** | **0/16** | **0/16** | **0.00000000** | **Yes** | 11.8s / 12.1s |

*Tested with 4 prompts / 4 tokens for the attention fix in isolation.

### Root Cause Analysis

The vLLM non-invariance has **two layers of root causes**:

#### 1. Attention kernel mismatch (scheduler-level)

When multiple sequences are batched, vLLM's V1 scheduler makes different scheduling
decisions than when processing sequences individually:

- **Single prompt**: Prefill → `PrefillNoCache` → `_npu_flash_attention`, Decode → `DecodeOnly` → `_npu_paged_attention`
- **Batch of N prompts**: Scheduler chunks across steps. Step 1 prefills some prompts.
  Step 2 mixes decode + new prefills → `ChunkedPrefill` → `npu_fused_infer_attention_score`

These are **three different NPU kernels** (`_npu_flash_attention`, `_npu_paged_attention`,
`npu_fused_infer_attention_score`) that produce numerically different results for the
same mathematical operation. This accounts for ~40% of the observed non-invariance.

The on-disk attention fix (`patch_v3.py`) splits `ChunkedPrefill` batches into
per-sequence calls using the same kernel as single-prompt mode, reducing max logprob
diff from 0.685 to 0.413.

#### 2. Matmul M-dimension sensitivity (model-level)

vLLM packs all tokens from all sequences into a single `[total_tokens, hidden_size]`
tensor and processes them through the model's linear projections together. CANN matmul
selects different algorithms based on the M dimension (total_tokens), producing
different results for the same token when it appears in different batch sizes.

- **Single prompt** (65 tokens): `F.linear` with M=65
- **Batch of 4 prompts** (119 tokens): `F.linear` with M=119

This produces different Q/K/V values from the same input, which propagates through
all subsequent layers.

#### 3. Not caused by prefix caching

Disabling prefix caching (`enable_prefix_caching=False`) and chunked prefill
(`enable_chunked_prefill=False`) had no effect, as the scheduler still chunks
multiple sequences across scheduling steps.

### Fix: `max_num_seqs=1`

Setting `max_num_seqs=1` forces vLLM's scheduler to process at most one sequence
per scheduling step. This ensures:

1. Every prefill step uses `PrefillNoCache` with exactly one sequence
2. Every decode step uses `DecodeOnly` with exactly one sequence
3. The model's linear projections always see M=seq_len (same as single-prompt mode)
4. The same NPU kernels and CANN algorithms are used regardless of batch composition

```python
from vllm import LLM, SamplingParams

llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
          max_num_seqs=1)  # <-- batch invariant mode
# All prompts now produce bit-exact identical results
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
```

### Performance

| Mode | Batch Invariant | Singles (16 prompts) | Batch (16 prompts) | Overhead |
|---|---|---|---|---|
| Default (`max_num_seqs=256`) | No | 11.8s | 1.2s | — |
| **`max_num_seqs=1`** | **Yes** | 11.8s | 12.1s | ~10x for batch |

The overhead is significant for batch inference (sequences are processed sequentially),
but singles performance is unchanged. This is the price of bit-exact batch invariance
on NPU: the CANN matmul's M-dependent algorithm selection cannot be overridden, so
sequences must be processed individually.

### Conclusion

| Framework | Fix | Batch Invariant | Overhead |
|---|---|---|---|
| **HuggingFace** | `aten::linear` per-sample patch | **Yes** (0/1319 mismatches) | ~1.3x |
| **vLLM** | `max_num_seqs=1` | **Yes** (0/16 mismatches) | ~10x batch |

Both fixes address the same root cause: CANN's M-dimension-dependent algorithm
selection in matmul. The HuggingFace fix can split 3D tensors `[B, seq, H]` into
per-sample 2D `[seq, H]` calls. vLLM uses 2D packed tensors `[total_tokens, H]`,
so the only way to ensure consistent M is to process one sequence at a time.
