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
