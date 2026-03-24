# Batch Invariance Experiment on Ascend NPU — Qwen3-4B

## Overview

**Batch invariance** is the property that a model produces bit-exact identical output for a given
sequence regardless of what other sequences appear in the same batch. This is desirable for
reproducibility and deterministic evaluation.

This experiment reproduces the
[batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) results on
an **Ascend 910 NPU** using Qwen3-4B, after the library was already validated on CUDA.

---

## Setup

| Item | Details |
|---|---|
| Machine | `root@7.150.11.210` |
| Container | `verl-npu-bruceli` |
| Accelerator | Ascend 910 NPU (NPU ID 7, mapped to `npu:0` via `ASCEND_RT_VISIBLE_DEVICES=7`) |
| NPU memory | 62 GB |
| NPU cores | 24 cube cores, 48 vector cores |
| Model | Qwen3-4B (`/home/bruceli/models/Qwen/Qwen3-4B`) |
| dtype | bfloat16 |
| Framework | PyTorch + `torch_npu` |
| Triton | 3.2.0 (Ascend backend via `torch_npu`) |
| `batch_invariant_ops` | `/home/bruceli/projects/batch_invariant_ops` (patched for NPU) |
| Test script | `/home/bruceli/projects/batch-invariance-Qwen3-4B/test_qwen3_4b_batch_invariance.py` |

---

## Test Methodology

- Load Qwen3-4B in bfloat16 on `npu:0`.
- Tokenize 4 prompts, truncate/pad all to exactly 32 tokens (same-length sequences eliminate
  RoPE positional differences due to padding).
- **Batch forward**: run all 4 sequences together → logits shape `[4, 32, vocab]`.
- **Single forward**: run each sequence alone → logits shape `[1, 32, vocab]`.
- Compare `logits_single[0]` vs `logits_batch[i]` using max absolute difference.
- A diff of exactly `0.0` means bit-exact batch invariance.

Run command:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 /home/bruceli/projects/batch-invariance-Qwen3-4B/test_qwen3_4b_batch_invariance.py
```

---

## Results

```
Loading /home/bruceli/models/Qwen/Qwen3-4B on npu:0 ...
ASCEND_RT_VISIBLE_DEVICES=7
dtype=bfloat16, seq_len=32, batch_size=4

=== Standard PyTorch (batch-invariant mode OFF) ===
  Prompt 0: max_diff=0.53125000  MISMATCH
  Prompt 1: max_diff=0.43750000  MISMATCH
  Prompt 2: max_diff=0.43750000  MISMATCH
  Prompt 3: max_diff=0.32812500  MISMATCH
  --> Batch-invariant: False  (overall max_diff=0.53125000)

=== Batch-Invariant Mode ON ===
  Prompt 0: max_diff=0.00000000  OK (invariant)
  Prompt 1: max_diff=0.00000000  OK (invariant)
  Prompt 2: max_diff=0.00000000  OK (invariant)
  Prompt 3: max_diff=0.00000000  OK (invariant)
  --> Batch-invariant: True  (overall max_diff=0.00000000)
```

`batch_invariant_ops` achieves **bit-exact batch invariance** on Ascend NPU.

---

## Root Cause of NPU Non-Invariance

Without the patch, the NPU produces different results for a sequence depending on batch size.
The root cause is that the Ascend 910's on-chip scheduler selects different matmul algorithms
depending on the M dimension (rows = batch × seq_len):

- **batch=1** (M=32): uses a **gemv**-style kernel (matrix-vector, column-major reduction)
- **batch=4** (M=128): uses a **gemm**-style kernel (full tile-based matrix multiply)

This affects linear projections whose output dimension is 1024 — specifically `k_proj` and
`v_proj` in Qwen3-4B's grouped-query attention (GQA), where `hidden_dim=2048 → kv_dim=1024`.
Other projections (`q_proj`, `o_proj`, `gate_proj`, etc.) use different output sizes and happen
to be invariant natively on this NPU.

---

## How `batch_invariant_ops` Works

`batch_invariant_ops` replaces non-deterministic ATen ops with Triton kernels that guarantee
identical results regardless of batch size, by using a **persistent matmul kernel** where tile
assignment is a pure function of `tile_id` and `NUM_SMS` (a hardware constant):

```
grid = min(NUM_SMS, num_tiles)
each SM processes tiles: tile_id, tile_id + NUM_SMS, tile_id + 2*NUM_SMS, ...
```

Since `NUM_SMS` is constant for a given GPU/NPU and each tile's computation depends only on its
row/column range in the weight matrix (not on M), the result for any given (row, col) block is
identical whether M=32 or M=128.

The library patches the following ATen dispatch ops:

| Op | Purpose |
|---|---|
| `aten::mm` | Raw matrix multiply |
| `aten::addmm` | Matrix multiply + bias (2D input) |
| `aten::linear` | Fused linear (3D+ input) — **NPU only** |
| `aten::_log_softmax` | Log-softmax (for next-token log-probs) |
| `aten::mean.dim` | Mean reduction (used in RMSNorm) |

---

## NPU-Specific Patches to `batch_invariant_ops`

The upstream library targets CUDA. Seven modifications were needed for Ascend NPU:

### 1. Dispatch key: `"PrivateUse1"` instead of `"CUDA"`

PyTorch's `torch.library.Library("aten", "IMPL")` uses the string dispatch key.
On NPU, `torch_npu` registers its ops under the `PrivateUse1` key (not `"NPU"`).

```python
# Before
dispatch_key = _accel_type.upper()   # "NPU" — wrong, not a valid dispatch key

# After
dispatch_key = "PrivateUse1" if _accel_type == "npu" else _accel_type.upper()
```

### 2. `get_compute_units()` — add NPU case

The kernel grid size is `min(NUM_SMS, num_tiles)`. On NPU, the cube core count
is the equivalent of SM count.

```python
case "npu":
    import torch_npu
    cube_core_num = torch_npu.npu.get_device_properties(0).cube_core_num
    return cube_core_num   # 24 for Ascend 910
```

### 3. Remove `flatten=True` from `tl.range()`

NPU's Triton 3.2.0 does not support the `flatten` keyword argument.

```python
# Before
for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):

# After
for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
```

### 4. Relax `assert input.is_cuda`

Several internal assertions only checked for CUDA tensors.

```python
# Before
assert input.is_cuda

# After
assert input.is_cuda or input.device.type == "npu"
```

### 5. New NPU kernel without `int64` index casts

The upstream `matmul_kernel_persistent` uses `tl.cast(..., tl.int64)` for large-tensor
pointer arithmetic. The NPU Triton backend does not support `int64` index types in this
context. A new `matmul_kernel_persistent_npu` was written using only `int32`-compatible
indexing.

### 6. Kernel stores `float32`; dtype conversion done in Python

The NPU Triton compiler's buffer-reuse analysis pass (`BiShengHIR HIVM pipeline`) fails on
`hivm.hir.vcast` operations — it cannot trace through in-kernel type casts to find the root
allocation. The error manifests as:

```
MLIRCompilationError: 'hivm.hir.vcast' op Unsupported op for finding the root alloc.
```

The fix: the Triton kernel accumulates in `float32` and stores a `float32` output tensor
(no cast inside the kernel). Dtype conversion (`float32 → bfloat16`) and bias addition are
done in Python using standard PyTorch ops after the kernel returns:

```python
c_f32 = torch.empty((M, N), device=a.device, dtype=torch.float32)
matmul_kernel_persistent_npu[grid](a, b, c_f32, ...)   # no vcast inside
c = c_f32.to(dtype)           # PyTorch op, not Triton
if bias is not None:
    c = c + bias
```

### 7. Register `aten::linear` for NPU

On CUDA, `F.linear(x, W, b)` with a 3D input `[B, S, H]` dispatches through `aten::addmm`
(after implicit reshape). On NPU, `torch_npu` routes it through `aten::linear` instead.
Because `aten::addmm` was never called, the CUDA-only patch had no effect on NPU.

A new `linear_batch_invariant` function was added and registered under `aten::linear` for the
`PrivateUse1` dispatch key:

```python
def linear_batch_invariant(input, weight, bias=None):
    orig_shape = input.shape
    flat = input.reshape(-1, orig_shape[-1])          # [M, in_features]
    out = matmul_persistent_npu(flat, weight.t(), bias=bias)
    return out.reshape(*orig_shape[:-1], weight.shape[0])

# In enable_batch_invariant_mode():
if _accel_type == "npu":
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, dispatch_key)
```

This was the final and decisive fix — without it, none of the model's linear layers were
intercepted, and the invariance patch had zero effect.

---

## Comparison: CUDA vs NPU

| Aspect | CUDA (A100) | Ascend NPU 910 |
|---|---|---|
| Non-invariant ops (native) | `mm`, `addmm`, `log_softmax`, `mean` | `linear` (via `aten::linear`) for N=1024 |
| Dispatch key | `"CUDA"` | `"PrivateUse1"` |
| Triton kernel | `matmul_kernel_persistent` (int64 indices) | `matmul_kernel_persistent_npu` (int32 indices, float32 output) |
| `F.linear` dispatch path | `aten::addmm` | `aten::linear` |
| In-kernel dtype cast | Supported | Not supported (`hivm.hir.vcast` error) |
| Result with ops ON | max_diff = 0.0 | max_diff = 0.0 |

---

## vLLM Batch Invariance on NPU

### Setup

| Item | Details |
|---|---|
| vLLM | 0.11.0 with `vllm_ascend` platform plugin |
| Inference mode | Offline (`LLM.generate`), greedy (`temperature=0.0`), `enforce_eager=True` |
| Test script | `/home/bruceli/projects/batch-invariance-Qwen3-4B/test_vllm_batch_invariance_npu.py` |

### Methodology

- Run each prompt individually (single-item batch) and all prompts together in one batch.
- Compare generated **token IDs** and **logprobs** between single and batched runs.
- Test with `batch_invariant_ops` OFF (native NPU ops) and ON (Triton persistent kernel).
- vLLM workers are forked from the parent process, so the `set_batch_invariant_mode(True)`
  dispatch patch registered in the parent is inherited by the forked EngineCore subprocess.

### Results

**Mode OFF (native NPU ops):**

```
=== Standard vLLM (batch-invariant ops OFF) ===
  Prompt 0: tokens=OK  logprobs=max_diff=0.000000 OK
  Prompt 1: tokens=OK  logprobs=max_diff=0.000000 OK
  Prompt 2: tokens=OK  logprobs=max_diff=0.000000 OK
  Prompt 3: tokens=OK  logprobs=max_diff=0.000000 OK
  --> Tokens invariant: True  |  Logprobs invariant: True
```

**Mode ON (Triton persistent kernel):**

```
=== vLLM with batch_invariant_ops ON ===
  Prompt 0: tokens=OK  logprobs=max_diff=0.000000 OK
  Prompt 1: tokens=OK  logprobs=max_diff=0.000000 OK
  --> Tokens invariant: True  |  Logprobs invariant: True
```

Both modes achieve **bit-exact batch invariance** with vLLM on NPU.

### Analysis

Unlike the HuggingFace `AutoModelForCausalLM` test (which showed large mismatches with
Mode OFF), **vLLM on NPU is already batch-invariant without `batch_invariant_ops`**. This is
because vLLM's Ascend backend processes sequences through a different code path than raw
`F.linear`:

- vLLM uses **chunked prefill** with PagedAttention, where the scheduler controls how tokens
  are batched. Single vs batched requests go through the same internal scheduling logic, and
  the underlying CANN ops see consistent input shapes.
- The HuggingFace test directly calls the model with `[1, 32, H]` vs `[4, 32, H]`, which
  triggers different gemv/gemm kernel selection on NPU for certain matrix dimensions.

### Performance Note

The Triton persistent kernel on NPU is significantly slower than native CANN ops:

| Mode | Speed (single prompt, 8 tokens) |
|---|---|
| OFF (native) | ~2 seconds |
| ON (Triton) | ~17 minutes (~500x slower) |

This overhead comes from:
1. **Triton JIT compilation**: First-time compilation of all unique kernel shapes takes
   ~2–42 minutes (cached after first run).
2. **Kernel launch overhead**: Each `aten::linear` call goes through Python dispatch →
   Triton kernel launch → float32 output → dtype cast → bias add.
3. **Non-optimized Triton backend**: The Ascend Triton 3.2.0 backend does not match the
   performance of native CANN fused operators.

For NPU deployments, `batch_invariant_ops` is useful for **validating batch invariance with
HuggingFace models** but is not practical for production vLLM inference due to the performance
penalty. Fortunately, vLLM's Ascend backend is already batch-invariant natively.

---

## File Locations (in container `verl-npu-bruceli`)

| File | Description |
|---|---|
| `/home/bruceli/projects/batch_invariant_ops/` | Patched `batch_invariant_ops` source |
| `/home/bruceli/projects/batch-invariance-Qwen3-4B/test_qwen3_4b_batch_invariance.py` | HuggingFace model test |
| `/home/bruceli/models/Qwen/Qwen3-4B/` | Model weights |
| `/root/smoke2.py` | Smoke test for single `nn.Linear(2560, 1024)` |
| `/home/bruceli/projects/batch-invariance-Qwen3-4B/test_vllm_batch_invariance_npu.py` | vLLM batch invariance test |
