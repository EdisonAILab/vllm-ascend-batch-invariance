# Batch Invariance Operator Analysis: Triton vs CANN vs PyTorch

## The Three Levels

```
Application (vLLM)
    ↓
PyTorch API (F.linear, torch.mm)        ← Our patches live here
    ↓
Triton / CANN kernel dispatch
    ↓
Ascend NPU hardware
```

## Upstream vLLM (CUDA) — Triton Level

The upstream `VLLM_BATCH_INVARIANT=1` framework ([PR #25603](https://github.com/vllm-project/vllm/pull/25603), [tracking issue #27433](https://github.com/vllm-project/vllm/issues/27433)) replaces ATen ops with custom Triton kernels that guarantee deterministic tile scheduling:

| Op | Triton Kernel | How It Achieves Invariance |
|---|---|---|
| `mm/addmm/linear` | `matmul_kernel_persistent` | Fixed tile-to-SM assignment: `tile_id % NUM_SMS`. Same SM always computes same output tile regardless of M |
| `bmm` | `bmm_kernel` ([PR #29345](https://github.com/vllm-project/vllm/pull/29345)) | Fixed K-iteration order with deterministic masking |
| `log_softmax` | `_log_softmax_kernel` | Per-row reduction, no cross-row dependency |
| `mean` | `mean_kernel` | Per-row reduction |
| `rms_norm` | `_rms_norm_kernel` | Per-row normalization |
| `fused_add_rms_norm` | C++ kernel | Disables 8-wide vectorized path that has M-dependent behavior |
| `topkGatingSoftmax` | C++ kernel | Forces `num_warps=32` instead of M-dependent calculation |

Key design: each Triton kernel controls its own tiling/scheduling, making the computation order a pure function of `tile_id` and `NUM_SMS` — independent of M.

Performance: ~1.6x overhead on CUDA (Hopper/Blackwell). Optimized BMM kernel recovered 18.1% throughput.

Reference: [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

## Ascend NPU — CANN Level (Black Box)

On Ascend, there are no Triton replacements. All compute goes through CANN native ops which are closed-source:

| CANN Op | M-Invariant? | Root Cause of Non-Invariance |
|---|---|---|
| CANN `matmul` | No (M >= 16) | Selects gemv vs gemm based on M |
| `npu_add_rms_norm` | No (M >= 49) | Internal algorithm switch at M threshold |
| `npu_rms_norm` | Yes | — |
| `_npu_flash_attention` | No (multi-seq packing) | Cross-sequence interference |
| `_npu_paged_attention` | No (multi-seq) | Cross-sequence interference |
| `npu_fused_infer_attention_score` | No | Different kernel than flash/paged |
| HCCL `allreduce` | No (M >= 412) | Internal buffer/algorithm threshold (~2 MB) |
| SiLU, mul, embedding, RoPE | Yes | Element-wise or per-row, no M dependency |
| `log_softmax` | Yes | Per-row reduction |

We cannot modify these — CANN is closed-source.

## Our Approach — PyTorch Level (Wrappers)

Since we can't fix the kernels, we control what shapes they see:

| Patch | Strategy | What CANN Sees |
|---|---|---|
| Matmul chunking | Chunk M to 128 | Always `[128, K]` input → same algorithm selected |
| RMSNorm decomposition | Replace fused with two separate ops | `npu_rms_norm` which is already invariant |
| Per-sequence attention | Loop over sequences | Always 1 sequence → same kernel path |
| Allreduce fixed-chunk | Pad each chunk to exactly 384 rows | Always `[384, H]` → same HCCL algorithm |

All patches are pure Python. No Triton kernels, no Ascend C code.

## Comparison

|  | Triton (upstream CUDA) | CANN (Ascend native) | PyTorch wrapper (our fix) |
|---|---|---|---|
| **Overhead** | ~1.6x | 0x (but non-invariant) | ~1.3x (TP=1), near-zero (TP=4) |
| **Correctness** | Guaranteed by kernel design | M-dependent | Guaranteed by shape control |
| **Graph capture** | Yes (torch.compile compatible) | Yes | No (requires `enforce_eager`) |
| **Model coverage** | All ATen ops automatically | — | Only tested operators |
| **Fragility** | Low (kernel-level guarantee) | — | High (bypassed by new code paths) |
| **Portability** | CUDA SM80+ only | Ascend only | Any backend |
| **Maintenance** | Maintained by vLLM upstream | Huawei | Manual per-model validation |

## Ascend-Specific Triton Kernels

Three Triton kernels exist in vllm-ascend but are for specialized architectures (not used by Qwen3 or standard transformers):

| File | Model Type | Batch-Invariant? | Analysis |
|---|---|---|---|
| `casual_conv1d.py` | Mamba/SSM | Yes | Grid: `(seq, dim)`, no M dependency |
| `fla.py` | Flash Linear Attention | Yes | Grid: `(row, group)`, per-row processing |
| `sigmoid_gating.py` | Gated Delta Rule | Likely yes | Grid: `(k, v, nh)`, sequential T processing |

## Why Triton Is Not Viable on Ascend (Currently)

The upstream `batch_invariant_ops` Triton persistent matmul was tested on Ascend NPU:

| Mode | Speed (single prompt, 8 tokens) |
|---|---|
| Native CANN | ~2 seconds |
| Triton persistent kernel | ~17 minutes (~500x slower) |

The Ascend Triton 3.2.0 backend has severe limitations:
- No `int64` index support (required workarounds)
- `hivm.hir.vcast` errors for in-kernel dtype conversion
- No `flatten` parameter in `tl.range()`
- ~500x slower than native CANN for matmul

Until Huawei's Triton backend matures significantly, Triton-level batch invariance is not practical on Ascend.

## Gaps for Future Work

| Gap | Impact | Needed For |
|---|---|---|
| **MoE gating** (`topkGatingSoftmax`) | May be M-dependent | DeepSeek-V3, Mixtral |
| **Fused MoE kernels** | Unknown on Ascend | DeepSeek-V3 |
| **W8A8 quantized matmul** | Bypasses our `dispatch_unquantized_gemm` | Ascend-quantized models |
| **ACLGraph compatibility** | Can't capture Python chunking loops | Production deployments |
| **MLA attention** | Not tested on Ascend | DeepSeek-V3 |
| **Mamba/SSM ops** | `casual_conv1d` Triton kernel untested | Mamba-based models |
| **CANN deterministic flag** | Would eliminate all PyTorch wrappers | All models, zero overhead |

## The Ideal Fix

A CANN-level solution would be a flag like:
```bash
export CANN_DETERMINISTIC_TILING=1
```
This would force all CANN kernels to use fixed internal tiling regardless of M — zero overhead, zero fragility, all models covered automatically. Until Huawei provides this, the PyTorch wrapper approach is the practical solution.
