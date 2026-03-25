"""
Lightweight batch-invariance fix for Ascend NPU.

Root cause: NPU's CANN matmul selects different algorithms (gemv vs gemm) depending
on the M dimension (rows). When batch_size=1 (M=seq_len) vs batch_size=4 (M=4*seq_len),
different algorithms produce numerically different results for the same row of data.
This affects aten::linear AND aten::mm on NPU.

The affected operators in Qwen3-4B are:
  - self_attn.k_proj (2560->1024) in every layer
  - self_attn.v_proj (2560->1024) in every layer

Fix: Patch aten::linear to split the batch dimension and process each sample
independently with torch.mm. This ensures every call uses the same M dimension
(seq_len), producing identical results regardless of batch size. Uses native CANN
ops (no Triton), so overhead is minimal (~B sequential calls instead of 1 batched).

Usage:
    from fix_batch_invariance_npu import enable_npu_batch_invariant_linear

    model = AutoModelForCausalLM.from_pretrained(...).to("npu:0")
    enable_npu_batch_invariant_linear()
    # Now all F.linear calls are batch-invariant
"""
import contextlib
import torch

__all__ = [
    "enable_npu_batch_invariant_linear",
    "disable_npu_batch_invariant_linear",
    "npu_batch_invariant_mode",
]

_lib = None
_enabled = False


def _linear_per_sample(input, weight, bias=None):
    """Batch-invariant F.linear: split batch, process each sample with same M via torch.mm."""
    if input.dim() == 2:
        # 2D input [M, K]: can't split by batch, just use mm directly
        out = torch.mm(input, weight.t())
        if bias is not None:
            out = out + bias
        return out

    # 3D+ input [B, S, K] or [B, ..., K]: split on dim 0, process each with same M
    orig_shape = input.shape
    B = orig_shape[0]
    inner_shape = orig_shape[1:]  # [S, K] or [..., K]
    K = orig_shape[-1]
    M_per_sample = 1
    for d in inner_shape[:-1]:
        M_per_sample *= d

    results = []
    for i in range(B):
        flat = input[i].reshape(M_per_sample, K)  # [M_per_sample, K]
        out = torch.mm(flat, weight.t())  # [M_per_sample, out_features]
        if bias is not None:
            out = out + bias
        results.append(out.reshape(*inner_shape[:-1], weight.shape[0]))

    return torch.stack(results, dim=0)


def enable_npu_batch_invariant_linear():
    """Patch aten::linear on NPU to process each batch element independently."""
    global _lib, _enabled
    if _enabled:
        return
    _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("aten::linear", _linear_per_sample, "PrivateUse1")
    _enabled = True


def disable_npu_batch_invariant_linear():
    """Remove the aten::linear patch."""
    global _lib, _enabled
    if _lib is not None:
        _lib._destroy()
    _lib = None
    _enabled = False


@contextlib.contextmanager
def npu_batch_invariant_mode():
    """Context manager that enables batch-invariant linear on NPU."""
    enable_npu_batch_invariant_linear()
    try:
        yield
    finally:
        disable_npu_batch_invariant_linear()
