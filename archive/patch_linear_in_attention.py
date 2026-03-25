"""Add per-sample aten::linear fix to attention_v1.py module level."""
import os, ast

filepath = "/vllm-ascend/vllm_ascend/attention/attention_v1.py"

with open(filepath, "r") as f:
    content = f.read()

# Add the per-sample linear fix at module level, right after imports
# Find the right insertion point - after the last import
insert_marker = "from ..utils import weak_ref_tensors"
assert insert_marker in content, f"Could not find insert marker: {insert_marker}"

linear_fix_code = '''

# --- Batch invariance fix: per-sample aten::linear ---
# CANN matmul selects different algorithms (gemv vs gemm) based on the M
# dimension. When batch size changes, M changes, producing different results
# for the same row. This fix processes each sample independently via torch.mm
# to ensure the same M regardless of batch composition.
import os as _os
_BATCH_INVARIANT_LINEAR_ENABLED = _os.environ.get(
    "VLLM_ASCEND_BATCH_INVARIANT_LINEAR", "1") == "1"

if _BATCH_INVARIANT_LINEAR_ENABLED:
    def _linear_per_sample(input_tensor, weight, bias=None):
        if input_tensor.dim() == 2:
            out = torch.mm(input_tensor, weight.t())
            if bias is not None:
                out = out + bias
            return out
        orig_shape = input_tensor.shape
        B = orig_shape[0]
        inner_shape = orig_shape[1:]
        K = orig_shape[-1]
        M_per_sample = 1
        for d in inner_shape[:-1]:
            M_per_sample *= d
        results = []
        for i in range(B):
            flat = input_tensor[i].reshape(M_per_sample, K)
            out = torch.mm(flat, weight.t())
            if bias is not None:
                out = out + bias
            results.append(out.reshape(*inner_shape[:-1], weight.shape[0]))
        return torch.stack(results, dim=0)

    _linear_lib = torch.library.Library("aten", "IMPL")
    _linear_lib.impl("aten::linear", _linear_per_sample, "PrivateUse1")
'''

content = content.replace(insert_marker, insert_marker + linear_fix_code)

# Remove .pyc
pyc = "/vllm-ascend/vllm_ascend/attention/__pycache__/attention_v1.cpython-311.pyc"
if os.path.exists(pyc):
    os.remove(pyc)

ast.parse(content)

with open(filepath, "w") as f:
    f.write(content)

print("Added per-sample aten::linear fix to attention_v1.py")
print("  Controlled by VLLM_ASCEND_BATCH_INVARIANT_LINEAR env var (default: enabled)")
