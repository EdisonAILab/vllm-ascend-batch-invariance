"""
Patch vllm-ascend's layernorm.py to avoid M-dependent npu_add_rms_norm.

Replaces npu_add_rms_norm with separate add + npu_rms_norm when
VLLM_NPU_BATCH_INVARIANT_MATMUL=1 (reusing the same env var).

npu_rms_norm is M-invariant, but npu_add_rms_norm is NOT (changes at M>=49).
"""
import sys
import os
import glob

LAYERNORM_PATH = "/vllm-ascend/vllm_ascend/ops/layernorm.py"
BACKUP_PATH = LAYERNORM_PATH + ".bak_invariance"

# Read file (from backup if exists for re-apply)
if os.path.exists(BACKUP_PATH):
    with open(BACKUP_PATH, "r") as f:
        content = f.read()
    print(f"Restored from backup {BACKUP_PATH}")
else:
    with open(LAYERNORM_PATH, "r") as f:
        content = f.read()
    with open(BACKUP_PATH, "w") as f:
        f.write(content)
    print(f"Backup saved to {BACKUP_PATH}")

# Add 'import os' if not present
if "\nimport os\n" not in content:
    content = "import os\n" + content

# Patch the else branch in _addrmsnorm_forward_oot that calls npu_add_rms_norm
old_code = """\
    else:
        if is_310p():
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                          self.variance_epsilon)
        else:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon)"""

new_code = """\
    else:
        if is_310p() or os.environ.get("VLLM_NPU_BATCH_INVARIANT_MATMUL", "0") == "1":
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                          self.variance_epsilon)
        else:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon)"""

if old_code not in content:
    print("ERROR: Could not find the npu_add_rms_norm code to patch")
    print("File may already be patched or has changed.")
    sys.exit(1)

content = content.replace(old_code, new_code)

with open(LAYERNORM_PATH, "w") as f:
    f.write(content)

# Clear .pyc cache
for pyc in glob.glob("/vllm-ascend/vllm_ascend/ops/__pycache__/layernorm*.pyc"):
    os.remove(pyc)
    print(f"Removed cached: {pyc}")

print("Patched layernorm.py successfully!")
print("npu_add_rms_norm replaced with add + npu_rms_norm when VLLM_NPU_BATCH_INVARIANT_MATMUL=1")
