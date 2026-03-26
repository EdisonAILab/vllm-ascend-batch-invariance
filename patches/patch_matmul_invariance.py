"""
Patch vLLM's unquantized gemm for batch-invariant matmul on NPU.

Modifies /vllm/vllm/model_executor/layers/utils.py to add an NPU path
that chunks the M dimension to a fixed size (128), ensuring every F.linear
call sees the same M regardless of batch composition.

Activated by env var: VLLM_NPU_BATCH_INVARIANT_MATMUL=1
"""
import sys
import glob
import os

UTILS_PATH = "/vllm/vllm/model_executor/layers/utils.py"
BACKUP_PATH = UTILS_PATH + ".bak_matmul"

# Restore from backup if exists (re-apply from clean state)
if os.path.exists(BACKUP_PATH):
    with open(BACKUP_PATH, "r") as f:
        content = f.read()
    print(f"Restored from backup {BACKUP_PATH}")
else:
    with open(UTILS_PATH, "r") as f:
        content = f.read()
    # Create backup
    with open(BACKUP_PATH, "w") as f:
        f.write(content)
    print(f"Backup saved to {BACKUP_PATH}")

# Add 'import os' if not present
if "\nimport os\n" not in content:
    content = content.replace("\nimport torch\n", "\nimport os\nimport torch\n")

# --- Add NPU gemm function after default_unquantized_gemm ---
NPU_GEMM = '''

# --- NPU batch-invariant matmul (chunk M to fixed size) ---
_NPU_CHUNK_M = int(os.environ.get("MATMUL_CHUNK_SIZE", "128"))

def npu_batch_invariant_gemm(layer: torch.nn.Module,
                              x: torch.Tensor,
                              weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None):
    """Batch-invariant F.linear for NPU.

    Chunks M dimension to _NPU_CHUNK_M so CANN selects same algorithm
    regardless of batch composition. Uses preallocated output to avoid
    extra memory from collecting chunks.
    """
    C = _NPU_CHUNK_M
    if x.dim() != 2:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        o2 = npu_batch_invariant_gemm(layer, x2, weight, bias)
        return o2.reshape(*shape[:-1], o2.shape[-1])
    M = x.shape[0]
    N = weight.shape[0]
    if M <= C:
        if M < C:
            x = torch.cat([x, x.new_zeros(C - M, x.shape[1])], dim=0)
        return torch.nn.functional.linear(x, weight, bias)[:M]
    # Preallocate output to avoid memory spike from collecting chunks
    out = x.new_empty(M, N)
    for s in range(0, M, C):
        e = min(s + C, M)
        chunk = x[s:e]
        n = chunk.shape[0]
        if n < C:
            chunk = torch.cat([chunk, x.new_zeros(C - n, x.shape[1])], dim=0)
        chunk_out = torch.nn.functional.linear(chunk, weight, None)
        out[s:e] = chunk_out[:n]
    if bias is not None:
        out = out + bias
    return out

'''

marker = "def default_unquantized_gemm(layer: torch.nn.Module,\n"
idx = content.find(marker)
if idx == -1:
    print("ERROR: Could not find default_unquantized_gemm")
    sys.exit(1)

# Find end of default_unquantized_gemm (next blank line after function)
func_end = content.find("\n\n", idx)
if func_end == -1:
    print("ERROR: Could not find end of default_unquantized_gemm")
    sys.exit(1)
func_end += 1

content = content[:func_end] + NPU_GEMM + content[func_end:]

# --- Modify dispatch_unquantized_gemm ---
old_dispatch = """def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm"""

new_dispatch = """def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif os.environ.get("VLLM_NPU_BATCH_INVARIANT_MATMUL", "0") == "1":
        return npu_batch_invariant_gemm
    else:
        return default_unquantized_gemm"""

if old_dispatch not in content:
    print("ERROR: Could not find dispatch_unquantized_gemm to patch")
    sys.exit(1)

content = content.replace(old_dispatch, new_dispatch)

with open(UTILS_PATH, "w") as f:
    f.write(content)

# Delete .pyc cache
for pyc in glob.glob("/vllm/vllm/model_executor/layers/__pycache__/utils*.pyc"):
    os.remove(pyc)
    print(f"Removed cached: {pyc}")

print("Patched utils.py successfully!")
print("Set VLLM_NPU_BATCH_INVARIANT_MATMUL=1 to enable.")
