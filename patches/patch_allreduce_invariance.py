"""
Patch vLLM's tensor_model_parallel_all_reduce for batch invariance on NPU.

HCCL allreduce is M-dependent at M>=412 rows (with hidden_size=2560, 4 ranks).
This produces different per-row results for different total tensor sizes.

Fix: chunk the allreduce to <=384 rows when VLLM_NPU_BATCH_INVARIANT_MATMUL=1.
"""
import os
import sys
import glob

# The allreduce is in vllm's parallel_state or distributed module
# tensor_model_parallel_all_reduce calls pynccl or torch.distributed.all_reduce
# We need to intercept at the vLLM level

# Find where tensor_model_parallel_all_reduce is defined
PARALLEL_STATE_PATH = "/vllm/vllm/distributed/communication_op.py"
BACKUP_PATH = PARALLEL_STATE_PATH + ".bak_allreduce"

# Remove stale backup from wrong path
import os as _os2
stale = "/vllm/vllm/distributed/parallel_state.py.bak_allreduce"
if _os2.path.exists(stale):
    _os2.remove(stale)

if os.path.exists(BACKUP_PATH):
    with open(BACKUP_PATH, "r") as f:
        content = f.read()
    print("Restored from backup %s" % BACKUP_PATH)
else:
    with open(PARALLEL_STATE_PATH, "r") as f:
        content = f.read()
    with open(BACKUP_PATH, "w") as f:
        f.write(content)
    print("Backup saved to %s" % BACKUP_PATH)

# Find tensor_model_parallel_all_reduce function
old_func = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)'''

new_func = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    When VLLM_NPU_BATCH_INVARIANT_MATMUL=1, processes allreduce in fixed-size
    chunks to ensure HCCL always sees the same tensor size regardless of M.
    Each chunk is padded to exactly _ALLREDUCE_PAD rows, so the HCCL algorithm
    selection is identical for every call. This makes the per-row allreduce
    result independent of how many total rows are in the batch.
    """
    import os as _os
    if (_os.environ.get("VLLM_NPU_BATCH_INVARIANT_MATMUL", "0") == "1"
            and input_.dim() == 2
            and get_tp_group().world_size > 1):
        import torch as _torch
        _ALLREDUCE_PAD = 384  # Fixed size: always below HCCL threshold (412)
        M, H = input_.shape
        for s in range(0, M, _ALLREDUCE_PAD):
            e = min(s + _ALLREDUCE_PAD, M)
            n = e - s
            if n == _ALLREDUCE_PAD:
                # Full chunk: allreduce in-place
                input_[s:e] = get_tp_group().all_reduce(input_[s:e])
            else:
                # Partial chunk: pad to fixed size, allreduce, extract
                padded = _torch.zeros(_ALLREDUCE_PAD, H, dtype=input_.dtype, device=input_.device)
                padded[:n] = input_[s:e]
                padded = get_tp_group().all_reduce(padded)
                input_[s:e] = padded[:n]
        return input_
    return get_tp_group().all_reduce(input_)'''

if old_func not in content:
    print("ERROR: Could not find tensor_model_parallel_all_reduce")
    sys.exit(1)

content = content.replace(old_func, new_func)

with open(PARALLEL_STATE_PATH, "w") as f:
    f.write(content)

# Clear pyc
for pyc in glob.glob("/vllm/vllm/distributed/__pycache__/parallel_state*.pyc"):
    os.remove(pyc)
    print("Removed cached: %s" % pyc)

print("Patched tensor_model_parallel_all_reduce with chunked allreduce (chunk=384)")
