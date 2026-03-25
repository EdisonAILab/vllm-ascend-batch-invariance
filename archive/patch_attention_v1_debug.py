"""Add debug logging to attention_v1.py to trace code paths."""
import sys

filepath = "/vllm-ascend/vllm_ascend/attention/attention_v1.py"

with open(filepath, "r") as f:
    content = f.read()

# Add logging to _forward_prefill_no_cache
old = '        else:\n            # Process each sequence individually for batch invariance.'
new = '        else:\n            # Process each sequence individually for batch invariance.\n            import sys as _sys\n            print(f"  [BATCH_INV] _forward_prefill_no_cache: num_seqs={num_seqs}, seq_lens={seq_lens.tolist()}", flush=True, file=_sys.stderr)'

if old not in content:
    print("ERROR: Could not find marker in _forward_prefill_no_cache")
    sys.exit(1)
content = content.replace(old, new)

# Add logging to forward() to trace attn_state
old_fwd = '            # V0-Style scheduler situation.\n            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:'
new_fwd = '            # V0-Style scheduler situation.\n            import sys as _sys\n            _state_name = attn_metadata.attn_state.name\n            _n_seqs = attn_metadata.seq_lens.shape[0] if attn_metadata.seq_lens is not None else 0\n            print(f"  [ATTN] state={_state_name} num_seqs={_n_seqs} num_tokens={num_tokens}", flush=True, file=_sys.stderr)\n            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:'

if old_fwd not in content:
    print("ERROR: Could not find forward dispatch marker")
    sys.exit(1)
content = content.replace(old_fwd, new_fwd)

# Remove .pyc
import os
pyc = "/vllm-ascend/vllm_ascend/attention/__pycache__/attention_v1.cpython-311.pyc"
if os.path.exists(pyc):
    os.remove(pyc)
    print("Removed .pyc cache")

with open(filepath, "w") as f:
    f.write(content)

print("Debug logging added to attention_v1.py")
