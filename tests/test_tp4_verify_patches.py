"""Verify all patches are active in TP=4 workers by hooking key functions."""
import os, json
os.environ["VLLM_NPU_BATCH_INVARIANT_MATMUL"] = "1"
os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

# Hook dispatch_unquantized_gemm BEFORE importing vllm
import vllm.model_executor.layers.utils as utils_mod
original_dispatch = utils_mod.dispatch_unquantized_gemm

_gemm_m_values = []

def hooked_dispatch():
    fn = original_dispatch()
    fn_name = fn.__name__

    if fn_name == "npu_batch_invariant_gemm":
        original_gemm = fn
        def tracked_gemm(layer, x, weight, bias=None):
            M = x.shape[0] if x.dim() == 2 else x.reshape(-1, x.shape[-1]).shape[0]
            _gemm_m_values.append(M)
            return original_gemm(layer, x, weight, bias)
        return tracked_gemm
    else:
        return fn

utils_mod.dispatch_unquantized_gemm = hooked_dispatch

# Also hook npu_add_rms_norm to check if fused version is called
import torch_npu
_fused_rms_calls = []
original_fused = torch_npu.npu_add_rms_norm

def tracked_fused(*args, **kwargs):
    _fused_rms_calls.append(args[0].shape[0] if len(args) > 0 else -1)
    return original_fused(*args, **kwargs)

torch_npu.npu_add_rms_norm = tracked_fused

import torch_npu as tnpu
from vllm import LLM, SamplingParams

MODEL = "/home/bruceli/models/Qwen/Qwen3-4B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "gsm8k_test.jsonl")) as f:
    prompts = [json.loads(line)["question"] for i, line in enumerate(f) if i < 16]

kwargs = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.9,
              enable_prefix_caching=False, enable_chunked_prefill=False)

llm = LLM(**kwargs)

# Run single prompt
_gemm_m_values.clear()
_fused_rms_calls.clear()
sp = SamplingParams(temperature=0.0, max_tokens=8)
llm.generate(["Hello"], sp)

print("=== After single-prompt inference ===")
print("  npu_batch_invariant_gemm calls: %d" % len(_gemm_m_values))
if _gemm_m_values:
    print("  M values seen: %s" % sorted(set(_gemm_m_values)))
else:
    print("  WARNING: gemm hook not triggered! Patch may not be active in workers!")

print("  npu_add_rms_norm (fused) calls: %d" % len(_fused_rms_calls))
if _fused_rms_calls:
    print("  WARNING: fused npu_add_rms_norm is being called! Decomposition not active!")
else:
    print("  OK: fused npu_add_rms_norm NOT called (decomposition active)")

del llm
