"""
Patch vllm-ascend to support HuggingFace FP8 checkpoints on Ascend NPU.

Problem: vllm-ascend's AscendQuantConfig.override_quantization_method()
unconditionally remaps ALL quant methods to "ascend", but the Ascend backend
expects per-layer quant descriptors that HF FP8 checkpoints don't have.

Fix: Skip the Ascend override for FP8 models. The standard compressed-tensors
handler will dequantize FP8 weights to bfloat16 during loading. The model runs
as unquantized bf16 at inference (no FP8 speedup, but correct execution).
"""
import os
import sys
import glob

QUANT_CONFIG_PATH = "/vllm-ascend/vllm_ascend/quantization/quant_config.py"
BACKUP_PATH = QUANT_CONFIG_PATH + ".bak_fp8"

# Read from backup if exists
if os.path.exists(BACKUP_PATH):
    with open(BACKUP_PATH, "r") as f:
        content = f.read()
    print("Restored from backup %s" % BACKUP_PATH)
else:
    with open(QUANT_CONFIG_PATH, "r") as f:
        content = f.read()
    with open(BACKUP_PATH, "w") as f:
        f.write(content)
    print("Backup saved to %s" % BACKUP_PATH)

# Patch 1: override_quantization_method - skip for FP8
old_override = """    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return ASCEND_QUANTIZATION_METHOD
        return None"""

new_override = """    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            # For FP8 models, use compressed-tensors which dequantizes
            # FP8 weights to bfloat16 at load time. The native fp8
            # handler uses CUDA-only Marlin kernels.
            quant_method = hf_quant_cfg.get("quant_method", "")
            if quant_method == "fp8":
                return "compressed-tensors"
            return ASCEND_QUANTIZATION_METHOD
        return None"""

if old_override not in content:
    print("ERROR: Could not find override_quantization_method to patch")
    sys.exit(1)

content = content.replace(old_override, new_override)

with open(QUANT_CONFIG_PATH, "w") as f:
    f.write(content)

# Clear .pyc cache
for pyc in glob.glob("/vllm-ascend/vllm_ascend/quantization/__pycache__/quant_config*.pyc"):
    os.remove(pyc)
    print("Removed cached: %s" % pyc)

print("Patched quant_config.py: FP8 models now bypass Ascend quant override")

# Patch 2: Add "fp8" and "compressed-tensors" to supported_quantization in platform.py
PLATFORM_PATH = "/vllm-ascend/vllm_ascend/platform.py"
PLATFORM_BACKUP = PLATFORM_PATH + ".bak_fp8"

if os.path.exists(PLATFORM_BACKUP):
    with open(PLATFORM_BACKUP, "r") as f:
        platform_content = f.read()
    print("Restored platform.py from backup")
else:
    with open(PLATFORM_PATH, "r") as f:
        platform_content = f.read()
    with open(PLATFORM_BACKUP, "w") as f:
        f.write(platform_content)
    print("Backup saved to %s" % PLATFORM_BACKUP)

old_supported = "    supported_quantization: list[str] = [ASCEND_QUANTIZATION_METHOD]"
new_supported = '    supported_quantization: list[str] = [ASCEND_QUANTIZATION_METHOD, "fp8", "compressed-tensors"]'

if old_supported not in platform_content:
    print("WARNING: Could not find supported_quantization to patch (may already be patched)")
else:
    platform_content = platform_content.replace(old_supported, new_supported)
    with open(PLATFORM_PATH, "w") as f:
        f.write(platform_content)
    for pyc in glob.glob("/vllm-ascend/vllm_ascend/__pycache__/platform*.pyc"):
        os.remove(pyc)
        print("Removed cached: %s" % pyc)
    print("Patched platform.py: added fp8 and compressed-tensors to supported_quantization")
