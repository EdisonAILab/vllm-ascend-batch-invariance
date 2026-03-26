"""Fix AscendRMSNorm to handle non-Ascend quant configs (e.g. Fp8Config)."""
import os

path = "/vllm-ascend/vllm_ascend/ops/layernorm.py"
with open(path) as f:
    content = f.read()

old = """        if vllm_config.quant_config is not None and \\
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):"""

new = """        if vllm_config.quant_config is not None and \\
                hasattr(vllm_config.quant_config, "quant_description") and \\
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):"""

if old not in content:
    print("WARNING: Could not find layernorm code to patch (may already be fixed)")
else:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    # Clear pyc
    import glob
    for pyc in glob.glob("/vllm-ascend/vllm_ascend/ops/__pycache__/layernorm*.pyc"):
        os.remove(pyc)
    print("Fixed layernorm.py: added hasattr check for quant_description")
