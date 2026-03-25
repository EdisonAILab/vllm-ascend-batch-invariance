"""
Diagnostic: Identify exact source of vLLM batch non-invariance on NPU.
Tests attention state selection and NPU attention kernel invariance.
"""
import os
import sys
import json

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
DEVICE = "npu:0"


def load_gsm8k_prompts(n):
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")
    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            prompts.append(json.loads(line)["question"])
    return prompts


print("=" * 70)
print("Test 1: NPU flash attention kernel batch invariance")
print("=" * 70)

# Load model to get realistic shapes
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE).eval()
num_heads = model.config.num_attention_heads  # 32
num_kv_heads = model.config.num_key_value_heads  # 8
head_size = model.config.hidden_size // num_heads  # 80
scale = head_size ** -0.5
print(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_size={head_size}, scale={scale}")

# Test _npu_flash_attention with 1 sequence vs N packed sequences
SEQ_LEN = 64

torch.manual_seed(42)
# Create N sequences of queries/keys/values
N = 4
queries = [torch.randn(SEQ_LEN, num_heads, head_size, dtype=torch.bfloat16, device=DEVICE) for _ in range(N)]
keys = [torch.randn(SEQ_LEN, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEVICE) for _ in range(N)]
values = [torch.randn(SEQ_LEN, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEVICE) for _ in range(N)]

# Build causal mask for single sequence
mask_single = torch.zeros(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bfloat16, device=DEVICE)
mask_flag = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1).bool()
mask_single.masked_fill_(mask_flag, torch.finfo(torch.bfloat16).min)

# Build causal mask for N sequences (same mask broadcast)
mask_batch = mask_single.expand(N, 1, SEQ_LEN, SEQ_LEN).contiguous()

# --- Run singles ---
single_outputs = []
for i in range(N):
    q = queries[i].unsqueeze(0).reshape(SEQ_LEN, num_heads, head_size)
    k = keys[i].unsqueeze(0).reshape(SEQ_LEN, num_kv_heads, head_size)
    v = values[i].unsqueeze(0).reshape(SEQ_LEN, num_kv_heads, head_size)
    out = torch.empty(SEQ_LEN, num_heads, head_size, dtype=torch.bfloat16, device=DEVICE)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int64, device=DEVICE)

    torch_npu._npu_flash_attention(
        query=q, key=k, value=v,
        mask=mask_single,
        seq_len=seq_lens,
        scale_value=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        out=out
    )
    single_outputs.append(out.clone())

# --- Run batch (packed) ---
q_batch = torch.cat(queries, dim=0)  # [N*SEQ_LEN, num_heads, head_size]
k_batch = torch.cat(keys, dim=0)
v_batch = torch.cat(values, dim=0)
out_batch = torch.empty(N * SEQ_LEN, num_heads, head_size, dtype=torch.bfloat16, device=DEVICE)
seq_lens_batch = torch.tensor([SEQ_LEN] * N, dtype=torch.int64, device=DEVICE)

torch_npu._npu_flash_attention(
    query=q_batch, key=k_batch, value=v_batch,
    mask=mask_single,  # broadcast across sequences
    seq_len=seq_lens_batch,
    scale_value=scale,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    out=out_batch
)

# --- Compare ---
print(f"\n_npu_flash_attention: single (N=1) vs packed (N={N})")
for i in range(N):
    batch_slice = out_batch[i * SEQ_LEN:(i + 1) * SEQ_LEN]
    diff = (single_outputs[i] - batch_slice).abs().max().item()
    status = "OK" if diff == 0.0 else "MISMATCH"
    print(f"  Seq {i}: max_diff={diff:.8f}  {status}")

# --- Also test determinism (same call twice) ---
out_batch2 = torch.empty_like(out_batch)
torch_npu._npu_flash_attention(
    query=q_batch, key=k_batch, value=v_batch,
    mask=mask_single, seq_len=seq_lens_batch,
    scale_value=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
    out=out_batch2
)
det_diff = (out_batch - out_batch2).abs().max().item()
print(f"  Determinism (same call twice): max_diff={det_diff:.8f}  {'OK' if det_diff == 0.0 else 'NON-DET'}")

print("\n" + "=" * 70)
print("Test 2: npu_fused_infer_attention_score (chunked prefill kernel)")
print("=" * 70)

# Test the chunked prefill attention kernel
block_size = 64
num_blocks = N * (SEQ_LEN // block_size + 1)

# Create KV cache in block format
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEVICE)
value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEVICE)

# Populate KV cache from the same keys/values used above
for i in range(N):
    block_idx = i * (SEQ_LEN // block_size + 1)
    key_cache[block_idx, :SEQ_LEN] = keys[i].view(SEQ_LEN, num_kv_heads, head_size)
    value_cache[block_idx, :SEQ_LEN] = values[i].view(SEQ_LEN, num_kv_heads, head_size)

# Build attention mask for chunked prefill (int8 upper triangular)
max_seq = 2048
attn_mask_cp = torch.triu(torch.ones(max_seq, max_seq, device=DEVICE), diagonal=1).to(torch.int8)

# Single sequence through chunked prefill
single_cp_outputs = []
for i in range(N):
    q = queries[i]  # [SEQ_LEN, num_heads, head_size]
    block_idx = i * (SEQ_LEN // block_size + 1)
    bt = torch.tensor([[block_idx]], dtype=torch.int32, device=DEVICE)

    k_view = key_cache.view(num_blocks, block_size, -1)
    v_view = value_cache.view(num_blocks, block_size, -1)

    out_cp, _ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k_view, value=v_view,
        atten_mask=attn_mask_cp,
        block_table=bt,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=[SEQ_LEN],
        actual_seq_lengths_kv=[SEQ_LEN],
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=3,
    )
    single_cp_outputs.append(out_cp.clone())

# Batch through chunked prefill
q_all = torch.cat(queries, dim=0)
block_tables = torch.tensor(
    [[i * (SEQ_LEN // block_size + 1)] for i in range(N)],
    dtype=torch.int32, device=DEVICE
)
k_view = key_cache.view(num_blocks, block_size, -1)
v_view = value_cache.view(num_blocks, block_size, -1)

out_cp_batch, _ = torch_npu.npu_fused_infer_attention_score(
    query=q_all, key=k_view, value=v_view,
    atten_mask=attn_mask_cp,
    block_table=block_tables,
    input_layout="TND",
    block_size=block_size,
    actual_seq_lengths=[SEQ_LEN] * N,
    actual_seq_lengths_kv=[SEQ_LEN] * N,
    num_key_value_heads=num_kv_heads,
    num_heads=num_heads,
    scale=scale,
    sparse_mode=3,
)

print(f"\nnpu_fused_infer_attention_score: single (N=1) vs batch (N={N})")
for i in range(N):
    batch_slice = out_cp_batch[i * SEQ_LEN:(i + 1) * SEQ_LEN]
    diff = (single_cp_outputs[i] - batch_slice).abs().max().item()
    status = "OK" if diff == 0.0 else "MISMATCH"
    print(f"  Seq {i}: max_diff={diff:.8f}  {status}")

print("\n" + "=" * 70)
print("Test 3: Cross-kernel comparison (flash_attention vs fused_infer)")
print("=" * 70)

# Compare single-sequence output from flash_attention vs fused_infer
for i in range(N):
    diff = (single_outputs[i] - single_cp_outputs[i]).abs().max().item()
    status = "OK" if diff == 0.0 else "MISMATCH"
    print(f"  Seq {i}: flash_attn vs fused_infer: max_diff={diff:.8f}  {status}")

del model
torch_npu.npu.empty_cache()

print("\n" + "=" * 70)
print("Test 4: vLLM attention state logging")
print("=" * 70)

# Monkey-patch _build_attn_state to log which state is selected
import vllm_ascend.worker.model_runner_v1 as mr
original_build = mr.AscendModelRunner._build_attn_state

def logging_build(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
    state = original_build(self, num_reqs, num_scheduled_tokens, num_valid_tokens)
    print(f"  [ATTN_STATE] num_reqs={num_reqs}, state={state.name}", flush=True)
    return state

mr.AscendModelRunner._build_attn_state = logging_build

from vllm import LLM, SamplingParams

prompts = load_gsm8k_prompts(4)
sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)

print("\n--- Single prompt runs ---")
llm = LLM(model=MODEL_PATH, dtype="bfloat16", enforce_eager=True,
          tensor_parallel_size=1, max_model_len=4096,
          gpu_memory_utilization=0.9)
for i, p in enumerate(prompts[:2]):
    print(f"\n  Running single prompt {i}:")
    out = llm.generate([p], sp)
    print(f"  tokens={list(out[0].outputs[0].token_ids[:4])}")

print("\n--- Batched run (all prompts together) ---")
print(f"\n  Running batch of {len(prompts)} prompts:")
out = llm.generate(prompts, sp)
for i, o in enumerate(out):
    print(f"  Prompt {i} tokens={list(o.outputs[0].token_ids[:4])}")

del llm
torch_npu.npu.empty_cache()

# Restore original
mr.AscendModelRunner._build_attn_state = original_build
