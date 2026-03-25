"""
Per-operator batch invariance diagnostic for Qwen3-4B on Ascend NPU.
Tests each linear layer and other ops individually to find which are non-invariant.
"""
import os
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/bruceli/models/Qwen/Qwen3-4B"
DEVICE = "npu:0"
SEQ_LEN = 32

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is Berlin. The two cities are",
    "Once upon a time in a land far away, there lived a princess who dreamed of becoming a",
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the",
    "Artificial intelligence has transformed many industries, including healthcare, finance, and",
]


def test_single_linear(name, linear, input_batch, input_singles):
    """Test batch invariance of a single nn.Linear layer."""
    with torch.no_grad():
        out_batch = linear(input_batch)  # [4, seq, out]
        diffs = []
        for i in range(input_batch.shape[0]):
            out_single = linear(input_singles[i])  # [1, seq, out]
            diff = (out_single[0] - out_batch[i]).abs().max().item()
            diffs.append(diff)
    max_diff = max(diffs)
    return max_diff, diffs


def main():
    print(f"Loading {MODEL_PATH} on {DEVICE} ...")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'not set')}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE).eval()

    # Tokenize prompts
    ids_list = []
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if len(ids) < SEQ_LEN:
            pad = torch.full((SEQ_LEN - len(ids),), tokenizer.eos_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        else:
            ids = ids[:SEQ_LEN]
        ids_list.append(ids)

    batch_ids = torch.stack(ids_list).to(DEVICE)

    # Get embeddings (input to first layer)
    with torch.no_grad():
        batch_embeds = model.model.embed_tokens(batch_ids)  # [4, 32, 2048]
        single_embeds = [model.model.embed_tokens(batch_ids[i:i+1]) for i in range(4)]

    print(f"\nEmbed shape: {batch_embeds.shape}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num attention heads: {model.config.num_attention_heads}")
    print(f"Num KV heads: {model.config.num_key_value_heads}")
    print(f"Intermediate size: {model.config.intermediate_size}")

    # ========================================================
    # Test 1: Per-layer linear projections using actual hidden states
    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 1: Per-layer linear projection batch invariance")
    print("=" * 70)

    non_invariant_ops = []

    # Process through each layer
    hidden_batch = batch_embeds.clone()
    hidden_singles = [e.clone() for e in single_embeds]

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx >= 2:  # Test first 2 layers (pattern repeats)
            break

        print(f"\n--- Layer {layer_idx} ---")

        # Test input_layernorm (RMSNorm)
        with torch.no_grad():
            normed_batch = layer.input_layernorm(hidden_batch)
            normed_singles = [layer.input_layernorm(hidden_singles[i]) for i in range(4)]

        norm_diffs = []
        for i in range(4):
            diff = (normed_singles[i][0] - normed_batch[i]).abs().max().item()
            norm_diffs.append(diff)
        max_norm_diff = max(norm_diffs)
        status = "OK" if max_norm_diff == 0.0 else "MISMATCH"
        print(f"  input_layernorm: max_diff={max_norm_diff:.8f}  {status}")
        if max_norm_diff > 0:
            non_invariant_ops.append(f"layer{layer_idx}.input_layernorm")

        # Test each attention projection
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(attn, proj_name)
            max_diff, diffs = test_single_linear(
                f"layer{layer_idx}.self_attn.{proj_name}",
                proj,
                normed_batch,
                normed_singles
            )
            status = "OK" if max_diff == 0.0 else "MISMATCH"
            out_features = proj.weight.shape[0]
            print(f"  self_attn.{proj_name} ({model.config.hidden_size}->{out_features}): max_diff={max_diff:.8f}  {status}  diffs=[{', '.join(f'{d:.8f}' for d in diffs)}]")
            if max_diff > 0:
                non_invariant_ops.append(f"layer{layer_idx}.self_attn.{proj_name}")

        # Test MLP projections
        mlp = layer.mlp
        with torch.no_grad():
            # Post attention layernorm
            post_attn_normed_batch = layer.post_attention_layernorm(hidden_batch)
            post_attn_normed_singles = [layer.post_attention_layernorm(hidden_singles[i]) for i in range(4)]

        norm_diffs2 = []
        for i in range(4):
            diff = (post_attn_normed_singles[i][0] - post_attn_normed_batch[i]).abs().max().item()
            norm_diffs2.append(diff)
        max_norm_diff2 = max(norm_diffs2)
        status = "OK" if max_norm_diff2 == 0.0 else "MISMATCH"
        print(f"  post_attention_layernorm: max_diff={max_norm_diff2:.8f}  {status}")
        if max_norm_diff2 > 0:
            non_invariant_ops.append(f"layer{layer_idx}.post_attention_layernorm")

        for proj_name in ["gate_proj", "up_proj"]:
            proj = getattr(mlp, proj_name)
            max_diff, diffs = test_single_linear(
                f"layer{layer_idx}.mlp.{proj_name}",
                proj,
                post_attn_normed_batch,
                post_attn_normed_singles
            )
            status = "OK" if max_diff == 0.0 else "MISMATCH"
            out_features = proj.weight.shape[0]
            print(f"  mlp.{proj_name} ({proj.weight.shape[1]}->{out_features}): max_diff={max_diff:.8f}  {status}  diffs=[{', '.join(f'{d:.8f}' for d in diffs)}]")
            if max_diff > 0:
                non_invariant_ops.append(f"layer{layer_idx}.mlp.{proj_name}")

        # down_proj needs intermediate-dim input (gate_proj output * up_proj output)
        with torch.no_grad():
            gate_batch = mlp.gate_proj(post_attn_normed_batch)
            up_batch = mlp.up_proj(post_attn_normed_batch)
            intermediate_batch = torch.nn.functional.silu(gate_batch) * up_batch
            intermediate_singles = []
            for i in range(4):
                g = mlp.gate_proj(post_attn_normed_singles[i])
                u = mlp.up_proj(post_attn_normed_singles[i])
                intermediate_singles.append(torch.nn.functional.silu(g) * u)

        max_diff, diffs = test_single_linear(
            f"layer{layer_idx}.mlp.down_proj",
            mlp.down_proj,
            intermediate_batch,
            intermediate_singles
        )
        status = "OK" if max_diff == 0.0 else "MISMATCH"
        out_features = mlp.down_proj.weight.shape[0]
        print(f"  mlp.down_proj ({mlp.down_proj.weight.shape[1]}->{out_features}): max_diff={max_diff:.8f}  {status}  diffs=[{', '.join(f'{d:.8f}' for d in diffs)}]")
        if max_diff > 0:
            non_invariant_ops.append(f"layer{layer_idx}.mlp.down_proj")

    # ========================================================
    # Test 2: Isolated nn.Linear with controlled inputs
    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 2: Isolated nn.Linear with random inputs (varying output dims)")
    print("=" * 70)

    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    num_kv_heads = model.config.num_key_value_heads
    head_dim = hidden // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim  # 8 * 80 = 640 or 8 * 128 = 1024
    q_dim = model.config.num_attention_heads * head_dim
    test_dims = {
        f"q_proj ({hidden}->{q_dim})": (hidden, q_dim),
        f"k_proj ({hidden}->{kv_dim})": (hidden, kv_dim),
        f"v_proj ({hidden}->{kv_dim})": (hidden, kv_dim),
        f"o_proj ({hidden}->{hidden})": (hidden, hidden),
        f"gate_proj ({hidden}->{intermediate})": (hidden, intermediate),
        f"up_proj ({hidden}->{intermediate})": (hidden, intermediate),
        f"down_proj ({intermediate}->{hidden})": (intermediate, hidden),
        f"custom ({hidden}->512)": (hidden, 512),
        f"custom ({hidden}->256)": (hidden, 256),
        f"custom ({hidden}->128)": (hidden, 128),
        f"custom ({hidden}->1024)": (hidden, 1024),
        f"custom ({hidden}->640)": (hidden, 640),
    }

    torch.manual_seed(42)
    for name, (in_dim, out_dim) in test_dims.items():
        linear = torch.nn.Linear(in_dim, out_dim, dtype=torch.bfloat16, device=DEVICE)
        x_batch = torch.randn(4, SEQ_LEN, in_dim, dtype=torch.bfloat16, device=DEVICE)

        with torch.no_grad():
            out_batch = linear(x_batch)
            diffs = []
            for i in range(4):
                out_single = linear(x_batch[i:i + 1])
                diff = (out_single[0] - out_batch[i]).abs().max().item()
                diffs.append(diff)

        max_diff = max(diffs)
        status = "OK" if max_diff == 0.0 else "MISMATCH"
        print(f"  {name}: max_diff={max_diff:.8f}  {status}")

    # ========================================================
    # Test 3: Raw torch.mm batch invariance (M=1 vs M=4)
    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 3: Raw torch.mm batch invariance (M=1 vs M=4)")
    print("=" * 70)

    mm_shapes = [
        (hidden, hidden), (hidden, kv_dim), (hidden, q_dim),
        (hidden, intermediate), (intermediate, hidden),
        (hidden, 512), (hidden, 256), (hidden, 1024), (hidden, 640),
    ]
    for K, N in mm_shapes:
        torch.manual_seed(42)
        W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
        x_rows = [torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)]
        x_batch = torch.cat(x_rows, dim=0)  # [4, K]

        with torch.no_grad():
            out_batch = torch.mm(x_batch, W)  # [4, N]
            diffs = []
            for i in range(4):
                out_single = torch.mm(x_rows[i], W)  # [1, N]
                diff = (out_single[0] - out_batch[i]).abs().max().item()
                diffs.append(diff)

        max_diff = max(diffs)
        status = "OK" if max_diff == 0.0 else "MISMATCH"
        print(f"  mm({K}x{N}): max_diff={max_diff:.8f}  {status}")

    # ========================================================
    # Test 4: F.linear batch invariance (3D input)
    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 4: F.linear batch invariance (3D input, aten::linear dispatch)")
    print("=" * 70)

    flinear_shapes = [
        (hidden, hidden), (hidden, kv_dim), (hidden, q_dim),
        (hidden, intermediate), (intermediate, hidden),
        (hidden, 512), (hidden, 256), (hidden, 1024), (hidden, 640),
    ]
    for in_dim, out_dim in flinear_shapes:
        torch.manual_seed(42)
        W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(out_dim, dtype=torch.bfloat16, device=DEVICE)
        x_batch = torch.randn(4, SEQ_LEN, in_dim, dtype=torch.bfloat16, device=DEVICE)

        with torch.no_grad():
            out_batch = torch.nn.functional.linear(x_batch, W, b)
            diffs = []
            for i in range(4):
                out_single = torch.nn.functional.linear(x_batch[i:i + 1], W, b)
                diff = (out_single[0] - out_batch[i]).abs().max().item()
                diffs.append(diff)

        max_diff = max(diffs)
        status = "OK" if max_diff == 0.0 else "MISMATCH"
        print(f"  F.linear({in_dim}->{out_dim}): max_diff={max_diff:.8f}  {status}")

    # ========================================================
    # Test 5: log_softmax and mean batch invariance
    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 5: log_softmax and mean batch invariance")
    print("=" * 70)

    torch.manual_seed(42)
    vocab = 152064  # Qwen3-4B vocab size
    x_batch = torch.randn(4, vocab, dtype=torch.bfloat16, device=DEVICE)

    with torch.no_grad():
        ls_batch = torch.log_softmax(x_batch, dim=-1)
        diffs = []
        for i in range(4):
            ls_single = torch.log_softmax(x_batch[i:i + 1], dim=-1)
            diff = (ls_single[0] - ls_batch[i]).abs().max().item()
            diffs.append(diff)
    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  log_softmax (vocab={vocab}): max_diff={max_diff:.8f}  {status}")

    x_batch2 = torch.randn(4, 2048, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        m_batch = x_batch2.mean(dim=-1)
        diffs = []
        for i in range(4):
            m_single = x_batch2[i:i + 1].mean(dim=-1)
            diff = (m_single[0] - m_batch[i]).abs().max().item()
            diffs.append(diff)
    max_diff = max(diffs)
    status = "OK" if max_diff == 0.0 else "MISMATCH"
    print(f"  mean (dim=2048): max_diff={max_diff:.8f}  {status}")

    # ========================================================
    # Summary
    # ========================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Non-invariant operators")
    print("=" * 70)
    if non_invariant_ops:
        for op in non_invariant_ops:
            print(f"  MISMATCH: {op}")
    else:
        print("  All tested operators are batch-invariant!")


if __name__ == "__main__":
    main()
