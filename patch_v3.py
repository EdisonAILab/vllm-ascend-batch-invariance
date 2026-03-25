"""
Patch attention_v1.py for batch invariance on Ascend NPU.

Root cause: vLLM's ChunkedPrefill state uses npu_fused_infer_attention_score,
which produces different results than _npu_flash_attention (PrefillNoCache)
and _npu_paged_attention (DecodeOnly). When running a single prompt, all steps
use PrefillNoCache/DecodeOnly. When running a batch, mixed steps use
ChunkedPrefill — a different kernel that produces different numerics.

Fix: In the forward() method, detect ChunkedPrefill state and split it into
individual per-sequence calls using the SAME kernels as single-prompt mode:
- Prefill sequences -> _forward_prefill_no_cache (uses _npu_flash_attention)
- Decode sequences -> _forward_decode_only (uses _npu_paged_attention)

Also fix _forward_prefill_no_cache to process each sequence individually
(prevents cross-sequence interference in _npu_flash_attention).
"""
import os
import sys
import ast

filepath = "/vllm-ascend/vllm_ascend/attention/attention_v1.py"

with open(filepath, "r") as f:
    content = f.read()

# === Fix 1: _forward_prefill_no_cache — per-sequence flash attention ===
old_prefill = """    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask

        if is_310p():
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
            mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(query=query,
                                       key=key,
                                       value=value,
                                       mask=mask,
                                       seq_len=attn_metadata.seq_lens,
                                       scale_value=self.scale,
                                       num_heads=self.num_heads,
                                       num_kv_heads=self.num_kv_heads,
                                       out=output)
        assert output is not None
        return output[:num_tokens, :, :]"""

new_prefill = """    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask
        seq_lens = attn_metadata.seq_lens
        num_seqs = seq_lens.shape[0]

        if is_310p():
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            mask = mask.repeat(num_seqs, 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=seq_lens,
                                           scale_value=self.scale,
                                           num_heads=self.num_heads,
                                           num_kv_heads=self.num_kv_heads,
                                           out=output)
        elif num_seqs <= 1:
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=seq_lens,
                                           scale_value=self.scale,
                                           num_heads=self.num_heads,
                                           num_kv_heads=self.num_kv_heads,
                                           out=output)
        else:
            # Per-sequence flash attention for batch invariance.
            # _npu_flash_attention shows cross-sequence interference
            # when packing multiple sequences.
            offset = 0
            for i in range(num_seqs):
                slen = seq_lens[i].item()
                torch_npu._npu_flash_attention(
                    query=query[offset:offset + slen],
                    key=key[offset:offset + slen],
                    value=value[offset:offset + slen],
                    mask=mask,
                    seq_len=seq_lens[i:i + 1],
                    scale_value=self.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    out=output[offset:offset + slen])
                offset += slen

        assert output is not None
        return output[:num_tokens, :, :]"""

assert old_prefill in content, "Could not find _forward_prefill_no_cache"
content = content.replace(old_prefill, new_prefill)

# === Fix 2: forward() — split ChunkedPrefill into per-sequence calls ===
# Replace the "Normal V1 situation" else block that calls _forward_v1_style
# with logic that dispatches each sequence to PrefillNoCache or DecodeOnly

old_v1_else = """            # Normal V1 situation.
            else:
                # npu_fused_infer_attention_score does not support cases
                # where query.shape[0] != attn_metadata.query_start_loc[-1].
                # Thus we need unpad it here.
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)"""

new_v1_else = """            # Normal V1 situation (ChunkedPrefill / mixed prefill+decode).
            else:
                # For batch invariance: split the batch into individual
                # per-sequence calls using the SAME kernels as single-prompt
                # mode. This avoids npu_fused_infer_attention_score which
                # produces different numerics than _npu_flash_attention and
                # _npu_paged_attention.
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                output = self._forward_chunked_per_sequence(
                    query, key, value, attn_metadata, output, num_tokens)"""

assert old_v1_else in content, "Could not find V1 else block"
content = content.replace(old_v1_else, new_v1_else)

# === Fix 3: Add _forward_chunked_per_sequence method ===
# Insert it right before the forward() method

old_forward_def = """    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:"""

new_forward_def = """    def _forward_chunked_per_sequence(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens: int = 0,
    ) -> torch.Tensor:
        \"\"\"Split ChunkedPrefill batch into per-sequence calls for batch
        invariance. Each prefill sequence goes through _npu_flash_attention
        (same as PrefillNoCache), and each decode sequence goes through
        _npu_paged_attention (same as DecodeOnly).\"\"\"
        query_lens = attn_metadata.actual_seq_lengths_q
        # Convert cumulative to per-sequence
        q_lens = []
        prev = 0
        for ql in query_lens:
            q_lens.append(ql - prev)
            prev = ql

        seq_lens_list = attn_metadata.seq_lens_list
        num_seqs = len(q_lens)
        offset = 0

        for i in range(num_seqs):
            qlen = q_lens[i]
            kv_len = seq_lens_list[i]

            if qlen == 1 and kv_len > 1:
                # Decode: use paged attention (same kernel as DecodeOnly)
                q_i = query[offset:offset + 1]
                seq_lens_i = attn_metadata.seq_lens[i:i + 1]
                bt_i = attn_metadata.block_tables[i:i + 1]
                out_i = output[offset:offset + 1]
                torch_npu._npu_paged_attention(
                    query=q_i,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=bt_i,
                    context_lens=seq_lens_i,
                    out=out_i)
            elif qlen == kv_len:
                # Fresh prefill: use flash attention (same as PrefillNoCache)
                q_i = query[offset:offset + qlen]
                k_i = key[offset:offset + qlen]
                v_i = value[offset:offset + qlen]
                out_i = output[offset:offset + qlen]
                seq_len_i = attn_metadata.seq_lens[i:i + 1]

                # Build causal mask for this sequence length
                mask = self._get_prefill_mask(qlen, query.dtype, query.device)

                torch_npu._npu_flash_attention(
                    query=q_i,
                    key=k_i,
                    value=v_i,
                    mask=mask,
                    seq_len=seq_len_i,
                    scale_value=self.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    out=out_i)
            else:
                # Partial prefill (prefix cache hit): use fused_infer as
                # fallback since flash_attention can't handle partial prefill
                q_i = query[offset:offset + qlen]
                bt_i = attn_metadata.block_tables[i:i + 1]
                num_block, block_size, _, _ = self.key_cache.shape
                k_view = self.key_cache.view(num_block, block_size, -1)
                v_view = self.value_cache.view(num_block, block_size, -1)
                out_i, _ = torch_npu.npu_fused_infer_attention_score(
                    query=q_i,
                    key=k_view,
                    value=v_view,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=bt_i,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=[qlen],
                    actual_seq_lengths_kv=[kv_len],
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,
                )
                output[offset:offset + qlen] = out_i
            offset += qlen

        return output

    def _get_prefill_mask(self, seq_len, dtype, device):
        \"\"\"Get causal attention mask for prefill.\"\"\"
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        key = (seq_len, dtype, device)
        if key not in self._mask_cache:
            mask_flag = torch.ones((seq_len, seq_len),
                                   dtype=torch.bool,
                                   device=device).tril_()
            mask_flag = ~mask_flag
            mask_value = float('-inf') if dtype == torch.float16 else 1
            mask = torch.zeros(seq_len, seq_len, dtype=dtype,
                               device=device).masked_fill_(mask_flag,
                                                           mask_value)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:"""

assert old_forward_def in content, "Could not find forward() definition"
content = content.replace(old_forward_def, new_forward_def)

# Remove .pyc cache
pyc = "/vllm-ascend/vllm_ascend/attention/__pycache__/attention_v1.cpython-311.pyc"
if os.path.exists(pyc):
    os.remove(pyc)

# Verify syntax
ast.parse(content)

with open(filepath, "w") as f:
    f.write(content)

print("Patched attention_v1.py for batch invariance:")
print("  1. _forward_prefill_no_cache: per-sequence _npu_flash_attention")
print("  2. forward(): ChunkedPrefill -> per-sequence PrefillNoCache/DecodeOnly")
print("  3. _forward_chunked_per_sequence: splits mixed batches")
print("  4. _get_prefill_mask: causal mask cache for per-sequence prefill")
