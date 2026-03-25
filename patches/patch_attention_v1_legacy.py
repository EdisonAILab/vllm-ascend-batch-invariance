"""Patch attention_v1.py for batch invariance on NPU."""
import sys

filepath = "/vllm-ascend/vllm_ascend/attention/attention_v1.py"

with open(filepath, "r") as f:
    content = f.read()

# Backup
with open(filepath + ".bak", "w") as f:
    f.write(content)

# --- Fix 1: _forward_prefill_no_cache ---
old_prefill = '''    def _forward_prefill_no_cache(
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
        return output[:num_tokens, :, :]'''

new_prefill = '''    def _forward_prefill_no_cache(
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
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
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
            # Single sequence - no batch invariance concern
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
            # Process each sequence individually for batch invariance.
            # _npu_flash_attention exhibits cross-sequence interference
            # when multiple sequences are packed together, producing
            # different results than processing each sequence alone.
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
        return output[:num_tokens, :, :]'''

if old_prefill not in content:
    print("ERROR: Could not find _forward_prefill_no_cache to patch!")
    sys.exit(1)

content = content.replace(old_prefill, new_prefill)

# --- Fix 2: _forward_v1_style (ChunkedPrefill) ---
old_v1 = '''        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \\
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \\
                attn_metadata.seq_lens.to(device=query.device)

        # TODO:The npu_fused_infer_attention_score op is planned to
        # be utilized in a wider range in upcoming versions.
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        key = self.key_cache.view(  # type: ignore
            num_block, block_size, -1)
        value = self.value_cache.view(  # type: ignore
            num_block, block_size, -1)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=attn_metadata.block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )

        return output'''

new_v1 = '''        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \\
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \\
                attn_metadata.seq_lens.to(device=query.device)

        # TODO:The npu_fused_infer_attention_score op is planned to
        # be utilized in a wider range in upcoming versions.
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        key = self.key_cache.view(  # type: ignore
            num_block, block_size, -1)
        value = self.value_cache.view(  # type: ignore
            num_block, block_size, -1)

        num_seqs = len(attn_metadata.actual_seq_lengths_q)
        if is_310p() or num_seqs <= 1:
            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        else:
            # Process each sequence individually for batch invariance.
            # npu_fused_infer_attention_score produces different results
            # when multiple sequences are batched vs processed individually.
            query_lens = attn_metadata.actual_seq_lengths_q
            # Convert cumulative query_lens to per-sequence lengths
            q_lens = []
            prev = 0
            for ql in query_lens:
                q_lens.append(ql - prev)
                prev = ql
            offset = 0
            for i in range(num_seqs):
                qlen = q_lens[i]
                q_i = query[offset:offset + qlen]
                bt_i = attn_metadata.block_tables[i:i + 1]
                out_i, _ = torch_npu.npu_fused_infer_attention_score(
                    query=q_i,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=bt_i,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=[qlen],
                    actual_seq_lengths_kv=[attn_metadata.seq_lens_list[i]],
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,
                )
                output[offset:offset + qlen] = out_i
                offset += qlen

        return output'''

if old_v1 not in content:
    print("ERROR: Could not find _forward_v1_style paged attention section to patch!")
    sys.exit(1)

content = content.replace(old_v1, new_v1)

with open(filepath, "w") as f:
    f.write(content)

print("Successfully patched attention_v1.py")
print("  - _forward_prefill_no_cache: per-sequence _npu_flash_attention")
print("  - _forward_v1_style: per-sequence npu_fused_infer_attention_score")
