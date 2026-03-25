"""Patch attention_v1.py: per-sequence attention + debug logging."""
import sys, os

filepath = "/vllm-ascend/vllm_ascend/attention/attention_v1.py"

with open(filepath, "r") as f:
    content = f.read()

# --- Fix 1: _forward_prefill_no_cache with debug logging ---
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
            # Per-sequence flash attention for batch invariance
            import sys as _sys
            print(f"[BATCH_INV] prefill_no_cache: num_seqs={num_seqs} seq_lens={seq_lens.tolist()}", file=_sys.stderr, flush=True)
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

# --- Fix 2: _forward_v1_style with debug logging ---
old_v1_block = """        output, _ = torch_npu.npu_fused_infer_attention_score(
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

        return output"""

new_v1_block = """        num_seqs = len(attn_metadata.actual_seq_lengths_q)
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
            # Per-sequence chunked prefill for batch invariance
            import sys as _sys
            print(f"[BATCH_INV] v1_style: num_seqs={num_seqs} q_lens={attn_metadata.actual_seq_lengths_q} kv_lens={attn_metadata.seq_lens_list}", file=_sys.stderr, flush=True)
            query_lens = attn_metadata.actual_seq_lengths_q
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

        return output"""

assert old_v1_block in content, "Could not find v1_style attention score block"
content = content.replace(old_v1_block, new_v1_block)

# Add debug logging to forward() dispatch - log which state is selected
old_dispatch = """            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)"""

new_dispatch = """            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                import sys as _sys
                _ns = attn_metadata.seq_lens.shape[0] if attn_metadata.seq_lens is not None else 0
                print(f"[ATTN_DISPATCH] PrefillNoCache num_seqs={_ns}", file=_sys.stderr, flush=True)
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)"""

assert old_dispatch in content, "Could not find PrefillNoCache dispatch"
content = content.replace(old_dispatch, new_dispatch)

# Add logging for ChunkedPrefill / v1_style dispatch
old_v1_dispatch = """            else:
                # npu_fused_infer_attention_score does not support cases
                # where query.shape[0] != attn_metadata.query_start_loc[-1].
                # Thus we need unpad it here.
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)"""

new_v1_dispatch = """            else:
                # npu_fused_infer_attention_score does not support cases
                # where query.shape[0] != attn_metadata.query_start_loc[-1].
                # Thus we need unpad it here.
                import sys as _sys
                _ns = len(attn_metadata.seq_lens_list) if attn_metadata.seq_lens_list else 0
                print(f"[ATTN_DISPATCH] ChunkedPrefill/V1Style num_seqs={_ns}", file=_sys.stderr, flush=True)
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)"""

assert old_v1_dispatch in content, "Could not find v1_style dispatch"
content = content.replace(old_v1_dispatch, new_v1_dispatch)

# Add decode logging
old_decode_dispatch = """            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                output = self._forward_decode_only(query, attn_metadata,
                                                   output)"""

new_decode_dispatch = """            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                import sys as _sys
                _ns = attn_metadata.seq_lens.shape[0] if attn_metadata.seq_lens is not None else 0
                print(f"[ATTN_DISPATCH] DecodeOnly num_seqs={_ns}", file=_sys.stderr, flush=True)
                output = self._forward_decode_only(query, attn_metadata,
                                                   output)"""

assert old_decode_dispatch in content, "Could not find DecodeOnly dispatch"
content = content.replace(old_decode_dispatch, new_decode_dispatch)

# Remove .pyc
pyc = "/vllm-ascend/vllm_ascend/attention/__pycache__/attention_v1.cpython-311.pyc"
if os.path.exists(pyc):
    os.remove(pyc)

with open(filepath, "w") as f:
    f.write(content)

# Verify syntax
import ast
ast.parse(content)
print("Patched attention_v1.py successfully (syntax OK)")
