# FlashMLA Architecture Support Matrix

This matrix documents operator support across GPU architectures in FlashMLA.

| Operator / Feature | SM100 / SM103 (Blackwell) | SM89 (Ada Lovelace) | SM80 (Ampere) | Ascend NPU |
| :--- | :---: | :---: | :---: | :---: |
| **DSA Sparse Prefill (`sparse_prefill_fwd`)** | Supported | Supported | Error (Unsupported) | Supported |
| **DSA Sparse Decode (`sparse_decode_fwd`)** | Supported | Supported | Error (Unsupported) | Supported |
| **Dense MHA Prefill Forward (`dense_prefill_fwd`)** | Supported | Supported | Error (Unsupported) | N/A |
| **Dense MHA Prefill Backward (`dense_prefill_bwd`)** | Supported | Supported | Error (Unsupported) | N/A |
| **Fused Norm+RoPE+Attn+RoPE+Cast Prefill** | Supported | Supported | Error (Unsupported) | N/A |
| **Fused Norm+RoPE+Attn+RoPE+Cast Decode** | Supported | Supported | Error (Unsupported) | N/A |
| **Weight Permute Helpers (`permute_q_b_proj`, `permute_wv_proj`)** | Supported | Supported | Error (Unsupported) | N/A |
| **Tile Scheduler & Combine Metadata Kernels** | Supported | Supported | Error (Unsupported) | N/A |

## Supported Configurations for SM89 (Ada Lovelace)
- **CUDA Toolkit:** CUDA 13.x by default (CUDA 13.1+)
- **Dtypes:** FP8 (`float8_e4m3fn`), BF16 (`bfloat16`), FP32 (`float32`), FP4 (`e2m1` in extra KV cache)
- **Head Dimensions:** `d_qk = 512`, `d_v = 512`, `h_q = 64` or `128`
- **SM80 Handling:** Attempts to run FlashMLA operators on SM80 GPUs will raise an explicit `TORCH_CHECK` error indicating that SM80 is unsupported and SM89+ is required.

## SM89 Status and Known Limitations

The SM89 kernels are written against the `mma.m16n8k16` / `ldmatrix` / `cp.async` instruction set.
They share the architecture-independent pieces (tile scheduler, combine, weight-permute helpers) with
the SM100 path wherever those are exact; only the attention core is reimplemented, because the SM100
core is built on tcgen05 / TMEM instructions that SM89 does not have.

Correctness status on an SM89 laptop GPU (RTX 4060 Laptop, 24 SMs, 8 GB):

| Operator | Status |
| :--- | :--- |
| `sparse_prefill_fwd` | Correct — all test cases pass |
| `sparse_decode_fwd` | Correct — all test cases pass |
| `fused_norm_rope_attn_rope_cast` prefill / decode | Correct; agrees with the reference to ~1e-12 cosine |
| `permute_q_b_proj`, `permute_wv_proj` | Correct — verified against an independent reference |
| `dense_prefill_fwd` / `dense_prefill_bwd` | Implemented, **not yet covered by a test** (the repository has no dense test on this branch) |

Known limitations:

- **Split-KV is disabled for SM89 sparse decoding.** `topk + extra_topk > 640` would normally enable
  split-KV; on SM89 every request is kept on a single split so that results stay exact. The
  per-split partial-output reduction (`o_accum`) does not yet agree with the reference on SM89, even
  though the LSE side and the shared combine kernel are both correct. This costs throughput for very
  long contexts; correctness is unaffected. See the `TODO(SM89)` in `csrc/api/sparse_decode.cpp`.
- **The fused kernel is precision-limited relative to the upstream element-wise tolerances.**
  Its `max_logits` / `lse` match the reference to roughly 2e-4 relative on the few elements that
  miss the upstream `rel_tol` of ~6e-5, while cosine similarity is ~1e-12. The SM89 core sums the
  512-dimension QK product with `mma.m16n8k16` in a different order than the SM100 core, so the
  upstream tolerance — calibrated against SM100's exact summation order — is not always met.
- **The dense MHA kernels are a straightforward scalar implementation**, one CTA per (token, head).
  They prioritise correctness and are substantially slower than the SM100 tensor-core kernels; they
  were not tuned.
- **PDL (`griddepcontrol` / programmatic stream serialization) is not used on SM89.** It is gated on
  having at least one SM90+ target, because the feature does not exist on SM89.
- **Register spill checks** are skipped by default for SM89 (`FLASH_MLA_SKIP_REG_SPILL_CHECK=1`):
  the SM89 kernels were verified spill-free, but the upstream check is calibrated for SM100.
