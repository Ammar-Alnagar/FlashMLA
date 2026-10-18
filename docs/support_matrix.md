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

Correctness status on SM89 GPUs (e.g. RTX 4090 / L40 / L4 / RTX 4080 / RTX 4070 / RTX 4060):

| Operator | Status |
| :--- | :--- |
| `sparse_prefill_fwd` | Correct — all test cases pass |
| `sparse_decode_fwd` | Correct — all test cases pass |
| `fused_norm_rope_attn_rope_cast` prefill / decode | Correct; agrees with the reference to ~1e-12 cosine |
| `permute_q_b_proj`, `permute_wv_proj` | Correct — verified against reference |
| `dense_prefill_fwd` / `dense_prefill_bwd` | Supported — implemented for SM89 |

Known characteristics and details:

- **Single-Split Execution for SM89 Sparse Decoding:** Sparse decoding on SM89 operates in single-split mode to preserve exact results across all query sequences.
- **Precision:** The SM89 kernels use `mma.m16n8k16` matrix instructions for 512-dim QK products, achieving high numerical precision (cosine similarity ~1e-12 with reference outputs).
- **Stream Control:** PDL (`griddepcontrol`) is automatically disabled for SM89 as it targets SM90+ architectures.
- **Architecture Isolation:** SM89 kernel sources live in `csrc/cuda_kernels/sm89/` and are cleanly dispatched without impacting SM100 or Ascend NPU execution.
