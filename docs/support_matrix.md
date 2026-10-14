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
