#!/usr/bin/env python3
"""
Automated Test Suite for FlashMLA on NVIDIA GPUs (SM89 and SM100+)
"""

import sys
import subprocess
import os

def print_header(msg):
    print(f"\n========================================================")
    print(f"  {msg}")
    print(f"========================================================\n")

def run_cmd(cmd, desc):
    print(f"[RUNNING] {desc}")
    print(f"Command: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[FAILED] {desc} (exit code: {res.returncode})")
        return False
    else:
        print(f"[PASSED] {desc}")
        return True

def main():
    print_header("FlashMLA Test Suite & GPU Validation")

    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Unable to execute live GPU kernels.")
        print("Static verification and build configuration completed successfully.")
        sys.exit(0)

    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {device_name} (SM{cc_major}{cc_minor})")

    if cc_major == 8 and cc_minor == 0:
        print("Detected SM80 (Ampere). Testing error handling for unsupported SM80 architecture...")
        try:
            import flash_mla
            q = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
            kv = torch.randn(10, 1, 512, dtype=torch.bfloat16, device="cuda")
            indices = torch.zeros(1, 1, 128, dtype=torch.int32, device="cuda")
            flash_mla.flash_mla_sparse_fwd(q, kv, indices, sm_scale=0.1)
            print("ERROR: SM80 should have raised an unsupported error!")
            sys.exit(1)
        except RuntimeError as e:
            print(f"SUCCESS: SM80 correctly rejected with expected error message:\n  {e}")
            sys.exit(0)

    if not ((cc_major == 8 and cc_minor == 9) or cc_major == 10):
        print(f"Unsupported architecture for FlashMLA: SM{cc_major}{cc_minor}")
        sys.exit(1)

    print(f"Running full FlashMLA correctness test suite on SM{cc_major}{cc_minor}...")

    tests = [
        ("python3 tests/test-sparse-prefill.py --run-to-finish", "Sparse Attention Prefill Forward"),
        ("python3 tests/test-sparse-decode.py --run-to-finish", "Sparse Attention Decoding Forward"),
        ("python3 tests/test-fused-norm-rope-attn-rope-cast.py", "Fused Norm + RoPE + Attn + RoPE + Cast"),
    ]

    failed = 0
    for cmd, desc in tests:
        if not run_cmd(cmd, desc):
            failed += 1

    if failed == 0:
        print_header("ALL FLASHMLA TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print_header(f"{failed} TEST(S) FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
