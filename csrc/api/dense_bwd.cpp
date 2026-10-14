#include "common.h"

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
#include "cuda_kernels/sm100/prefill/dense/interface.h"
#include "cuda_kernels/sm89/dense/dense_sm89.h"

static void dense_prefill_bwd(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                             at::Tensor v, at::Tensor o, at::Tensor lse,
                             at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                             at::Tensor dq, at::Tensor dk, at::Tensor dv,
                             int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {
    Arch arch = Arch();
    arch.check_supported_arch();
    if (arch.is_sm89()) {
        sm89::dense::FMHACutlassSM89BwdRun(
            workspace_buffer, d_o, q, k, v, o, lse,
            cumulative_seqlen_q, cumulative_seqlen_kv, dq, dk, dv,
            mask_mode_code, softmax_scale, max_seqlen_q, max_seqlen_kv, is_varlen
        );
    } else {
        FMHACutlassSM100BwdRun(
            workspace_buffer, d_o, q, k, v, o, lse,
            cumulative_seqlen_q, cumulative_seqlen_kv, dq, dk, dv,
            mask_mode_code, softmax_scale, max_seqlen_q, max_seqlen_kv, is_varlen
        );
    }
}
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

void register_dense_bwd(pybind11::module_& m) {
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    m.def("dense_prefill_bwd",
        &dense_prefill_bwd,
        "Run Dense Attention Prefill Backward (cutlass FMHA)");
#else
    (void)m;    // The dense attention kernels are CUDA-only
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA
}
