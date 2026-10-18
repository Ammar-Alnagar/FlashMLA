#include "common.h"

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
#ifdef FLASH_MLA_IS_BUILD_ON_SM100
#include "cuda_kernels/sm100/prefill/dense/interface.h"
#endif
#include "cuda_kernels/sm89/dense/dense_sm89.h"

static void dense_prefill_fwd(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                             at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                             at::Tensor o, at::Tensor lse,
                             int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {
    Arch arch = Arch();
    arch.check_supported_arch();
    if (arch.is_sm89()) {
        sm89::dense::FMHACutlassSM89FwdRun(
            workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv,
            o, lse, mask_mode_code, softmax_scale, max_seqlen_q, max_seqlen_kv, is_varlen
        );
    }
#ifdef FLASH_MLA_IS_BUILD_ON_SM100
    else {
        FMHACutlassSM100FwdRun(
            workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv,
            o, lse, mask_mode_code, softmax_scale, max_seqlen_q, max_seqlen_kv, is_varlen
        );
    }
#endif
}
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("dense_prefill_fwd", TORCH_BOX(&dense_prefill_fwd));
}
}
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA
