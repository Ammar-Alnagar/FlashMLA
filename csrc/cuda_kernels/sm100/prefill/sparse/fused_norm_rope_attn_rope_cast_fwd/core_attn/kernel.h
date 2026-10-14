#pragma once

#include "cuda_kernels/defines.h"
#include "cuda_kernels/common/fused_norm_rope_attn_rope_cast_fwd/core_attn/params.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

// `Config`, `ParamsTemplate` and `ParamT` are architecture independent and are shared verbatim with
// the sm89 kernels, so they live in csrc/cuda_kernels/common/... and are pulled in here. This header
// keeps the sm100 namespace intact: `kernel.cuh` relies on unqualified lookup reaching helpers in
// the enclosing `sm100` namespace (e.g. `get_max`), which would break from any other namespace.
using flash_mla::fused_norm_rope_attn_rope_cast::core_attn::Config;
using flash_mla::fused_norm_rope_attn_rope_cast::core_attn::ParamsTemplate;
using flash_mla::fused_norm_rope_attn_rope_cast::core_attn::ParamT;

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}  // namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn