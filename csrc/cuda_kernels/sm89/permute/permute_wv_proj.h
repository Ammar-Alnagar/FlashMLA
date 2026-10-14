#pragma once

#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.h"

namespace sm89::permute_wv_proj {

using Params = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj::Params;

void run_permute_wv_proj_kernel(const Params& params);

}  // namespace sm89::permute_wv_proj
