#pragma once

#include "cuda_kernels/common/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/params.h"

namespace sm89::permute_wv_proj {

using Params = flash_mla::fused_norm_rope_attn_rope_cast::permute_wv_proj::Params;

void run_permute_wv_proj_kernel(const Params& params);

}  // namespace sm89::permute_wv_proj
