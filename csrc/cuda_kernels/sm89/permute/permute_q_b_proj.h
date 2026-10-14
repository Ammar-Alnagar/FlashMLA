#pragma once

#include "cuda_kernels/common/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/params.h"

namespace sm89::permute_q_b_proj {

using Params = flash_mla::fused_norm_rope_attn_rope_cast::permute_q_b_proj::Params;

void run_permute_q_b_proj_kernel(const Params& params);

}  // namespace sm89::permute_q_b_proj
