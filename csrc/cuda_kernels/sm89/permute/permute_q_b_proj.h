#pragma once

#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.h"

namespace sm89::permute_q_b_proj {

using Params = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj::Params;

void run_permute_q_b_proj_kernel(const Params& params);

}  // namespace sm89::permute_q_b_proj
