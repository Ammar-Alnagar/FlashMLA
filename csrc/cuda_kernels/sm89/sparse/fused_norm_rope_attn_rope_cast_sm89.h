#pragma once

#include "params.h"
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"

namespace sm89::fused {

using PrefillParams = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<SparseAttnFwdMode::Prefill>;
using DecodeParams = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<SparseAttnFwdMode::Decode>;

void run_fused_prefill_sm89(const PrefillParams &params);
void run_fused_decode_sm89(const DecodeParams &params);

}  // namespace sm89::fused
