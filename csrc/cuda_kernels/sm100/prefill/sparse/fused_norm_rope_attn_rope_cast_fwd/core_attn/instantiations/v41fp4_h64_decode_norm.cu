#include "cuda_kernels/common/fused_norm_rope_attn_rope_cast_fwd/core_attn/params.h"
#include "../kernel.cuh"

namespace flash_mla::fused_norm_rope_attn_rope_cast::core_attn {

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, 64, true}>(const ParamT<SparseAttnFwdMode::Decode>& params);

}
