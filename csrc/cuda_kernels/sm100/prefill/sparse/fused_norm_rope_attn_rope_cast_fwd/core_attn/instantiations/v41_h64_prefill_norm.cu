#include "../kernel.h"
#include "../kernel.cuh"

namespace flash_mla::fused_norm_rope_attn_rope_cast::core_attn {

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Prefill, ModelType::V41, ModelType::V41, 64, true}>(const ParamT<SparseAttnFwdMode::Prefill>& params);

}
