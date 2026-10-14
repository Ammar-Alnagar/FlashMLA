#include "permute_q_b_proj.h"
#include <kerutils/kerutils.cuh>

namespace sm89::permute_q_b_proj {

__launch_bounds__(128)
__global__ void permute_q_b_proj_kernel_sm89(const Params params) {
    uint32_t row_idx = blockIdx.x;
    if (row_idx >= params.h_q * params.d_q) return;

    uint32_t head_idx = row_idx / params.d_q;
    uint32_t head_dim_idx = row_idx % params.d_q;
    uint32_t out_row_idx = head_dim_idx % 16 + head_idx * 16 + (head_dim_idx / 16u) * (params.h_q * 16);
    uint32_t num_scales_per_row = params.q_lora_rank / params.gran / 4;

    // 1. Copy weight bytes for this row from src row_idx to dst out_row_idx
    const fp8_e4m3* src_row = params.q_b_proj + row_idx * params.stride_q_b_dim0;
    fp8_e4m3* dst_row = params.q_b_proj_permuted + out_row_idx * params.stride_q_b_permuted_dim0;

    for (uint32_t col = threadIdx.x * 16; col < params.q_lora_rank; col += blockDim.x * 16) {
        if (col + 15 < params.q_lora_rank) {
            uint4 val = *reinterpret_cast<const uint4*>(src_row + col);
            *reinterpret_cast<uint4*>(dst_row + col) = val;
        } else {
            for (uint32_t c = col; c < params.q_lora_rank; ++c) {
                dst_row[c] = src_row[c];
            }
        }
    }

    // 2. Permute scale factors
    for (uint32_t scale_idx = threadIdx.x; scale_idx < num_scales_per_row; scale_idx += blockDim.x) {
        auto cur_scale = params.scale_factors[row_idx + scale_idx * params.stride_scale_factors_dim1];
        params.scale_factors_permuted[out_row_idx + scale_idx * params.stride_scale_factors_permuted_dim1] = cur_scale;
    }
}

void run_permute_q_b_proj_kernel(const Params& params) {
    permute_q_b_proj_kernel_sm89<<<params.h_q * params.d_q, 128, 0, params.stream>>>(params);
}

}  // namespace sm89::permute_q_b_proj
