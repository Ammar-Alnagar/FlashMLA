#include "permute_wv_proj.h"
#include <kerutils/kerutils.cuh>

namespace sm89::permute_wv_proj {

static constexpr uint32_t CHUNK_SIZE = 32;
static constexpr uint32_t D_O = 512;

__launch_bounds__(128)
__global__ void permute_wv_proj_kernel_sm89(const Params params) {
    uint32_t row_idx = blockIdx.x;
    uint32_t wv_group_idx = blockIdx.y;
    uint32_t row_size = params.wv_group_size * D_O;

    const fp8_e4m3* src_row = params.wv_proj + wv_group_idx * params.stride_wv_proj_dim0 + row_idx * params.stride_wv_proj_dim1;
    fp8_e4m3* dst_row = params.wv_proj_permuted + wv_group_idx * params.stride_wv_proj_permuted_dim0 + row_idx * params.stride_wv_proj_permuted_dim1;

    // 1. Permute scale factors
    uint32_t num_sf_per_head = D_O / 32;
    uint32_t num_sf = params.wv_group_size * num_sf_per_head;
    for (uint32_t i = threadIdx.x; i < num_sf; i += blockDim.x) {
        uint32_t head_idx = i / num_sf_per_head;
        uint32_t head_dim_chunk_idx = i % num_sf_per_head;
        uint32_t input_sf_idx_in_row = i;
        uint8_t cur_sf = *((uint8_t*)(params.scale_factors + wv_group_idx * params.stride_scale_factors_dim0 + row_idx + (input_sf_idx_in_row / 4) * params.stride_scale_factors_dim2) + input_sf_idx_in_row % 4);
        uint32_t output_sf_idx_in_row = head_idx + head_dim_chunk_idx * params.wv_group_size;
        *((uint8_t*)(params.scale_factors_permuted + wv_group_idx * params.stride_scale_factors_permuted_dim0 + row_idx + (output_sf_idx_in_row / 4) * params.stride_scale_factors_permuted_dim2) + output_sf_idx_in_row % 4) = cur_sf;
    }

    // 2. Permute row chunks
    uint32_t num_chunks = params.wv_group_size * (D_O / CHUNK_SIZE);
    for (uint32_t i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        uint32_t head_idx = i / (D_O / CHUNK_SIZE);
        uint32_t head_dim_chunk_idx = i % (D_O / CHUNK_SIZE);
        uint32_t chunk_idx_in_permuted_row = head_dim_chunk_idx * params.wv_group_size + head_idx;

        const fp8_e4m3* src_chunk = src_row + i * CHUNK_SIZE;
        fp8_e4m3* dst_chunk = dst_row + chunk_idx_in_permuted_row * CHUNK_SIZE;

        *reinterpret_cast<uint4*>(dst_chunk + 0) = *reinterpret_cast<const uint4*>(src_chunk + 0);
        *reinterpret_cast<uint4*>(dst_chunk + 16) = *reinterpret_cast<const uint4*>(src_chunk + 16);
    }
}

void run_permute_wv_proj_kernel(const Params& params) {
    permute_wv_proj_kernel_sm89<<<dim3(params.d_proj_out, params.n_wv_group), 128, 0, params.stream>>>(params);
}

}  // namespace sm89::permute_wv_proj
