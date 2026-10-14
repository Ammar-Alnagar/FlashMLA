#include "sparse_decode_sm89.h"
#include "dequant_sm89.cuh"
#include "cuda_kernels/smxx/decode/combine/combine.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm89::sparse_decode {

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// 1 CTA processes 1 request/query token and 1 query head using 128 threads
__global__ void __launch_bounds__(128) sparse_decode_kernel_sm89(const SparseAttnDecodeParams params) {
    int req_idx = blockIdx.x;
    int s_q_idx = blockIdx.y;
    int h_q_idx = blockIdx.z;

    if (req_idx >= params.b || s_q_idx >= params.s_q || h_q_idx >= params.h_q) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int max_topk = params.topk;
    if (params.topk_length != nullptr) {
        int user_topk = params.topk_length[req_idx];
        if (user_topk < max_topk) {
            max_topk = user_topk;
        }
    }

    int extra_max_topk = params.extra_topk;
    if (params.extra_topk_length != nullptr) {
        int user_extra_topk = params.extra_topk_length[req_idx];
        if (user_extra_topk < extra_max_topk) {
            extra_max_topk = user_extra_topk;
        }
    }

    // Load Q vector [512] for (req_idx, s_q_idx, h_q_idx)
    const nv_bfloat16* q_base = reinterpret_cast<const nv_bfloat16*>(params.q);
    const nv_bfloat16* q_ptr = q_base + req_idx * params.stride_q_b + s_q_idx * params.stride_q_s_q + h_q_idx * params.stride_q_h_q;

    nv_bfloat16 q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_reg[i] = q_ptr[tid * 4 + i];
    }

    const int* idx_ptr = params.indices + req_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
    const uint8_t* kv_bytes = reinterpret_cast<const uint8_t*>(params.kv);

    float max_score = -CUDART_INF_F;

    // First pass: find max score over primary KV cache
    for (int t = 0; t < max_topk; ++t) {
        int token_idx = idx_ptr[t];
        if (token_idx != -1) {
            int page_idx = token_idx / params.page_block_size;
            int page_offset = token_idx % params.page_block_size;
            const uint8_t* token_ptr = kv_bytes + page_idx * params.stride_kv_block + page_offset * params.stride_kv_row;

            float dot_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int dim_idx = tid * 4 + i;
                float q_val = __bfloat162float(q_reg[i]);
                float k_val = dequant::dequant_v41_fp8(token_ptr, dim_idx);
                dot_sum += q_val * k_val;
            }
            dot_sum = warp_reduce_sum(dot_sum);

            __shared__ float warp_dots[4];
            if (lane_id == 0) warp_dots[warp_id] = dot_sum;
            __syncthreads();

            if (tid == 0) {
                float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                float score = total_dot * params.sm_scale;
                if (score > max_score) max_score = score;
            }
            __syncthreads();
        }
    }

    // Process extra_kv if provided
    if (params.extra_kv != nullptr && params.extra_indices != nullptr) {
        const int* extra_idx_ptr = params.extra_indices + req_idx * params.stride_extra_indices_b + s_q_idx * params.stride_extra_indices_s_q;
        const uint8_t* extra_kv_bytes = reinterpret_cast<const uint8_t*>(params.extra_kv);

        for (int t = 0; t < extra_max_topk; ++t) {
            int token_idx = extra_idx_ptr[t];
            if (token_idx != -1) {
                int page_idx = token_idx / params.extra_page_block_size;
                int page_offset = token_idx % params.extra_page_block_size;
                const uint8_t* token_ptr = extra_kv_bytes + page_idx * params.stride_extra_kv_block + page_offset * params.stride_extra_kv_row;

                float dot_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int dim_idx = tid * 4 + i;
                    float q_val = __bfloat162float(q_reg[i]);
                    float k_val = (params.extra_model_type == ModelType::V41_FP4) ?
                        dequant::dequant_v41_fp4(token_ptr, dim_idx) :
                        dequant::dequant_v41_fp8(token_ptr, dim_idx);
                    dot_sum += q_val * k_val;
                }
                dot_sum = warp_reduce_sum(dot_sum);

                __shared__ float warp_dots[4];
                if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                __syncthreads();

                if (tid == 0) {
                    float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                    float score = total_dot * params.sm_scale;
                    if (score > max_score) max_score = score;
                }
                __syncthreads();
            }
        }
    }

    __shared__ float smem_max_score;
    if (tid == 0) smem_max_score = max_score;
    __syncthreads();
    max_score = smem_max_score;

    float sum_exp = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (max_score != -CUDART_INF_F) {
        // Second pass over primary KV cache
        for (int t = 0; t < max_topk; ++t) {
            int token_idx = idx_ptr[t];
            if (token_idx != -1) {
                int page_idx = token_idx / params.page_block_size;
                int page_offset = token_idx % params.page_block_size;
                const uint8_t* token_ptr = kv_bytes + page_idx * params.stride_kv_block + page_offset * params.stride_kv_row;

                float dot_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int dim_idx = tid * 4 + i;
                    float q_val = __bfloat162float(q_reg[i]);
                    float k_val = dequant::dequant_v41_fp8(token_ptr, dim_idx);
                    dot_sum += q_val * k_val;
                }
                dot_sum = warp_reduce_sum(dot_sum);

                __shared__ float warp_dots[4];
                if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                __syncthreads();

                float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                float score = total_dot * params.sm_scale;
                float weight = expf(score - max_score);
                sum_exp += weight;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int dim_idx = tid * 4 + i;
                    float v_val = dequant::dequant_v41_fp8(token_ptr, dim_idx);
                    o_acc[i] += weight * v_val;
                }
            }
        }

        // Second pass over extra KV cache
        if (params.extra_kv != nullptr && params.extra_indices != nullptr) {
            const int* extra_idx_ptr = params.extra_indices + req_idx * params.stride_extra_indices_b + s_q_idx * params.stride_extra_indices_s_q;
            const uint8_t* extra_kv_bytes = reinterpret_cast<const uint8_t*>(params.extra_kv);

            for (int t = 0; t < extra_max_topk; ++t) {
                int token_idx = extra_idx_ptr[t];
                if (token_idx != -1) {
                    int page_idx = token_idx / params.extra_page_block_size;
                    int page_offset = token_idx % params.extra_page_block_size;
                    const uint8_t* token_ptr = extra_kv_bytes + page_idx * params.stride_extra_kv_block + page_offset * params.stride_extra_kv_row;

                    float dot_sum = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int dim_idx = tid * 4 + i;
                        float q_val = __bfloat162float(q_reg[i]);
                        float k_val = (params.extra_model_type == ModelType::V41_FP4) ?
                            dequant::dequant_v41_fp4(token_ptr, dim_idx) :
                            dequant::dequant_v41_fp8(token_ptr, dim_idx);
                        dot_sum += q_val * k_val;
                    }
                    dot_sum = warp_reduce_sum(dot_sum);

                    __shared__ float warp_dots[4];
                    if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                    __syncthreads();

                    float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                    float score = total_dot * params.sm_scale;
                    float weight = expf(score - max_score);
                    sum_exp += weight;

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int dim_idx = tid * 4 + i;
                        float v_val = (params.extra_model_type == ModelType::V41_FP4) ?
                            dequant::dequant_v41_fp4(token_ptr, dim_idx) :
                            dequant::dequant_v41_fp8(token_ptr, dim_idx);
                        o_acc[i] += weight * v_val;
                    }
                }
            }
        }
    }

    // Write LSE and output
    int lse_idx = req_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + h_q_idx;
    if (tid == 0) {
        if (max_score == -CUDART_INF_F || sum_exp == 0.0f) {
            params.lse[lse_idx] = CUDART_INF_F;
        } else {
            params.lse[lse_idx] = max_score + logf(sum_exp);
        }
    }

    float final_scale = 0.0f;
    if (sum_exp > 0.0f) {
        final_scale = 1.0f / sum_exp;
        if (params.attn_sink != nullptr) {
            float sink_val = params.attn_sink[h_q_idx];
            float lse_val = max_score + logf(sum_exp);
            float sink_factor = expf(lse_val) / (expf(lse_val) + expf(sink_val));
            final_scale *= sink_factor;
        }
    }

    if (params.out != nullptr) {
        nv_bfloat16* out_base = reinterpret_cast<nv_bfloat16*>(params.out);
        nv_bfloat16* out_ptr = out_base + req_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q + h_q_idx * params.stride_o_h_q;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_ptr[tid * 4 + i] = __float2bfloat16(o_acc[i] * final_scale);
        }
    }
}

void run_sparse_decode_sm89(const SparseAttnDecodeParams &params) {
    if (params.enable_split_kv) {
        SparseAttnDecodeParams unsplit_params = params;
        unsplit_params.enable_split_kv = false;
        dim3 grid(unsplit_params.b, unsplit_params.s_q, unsplit_params.h_q);
        dim3 block(128);
        sparse_decode_kernel_sm89<<<grid, block, 0, unsplit_params.stream>>>(unsplit_params);
    } else {
        dim3 grid(params.b, params.s_q, params.h_q);
        dim3 block(128);
        sparse_decode_kernel_sm89<<<grid, block, 0, params.stream>>>(params);
    }
}

}  // namespace sm89::sparse_decode
