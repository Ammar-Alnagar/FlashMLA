#include "fused_norm_rope_attn_rope_cast_sm89.h"
#include "dequant_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm89::fused {

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__inline__ __device__ uint8_t float_to_e8m0_byte(float scale) {
    float log_val = log2f(fmaxf(scale, 1e-20f));
    int exp_val = __float2int_rn(log_val) + 127;
    return static_cast<uint8_t>(fminf(fmaxf(static_cast<float>(exp_val), 0.0f), 255.0f));
}

__global__ void fused_prefill_kernel_sm89(const PrefillParams params) {
    int s_q_idx = blockIdx.x;
    if (s_q_idx >= params.s_q) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int token_pos = params.token_positions[s_q_idx];

    const nv_bfloat16* q_base = reinterpret_cast<const nv_bfloat16*>(params.q);

    __shared__ float smem_q_head[512];

    int max_topk = params.topk;
    if (params.topk_length != nullptr) {
        int user_topk = params.topk_length[s_q_idx];
        if (user_topk < max_topk) max_topk = user_topk;
    }

    const int* idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;
    const nv_bfloat16* kv_base = reinterpret_cast<const nv_bfloat16*>(params.kv);

    for (int h_q_idx = 0; h_q_idx < params.h_q; ++h_q_idx) {
        const nv_bfloat16* q_head_ptr = q_base + s_q_idx * params.stride_q_s_q + h_q_idx * params.stride_q_h_q;

        // Step 1: Per-head RMSNorm & RoPE into smem_q_head
        float head_sum_sq = 0.0f;
        if (params.enable_q_norm) {
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                float val = __bfloat162float(q_head_ptr[d]);
                head_sum_sq += val * val;
            }
            head_sum_sq = warp_reduce_sum(head_sum_sq);
            __shared__ float warp_sums[4];
            if (lane_id == 0) warp_sums[warp_id] = head_sum_sq;
            __syncthreads();

            float total_head_sq = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
            float inv_rms = rsqrtf(total_head_sq / static_cast<float>(params.d_qk) + params.rms_norm_eps);
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                smem_q_head[d] = __bfloat162float(q_head_ptr[d]) * inv_rms;
            }
            __syncthreads();
        } else {
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                smem_q_head[d] = __bfloat162float(q_head_ptr[d]);
            }
            __syncthreads();
        }

        const float* cos_sin_ptr = params.cos_sin_cache + token_pos * params.rope_dim;
        int num_pairs = params.rope_dim / 2;
        for (int p = tid; p < num_pairs; p += blockDim.x) {
            int dim0 = p * 2;
            int dim1 = p * 2 + 1;

            float cos_val = cos_sin_ptr[dim0];
            float sin_val = cos_sin_ptr[dim1];

            float q0 = smem_q_head[dim0];
            float q1 = smem_q_head[dim1];

            smem_q_head[dim0] = q0 * cos_val - q1 * sin_val;
            smem_q_head[dim1] = q1 * cos_val + q0 * sin_val;
        }
        __syncthreads();

        // Step 2: Core Attention
        float max_score = -CUDART_INF_F;
        for (int t = 0; t < max_topk; ++t) {
            int kv_idx = idx_ptr[t];
            if (kv_idx >= 0 && kv_idx < params.s_kv) {
                const nv_bfloat16* k_ptr = kv_base + kv_idx * params.stride_kv_s_kv;
                float dot_sum = 0.0f;
                for (int d = tid; d < params.d_qk; d += blockDim.x) {
                    dot_sum += smem_q_head[d] * __bfloat162float(k_ptr[d]);
                }
                dot_sum = warp_reduce_sum(dot_sum);

                __shared__ float warp_dots[4];
                if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                __syncthreads();

                float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                float score = total_dot * params.sm_scale;
                if (score > max_score) max_score = score;
            }
        }

        __shared__ float smem_max;
        if (tid == 0) smem_max = max_score;
        __syncthreads();
        max_score = smem_max;

        float sum_exp = 0.0f;
        float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (max_score != -CUDART_INF_F) {
            for (int t = 0; t < max_topk; ++t) {
                int kv_idx = idx_ptr[t];
                if (kv_idx >= 0 && kv_idx < params.s_kv) {
                    const nv_bfloat16* k_ptr = kv_base + kv_idx * params.stride_kv_s_kv;
                    float dot_sum = 0.0f;
                    for (int d = tid; d < params.d_qk; d += blockDim.x) {
                        dot_sum += smem_q_head[d] * __bfloat162float(k_ptr[d]);
                    }
                    dot_sum = warp_reduce_sum(dot_sum);

                    __shared__ float warp_dots[4];
                    if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                    __syncthreads();

                    float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                    float score = total_dot * params.sm_scale;
                    float weight = expf(score - max_score);
                    sum_exp += weight;

                    for (int idx = 0; idx < 4; ++idx) {
                        int d = tid + idx * blockDim.x;
                        if (d < params.d_v) {
                            o_acc[idx] += weight * __bfloat162float(k_ptr[d]);
                        }
                    }
                    __syncthreads();
                }
            }
        }

        int out_idx = s_q_idx * params.h_q + h_q_idx;
        if (tid == 0) {
            if (max_score == -CUDART_INF_F || sum_exp == 0.0f) {
                params.max_logits[out_idx] = -CUDART_INF_F;
                params.lse[out_idx] = CUDART_INF_F;
            } else {
                params.max_logits[out_idx] = max_score;
                params.lse[out_idx] = max_score + logf(sum_exp);
            }
        }

        // Quantize and write O into FP8 out_fp8 and out_sf as byte scales if present
        if (params.out_fp8 != nullptr && params.out_sf != nullptr) {
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            int wv_group_idx = h_q_idx / params.wv_group_size;
            int head_in_group = h_q_idx % params.wv_group_size;

            __shared__ float smem_o_head[512];
            for (int idx = 0; idx < 4; ++idx) {
                int d = tid + idx * blockDim.x;
                if (d < params.d_v) {
                    smem_o_head[d] = o_acc[idx] * inv_sum;
                }
            }
            __syncthreads();

            int total_group_dims = params.wv_group_size * params.d_v;
            int num_scale_tiles = total_group_dims / 32;

            __nv_fp8_e4m3* fp8_dst = params.out_fp8 + s_q_idx * params.n_wv_group * total_group_dims + wv_group_idx * total_group_dims;
            uint8_t* sf_byte_dst = reinterpret_cast<uint8_t*>(params.out_sf) + s_q_idx * params.stride_out_sf_s_q * 4 + wv_group_idx * params.stride_out_sf_wv_group * 4;

            for (int tile_idx = tid; tile_idx < num_scale_tiles; tile_idx += blockDim.x) {
                int tile_head = tile_idx / (params.d_v / 32);
                if (tile_head == head_in_group) {
                    int tile_offset_in_head = (tile_idx % (params.d_v / 32)) * 32;
                    float max_val = 1e-8f;
                    for (int c = 0; c < 32; ++c) {
                        max_val = fmaxf(max_val, fabsf(smem_o_head[tile_offset_in_head + c]));
                    }
                    float scale = max_val / 448.0f;
                    float inv_scale = 1.0f / scale;

                    int global_tile_offset = tile_head * params.d_v + tile_offset_in_head;
                    for (int c = 0; c < 32; ++c) {
                        float v = smem_o_head[tile_offset_in_head + c] * inv_scale;
                        fp8_dst[global_tile_offset + c] = __nv_fp8_e4m3(__float2half(v));
                    }

                    sf_byte_dst[tile_idx] = float_to_e8m0_byte(scale);
                }
            }
            __syncthreads();
        }
    }
}

__global__ void fused_decode_kernel_sm89(const DecodeParams params) {
    int s_q_idx = blockIdx.x;
    if (s_q_idx >= params.s_q) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int token_pos = params.token_positions[s_q_idx];

    const nv_bfloat16* q_base = reinterpret_cast<const nv_bfloat16*>(params.q);

    __shared__ float smem_q_head[512];

    int max_topk = params.topk;
    if (params.topk_length != nullptr) {
        int user_topk = params.topk_length[s_q_idx];
        if (user_topk < max_topk) max_topk = user_topk;
    }

    const int* idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;
    const uint8_t* kv_bytes = reinterpret_cast<const uint8_t*>(params.kv);

    for (int h_q_idx = 0; h_q_idx < params.h_q; ++h_q_idx) {
        const nv_bfloat16* q_head_ptr = q_base + s_q_idx * params.stride_q_s_q + h_q_idx * params.stride_q_h_q;

        // Step 1: Per-head RMSNorm & RoPE into smem_q_head
        float head_sum_sq = 0.0f;
        if (params.enable_q_norm) {
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                float val = __bfloat162float(q_head_ptr[d]);
                head_sum_sq += val * val;
            }
            head_sum_sq = warp_reduce_sum(head_sum_sq);
            __shared__ float warp_sums[4];
            if (lane_id == 0) warp_sums[warp_id] = head_sum_sq;
            __syncthreads();

            float total_head_sq = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
            float inv_rms = rsqrtf(total_head_sq / static_cast<float>(params.d_qk) + params.rms_norm_eps);
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                smem_q_head[d] = __bfloat162float(q_head_ptr[d]) * inv_rms;
            }
            __syncthreads();
        } else {
            for (int d = tid; d < params.d_qk; d += blockDim.x) {
                smem_q_head[d] = __bfloat162float(q_head_ptr[d]);
            }
            __syncthreads();
        }

        const float* cos_sin_ptr = params.cos_sin_cache + token_pos * params.rope_dim;
        int num_pairs = params.rope_dim / 2;
        for (int p = tid; p < num_pairs; p += blockDim.x) {
            int dim0 = p * 2;
            int dim1 = p * 2 + 1;

            float cos_val = cos_sin_ptr[dim0];
            float sin_val = cos_sin_ptr[dim1];

            float q0 = smem_q_head[dim0];
            float q1 = smem_q_head[dim1];

            smem_q_head[dim0] = q0 * cos_val - q1 * sin_val;
            smem_q_head[dim1] = q1 * cos_val + q0 * sin_val;
        }
        __syncthreads();

        // Step 2: Core Attention (Decode)
        float max_score = -CUDART_INF_F;

        for (int t = 0; t < max_topk; ++t) {
            int token_idx = idx_ptr[t];
            if (token_idx != -1) {
                int page_idx = token_idx / params.page_block_size;
                int page_offset = token_idx % params.page_block_size;
                const uint8_t* token_ptr = kv_bytes + page_idx * params.stride_kv_block + page_offset * params.stride_kv_row;

                float dot_sum = 0.0f;
                for (int d = tid; d < params.d_qk; d += blockDim.x) {
                    float q_val = smem_q_head[d];
                    float k_val = dequant::dequant_v41_fp8(token_ptr, d);
                    dot_sum += q_val * k_val;
                }
                dot_sum = warp_reduce_sum(dot_sum);

                __shared__ float warp_dots[4];
                if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                __syncthreads();

                float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                float score = total_dot * params.sm_scale;
                if (score > max_score) max_score = score;
            }
        }

        __shared__ float smem_max;
        if (tid == 0) smem_max = max_score;
        __syncthreads();
        max_score = smem_max;

        float sum_exp = 0.0f;
        float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (max_score != -CUDART_INF_F) {
            for (int t = 0; t < max_topk; ++t) {
                int token_idx = idx_ptr[t];
                if (token_idx != -1) {
                    int page_idx = token_idx / params.page_block_size;
                    int page_offset = token_idx % params.page_block_size;
                    const uint8_t* token_ptr = kv_bytes + page_idx * params.stride_kv_block + page_offset * params.stride_kv_row;

                    float dot_sum = 0.0f;
                    for (int d = tid; d < params.d_qk; d += blockDim.x) {
                        float q_val = smem_q_head[d];
                        float k_val = dequant::dequant_v41_fp8(token_ptr, d);
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

                    for (int idx = 0; idx < 4; ++idx) {
                        int d = tid + idx * blockDim.x;
                        if (d < params.d_v) {
                            float v_val = dequant::dequant_v41_fp8(token_ptr, d);
                            o_acc[idx] += weight * v_val;
                        }
                    }
                    __syncthreads();
                }
            }
        }

        int lse_idx = s_q_idx * params.h_q + h_q_idx;
        if (tid == 0) {
            if (max_score == -CUDART_INF_F || sum_exp == 0.0f) {
                params.lse[lse_idx] = CUDART_INF_F;
            } else {
                params.lse[lse_idx] = max_score + logf(sum_exp);
            }
        }

        if (params.out_fp8 != nullptr && params.out_sf != nullptr) {
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            int wv_group_idx = h_q_idx / params.wv_group_size;
            int head_in_group = h_q_idx % params.wv_group_size;

            __shared__ float smem_o_head[512];
            for (int idx = 0; idx < 4; ++idx) {
                int d = tid + idx * blockDim.x;
                if (d < params.d_v) {
                    smem_o_head[d] = o_acc[idx] * inv_sum;
                }
            }
            __syncthreads();

            int total_group_dims = params.wv_group_size * params.d_v;
            int num_scale_tiles = total_group_dims / 32;

            __nv_fp8_e4m3* fp8_dst = params.out_fp8 + s_q_idx * params.n_wv_group * total_group_dims + wv_group_idx * total_group_dims;
            uint8_t* sf_byte_dst = reinterpret_cast<uint8_t*>(params.out_sf) + s_q_idx * params.stride_out_sf_s_q * 4 + wv_group_idx * params.stride_out_sf_wv_group * 4;

            for (int tile_idx = tid; tile_idx < num_scale_tiles; tile_idx += blockDim.x) {
                int tile_head = tile_idx / (params.d_v / 32);
                if (tile_head == head_in_group) {
                    int tile_offset_in_head = (tile_idx % (params.d_v / 32)) * 32;
                    float max_val = 1e-8f;
                    for (int c = 0; c < 32; ++c) {
                        max_val = fmaxf(max_val, fabsf(smem_o_head[tile_offset_in_head + c]));
                    }
                    float scale = max_val / 448.0f;
                    float inv_scale = 1.0f / scale;

                    int global_tile_offset = tile_head * params.d_v + tile_offset_in_head;
                    for (int c = 0; c < 32; ++c) {
                        float v = smem_o_head[tile_offset_in_head + c] * inv_scale;
                        fp8_dst[global_tile_offset + c] = __nv_fp8_e4m3(__float2half(v));
                    }

                    sf_byte_dst[tile_idx] = float_to_e8m0_byte(scale);
                }
            }
            __syncthreads();
        }
    }
}

void run_fused_prefill_sm89(const PrefillParams &params) {
    fused_prefill_kernel_sm89<<<params.s_q, 128, 0, params.stream>>>(params);
}

void run_fused_decode_sm89(const DecodeParams &params) {
    fused_decode_kernel_sm89<<<params.s_q, 128, 0, params.stream>>>(params);
}

}  // namespace sm89::fused
