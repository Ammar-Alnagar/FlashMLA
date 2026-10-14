#include "sparse_prefill_sm89.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm89::sparse_prefill {

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// 1 block processes (1 query token, 1 query head) using 128 threads
__global__ void __launch_bounds__(128) sparse_prefill_kernel_sm89(const SparseAttnFwdParams params) {
    int s_q_idx = blockIdx.x;
    int h_q_idx = blockIdx.y;

    if (s_q_idx >= params.s_q || h_q_idx >= params.h_q) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int max_topk = params.topk;
    if (params.topk_length != nullptr) {
        int user_topk = params.topk_length[s_q_idx];
        if (user_topk < max_topk) {
            max_topk = user_topk;
        }
    }

    // Load Q vector [512] for (s_q_idx, h_q_idx) into registers
    const nv_bfloat16* q_base = reinterpret_cast<const nv_bfloat16*>(params.q);
    const nv_bfloat16* q_ptr = q_base + s_q_idx * params.stride_q_s_q + h_q_idx * params.stride_q_h_q;

    nv_bfloat16 q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = tid * 4 + i;
        q_reg[i] = (d < params.d_qk) ? q_ptr[d] : __float2bfloat16(0.0f);
    }

    const int* idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;
    const nv_bfloat16* kv_base = reinterpret_cast<const nv_bfloat16*>(params.kv);

    float max_score = -CUDART_INF_F;

    // First pass: find max score over all valid topk
    for (int t = 0; t < max_topk; ++t) {
        int kv_idx = idx_ptr[t];
        if (kv_idx >= 0 && kv_idx < params.s_kv) {
            const nv_bfloat16* k_ptr = kv_base + kv_idx * params.stride_kv_s_kv;
            float dot_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = tid * 4 + i;
                if (d < params.d_qk) {
                    float q_val = __bfloat162float(q_reg[i]);
                    float k_val = __bfloat162float(k_ptr[d]);
                    dot_sum += q_val * k_val;
                }
            }
            dot_sum = warp_reduce_sum(dot_sum);

            __shared__ float warp_dots[4];
            if (lane_id == 0) {
                warp_dots[warp_id] = dot_sum;
            }
            __syncthreads();

            if (tid == 0) {
                float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
                float score = total_dot * params.sm_scale;
                if (score > max_score) {
                    max_score = score;
                }
            }
            __syncthreads();
        }
    }

    __shared__ float smem_max_score;
    if (tid == 0) smem_max_score = max_score;
    __syncthreads();
    max_score = smem_max_score;

    float sum_exp = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (max_score != -CUDART_INF_F) {
        // Second pass: compute softmax & accumulate output
        for (int t = 0; t < max_topk; ++t) {
            int kv_idx = idx_ptr[t];
            if (kv_idx >= 0 && kv_idx < params.s_kv) {
                const nv_bfloat16* k_ptr = kv_base + kv_idx * params.stride_kv_s_kv;
                float dot_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int d = tid * 4 + i;
                    if (d < params.d_qk) {
                        float q_val = __bfloat162float(q_reg[i]);
                        float k_val = __bfloat162float(k_ptr[d]);
                        dot_sum += q_val * k_val;
                    }
                }
                dot_sum = warp_reduce_sum(dot_sum);

                __shared__ float warp_dots[4];
                if (lane_id == 0) warp_dots[warp_id] = dot_sum;
                __syncthreads();

                float score = (warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3]) * params.sm_scale;
                float weight = expf(score - max_score);
                sum_exp += weight;

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int d = tid * 4 + i;
                    if (d < params.d_v) {
                        float v_val = __bfloat162float(k_ptr[d]);
                        o_acc[i] += weight * v_val;
                    }
                }
                __syncthreads();
            }
        }
    }

    // Write outputs
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

    nv_bfloat16* out_base = reinterpret_cast<nv_bfloat16*>(params.out);
    nv_bfloat16* out_ptr = out_base + s_q_idx * (params.h_q * params.d_v) + h_q_idx * params.d_v;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = tid * 4 + i;
        if (d < params.d_v) {
            out_ptr[d] = __float2bfloat16(o_acc[i] * final_scale);
        }
    }
}

void run_sparse_prefill_sm89(const SparseAttnFwdParams &params) {
    dim3 grid(params.s_q, params.h_q);
    dim3 block(128);
    sparse_prefill_kernel_sm89<<<grid, block, 0, params.stream>>>(params);
}

}  // namespace sm89::sparse_prefill
