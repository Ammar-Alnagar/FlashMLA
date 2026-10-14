#include "dense_sm89.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm89::dense {

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Forward kernel for variable-length dense MHA prefill on SM89
__global__ void fmha_dense_fwd_kernel_sm89(
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int mask_mode_code,
    float softmax_scale,
    int total_qo_tokens,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim_qk,
    int head_dim_v
) {
    int tok_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (tok_idx >= total_qo_tokens || head_idx >= num_qo_heads) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int batch_idx = 0;
    while (cu_seqlens_q[batch_idx + 1] <= tok_idx) {
        batch_idx++;
    }

    int q_start = cu_seqlens_q[batch_idx];
    int pos_in_q = tok_idx - q_start;

    int kv_start = cu_seqlens_kv[batch_idx];
    int kv_len = cu_seqlens_kv[batch_idx + 1] - kv_start;

    int kv_head_idx = head_idx / (num_qo_heads / num_kv_heads);

    const nv_bfloat16* q_ptr = q + tok_idx * num_qo_heads * head_dim_qk + head_idx * head_dim_qk;

    float max_score = -CUDART_INF_F;

    // Pass 1: Max score calculation over KV sequence
    for (int j = 0; j < kv_len; ++j) {
        if (mask_mode_code == 1 && j > pos_in_q) continue;

        const nv_bfloat16* k_ptr = k + (kv_start + j) * num_kv_heads * head_dim_qk + kv_head_idx * head_dim_qk;
        float dot_sum = 0.0f;
        for (int d = tid; d < head_dim_qk; d += blockDim.x) {
            dot_sum += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        dot_sum = warp_reduce_sum(dot_sum);

        __shared__ float warp_dots[4];
        if (lane_id == 0) warp_dots[warp_id] = dot_sum;
        __syncthreads();

        if (tid == 0) {
            float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
            float score = total_dot * softmax_scale;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    __shared__ float smem_max;
    if (tid == 0) smem_max = max_score;
    __syncthreads();
    max_score = smem_max;

    float sum_exp = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (max_score != -CUDART_INF_F) {
        // Pass 2: Softmax weighting and output accumulation
        for (int j = 0; j < kv_len; ++j) {
            if (mask_mode_code == 1 && j > pos_in_q) continue;

            const nv_bfloat16* k_ptr = k + (kv_start + j) * num_kv_heads * head_dim_qk + kv_head_idx * head_dim_qk;
            const nv_bfloat16* v_ptr = v + (kv_start + j) * num_kv_heads * head_dim_v + kv_head_idx * head_dim_v;

            float dot_sum = 0.0f;
            for (int d = tid; d < head_dim_qk; d += blockDim.x) {
                dot_sum += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            }
            dot_sum = warp_reduce_sum(dot_sum);

            __shared__ float warp_dots[4];
            if (lane_id == 0) warp_dots[warp_id] = dot_sum;
            __syncthreads();

            float total_dot = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
            float score = total_dot * softmax_scale;
            float weight = expf(score - max_score);
            sum_exp += weight;

            for (int idx = 0; idx < 4; ++idx) {
                int d = tid + idx * blockDim.x;
                if (d < head_dim_v) {
                    o_acc[idx] += weight * __bfloat162float(v_ptr[d]);
                }
            }
            __syncthreads(); // Prevent race conditions on warp_dots across loop iterations
        }
    }

    if (tid == 0) {
        lse[tok_idx * num_qo_heads + head_idx] = (max_score == -CUDART_INF_F || sum_exp == 0.0f) ? CUDART_INF_F : (max_score + logf(sum_exp));
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int idx = 0; idx < 4; ++idx) {
        int d = tid + idx * blockDim.x;
        if (d < head_dim_v) {
            out[tok_idx * num_qo_heads * head_dim_v + head_idx * head_dim_v + d] = __float2bfloat16(o_acc[idx] * inv_sum);
        }
    }
}

// Backward kernel for variable-length dense MHA prefill on SM89
__global__ void fmha_dense_bwd_kernel_sm89(
    const nv_bfloat16* __restrict__ d_o,
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const nv_bfloat16* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_kv,
    nv_bfloat16* __restrict__ dq,
    nv_bfloat16* __restrict__ dk,
    nv_bfloat16* __restrict__ dv,
    int mask_mode_code,
    float softmax_scale,
    int total_qo_tokens,
    int total_kv_tokens,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim_qk,
    int head_dim_v
) {
    int tok_q_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (tok_q_idx >= total_qo_tokens || head_idx >= num_qo_heads) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int batch_idx = 0;
    while (cu_seqlens_q[batch_idx + 1] <= tok_q_idx) {
        batch_idx++;
    }

    int q_start = cu_seqlens_q[batch_idx];
    int pos_in_q = tok_q_idx - q_start;

    int kv_start = cu_seqlens_kv[batch_idx];
    int kv_len = cu_seqlens_kv[batch_idx + 1] - kv_start;
    int kv_head_idx = head_idx / (num_qo_heads / num_kv_heads);

    const nv_bfloat16* q_ptr = q + tok_q_idx * num_qo_heads * head_dim_qk + head_idx * head_dim_qk;
    const nv_bfloat16* do_ptr = d_o + tok_q_idx * num_qo_heads * head_dim_v + head_idx * head_dim_v;
    const nv_bfloat16* o_ptr = o + tok_q_idx * num_qo_heads * head_dim_v + head_idx * head_dim_v;
    float lse_val = lse[tok_q_idx * num_qo_heads + head_idx];

    float di_sum = 0.0f;
    for (int d = tid; d < head_dim_v; d += blockDim.x) {
        di_sum += __bfloat162float(do_ptr[d]) * __bfloat162float(o_ptr[d]);
    }
    di_sum = warp_reduce_sum(di_sum);
    __shared__ float warp_dis[4];
    if (lane_id == 0) warp_dis[warp_id] = di_sum;
    __syncthreads();

    float Di = warp_dis[0] + warp_dis[1] + warp_dis[2] + warp_dis[3];

    float dq_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int j = 0; j < kv_len; ++j) {
        if (mask_mode_code == 1 && j > pos_in_q) continue;

        int tok_kv_idx = kv_start + j;
        const nv_bfloat16* k_ptr = k + tok_kv_idx * num_kv_heads * head_dim_qk + kv_head_idx * head_dim_qk;
        const nv_bfloat16* v_ptr = v + tok_kv_idx * num_kv_heads * head_dim_v + kv_head_idx * head_dim_v;

        // Score Q * K
        float dot_sum = 0.0f;
        for (int d = tid; d < head_dim_qk; d += blockDim.x) {
            dot_sum += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        dot_sum = warp_reduce_sum(dot_sum);
        __shared__ float warp_dots[4];
        if (lane_id == 0) warp_dots[warp_id] = dot_sum;
        __syncthreads();

        float score = (warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3]) * softmax_scale;
        float p_val = expf(score - lse_val);

        // Compute dP = dO * V
        float dp_sum = 0.0f;
        for (int d = tid; d < head_dim_v; d += blockDim.x) {
            dp_sum += __bfloat162float(do_ptr[d]) * __bfloat162float(v_ptr[d]);
        }
        dp_sum = warp_reduce_sum(dp_sum);
        if (lane_id == 0) warp_dots[warp_id] = dp_sum;
        __syncthreads();

        float dP = warp_dots[0] + warp_dots[1] + warp_dots[2] + warp_dots[3];
        float dS = p_val * (dP - Di) * softmax_scale;

        for (int idx = 0; idx < 4; ++idx) {
            int d = tid + idx * blockDim.x;
            if (d < head_dim_qk) {
                dq_acc[idx] += dS * __bfloat162float(k_ptr[d]);
            }
        }

        nv_bfloat16* dk_ptr = dk + tok_kv_idx * num_kv_heads * head_dim_qk + kv_head_idx * head_dim_qk;
        nv_bfloat16* dv_ptr = dv + tok_kv_idx * num_kv_heads * head_dim_v + kv_head_idx * head_dim_v;

        for (int d = tid; d < head_dim_qk; d += blockDim.x) {
            atomicAdd(dk_ptr + d, __float2bfloat16(dS * __bfloat162float(q_ptr[d])));
        }
        for (int d = tid; d < head_dim_v; d += blockDim.x) {
            atomicAdd(dv_ptr + d, __float2bfloat16(p_val * __bfloat162float(do_ptr[d])));
        }
        __syncthreads(); // Prevent race conditions on warp_dots across loop iterations
    }

    // Write dQ
    nv_bfloat16* dq_ptr = dq + tok_q_idx * num_qo_heads * head_dim_qk + head_idx * head_dim_qk;
    for (int idx = 0; idx < 4; ++idx) {
        int d = tid + idx * blockDim.x;
        if (d < head_dim_qk) {
            dq_ptr[d] = __float2bfloat16(dq_acc[idx]);
        }
    }
}

void FMHACutlassSM89FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                           at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                           at::Tensor o, at::Tensor lse,
                           int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {
    (void)workspace_buffer; (void)max_seqlen_q; (void)max_seqlen_kv; (void)is_varlen;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int total_qo_tokens = q.size(0);
    int num_qo_heads = q.size(1);
    int head_dim_qk = q.size(2);
    int num_kv_heads = k.size(1);
    int head_dim_v = v.size(2);

    dim3 grid(total_qo_tokens, num_qo_heads);
    dim3 block(128);

    fmha_dense_fwd_kernel_sm89<<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(v.data_ptr()),
        cumulative_seqlen_q.data_ptr<int>(),
        cumulative_seqlen_kv.data_ptr<int>(),
        reinterpret_cast<nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        mask_mode_code,
        softmax_scale,
        total_qo_tokens,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v
    );
}

void FMHACutlassSM89BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                           at::Tensor v, at::Tensor o, at::Tensor lse,
                           at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                           at::Tensor dq, at::Tensor dk, at::Tensor dv,
                           int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {
    (void)workspace_buffer; (void)max_seqlen_q; (void)max_seqlen_kv; (void)is_varlen;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dk.zero_();
    dv.zero_();

    int total_qo_tokens = q.size(0);
    int total_kv_tokens = k.size(0);
    int num_qo_heads = q.size(1);
    int num_kv_heads = k.size(1);
    int head_dim_qk = q.size(2);
    int head_dim_v = v.size(2);

    dim3 grid(total_qo_tokens, num_qo_heads);
    dim3 block(128);

    fmha_dense_bwd_kernel_sm89<<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(d_o.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        cumulative_seqlen_q.data_ptr<int>(),
        cumulative_seqlen_kv.data_ptr<int>(),
        reinterpret_cast<nv_bfloat16*>(dq.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(dk.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(dv.data_ptr()),
        mask_mode_code,
        softmax_scale,
        total_qo_tokens,
        total_kv_tokens,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v
    );
}

}  // namespace sm89::dense
