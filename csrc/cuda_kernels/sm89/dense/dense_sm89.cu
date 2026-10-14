/*
SM89 (Ada Lovelace) dense (non-MLA) MHA prefill, forward and backward.

Shapes: q is [total_qo_tokens, num_qo_heads, head_dim_qk] bf16, k / v are
[total_kv_tokens, num_kv_heads, *] bf16, with `head_dim_vo == 128` and `head_dim_qk in {128, 192}`
(the pairs the sm100 CUTLASS FMHA is instantiated for). Batches are described by cumulative
sequence lengths; `mask_mode_code == 1` means causal, bottom-right aligned: query token `i` of a
sequence attends to KV tokens `[0, kv_len - q_len + i]`.

Layout: one CTA per (query token, head), with every cross-thread quantity reduced through shared
memory and the barriers written out explicitly. This is a correctness-first path - the sm100
backend uses CUTLASS FMHA with TMA, warp specialisation and tcgen05, and is two to three orders of
magnitude faster. The sparse MLA operators, which are what FlashMLA exists for, do use tensor
cores on sm89 as well; see ../sparse/ and the note in docs/support_matrix.md.
*/
#include "dense_sm89.h"

#include "cuda_kernels/common/types.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <kerutils/kerutils.cuh>
#include <math_constants.h>
#include <torch/extension.h>

namespace sm89::dense {

namespace {

constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = NUM_THREADS / 32;

// P and dS of one query token are cached in shared memory; the backward requires
// max_seqlen_kv <= MAX_CACHED_KV (checked host-side). 2 * 8192 floats = 64 KB, within the 99 KB
// an sm_89 block may opt into.
constexpr int MAX_CACHED_KV = 8192;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// Block-wide sum; every thread receives the total. `smem` needs NUM_WARPS floats and must not be
// reused by the caller until this function has returned.
__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) total += smem[w];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float load_bf16(const bf16 *p) {
    __nv_bfloat16 v;
    memcpy(&v, p, sizeof(v));
    return __bfloat162float(v);
}

__device__ __forceinline__ void store_bf16(bf16 *p, float v) {
    const __nv_bfloat16 b = __float2bfloat16(v);
    memcpy(p, &b, sizeof(b));
}

struct Row {
    int kv_start;
    int kv_len;
    int q_pos;
};

// The batch owning global query token `token`. The scan is bounded by `num_batches`; the unbounded
// `while (cu_seqlens[b + 1] <= token) b++` would run off the end when a batch has zero length.
__device__ __forceinline__ Row get_row(
    const int *__restrict__ cu_seqlens_q, const int *__restrict__ cu_seqlens_kv,
    int num_batches, int token
) {
    __shared__ int s_batch[1];
    if (threadIdx.x == 0) {
        int b = 0;
        while (b + 1 < num_batches && cu_seqlens_q[b + 1] <= token) ++b;
        s_batch[0] = b;
    }
    __syncthreads();
    const int b = s_batch[0];
    Row r;
    r.q_pos = token - cu_seqlens_q[b];
    r.kv_start = cu_seqlens_kv[b];
    r.kv_len = cu_seqlens_kv[b + 1] - r.kv_start;
    return r;
}

// ---- forward ----------------------------------------------------------------------------

template<int HEAD_DIM_QK, int HEAD_DIM_VO>
__global__ void __launch_bounds__(NUM_THREADS)
fmha_dense_fwd_kernel_sm89(
    const bf16 *__restrict__ q, const bf16 *__restrict__ k, const bf16 *__restrict__ v,
    const int *__restrict__ cu_seqlens_q, const int *__restrict__ cu_seqlens_kv,
    bf16 *__restrict__ o, float *__restrict__ lse,
    int mask_mode_code, float softmax_scale, int num_batches,
    int total_qo_tokens, int num_qo_heads, int num_kv_heads
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    if (token >= total_qo_tokens) return;

    const Row row = get_row(cu_seqlens_q, cu_seqlens_kv, num_batches, token);
    const int kv_head = head / (num_qo_heads / num_kv_heads);
    const int causal_limit = row.kv_len - row.q_pos;

    const bf16 *q_row = q + (size_t)token * num_qo_heads * HEAD_DIM_QK + (size_t)head * HEAD_DIM_QK;
    __shared__ float smem[NUM_WARPS];
    // The row max is reduced by thread 0 alone, so it has to travel through shared memory: a
    // register write is not visible to the other threads of the block.
    __shared__ float s_max_score;
    if (threadIdx.x == 0) s_max_score = -CUDART_INF_F;
    __syncthreads();

    // pass 1: the row max
    for (int j = 0; j < row.kv_len; ++j) {
        if (mask_mode_code == 1 && j > causal_limit) continue;
        const bf16 *k_row = k + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_QK + (size_t)kv_head * HEAD_DIM_QK;
        float acc = 0.0f;
        for (int d = threadIdx.x; d < HEAD_DIM_QK; d += NUM_THREADS) {
            acc = fmaf(load_bf16(q_row + d), load_bf16(k_row + d), acc);
        }
        const float total = block_reduce_sum(acc, smem) * softmax_scale;
        if (threadIdx.x == 0) s_max_score = fmaxf(s_max_score, total);
        __syncthreads();    // every thread can now read the running max
    }
    const float max_score = s_max_score;

    // pass 2: softmax-weighted accumulation, one output dim chunk at a time
    float sum_exp = 0.0f;
    for (int idx = 0; idx < (HEAD_DIM_VO + NUM_THREADS - 1) / NUM_THREADS; ++idx) {
        const int d = threadIdx.x + idx * NUM_THREADS;
        if (d >= HEAD_DIM_VO) continue;
        float acc = 0.0f;
        for (int j = 0; j < row.kv_len; ++j) {
            if (mask_mode_code == 1 && j > causal_limit) continue;
            const bf16 *k_row = k + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_QK + (size_t)kv_head * HEAD_DIM_QK;
            float dot = 0.0f;
            for (int dd = threadIdx.x; dd < HEAD_DIM_QK; dd += NUM_THREADS) {
                dot = fmaf(load_bf16(q_row + dd), load_bf16(k_row + dd), dot);
            }
            const float score = dot * softmax_scale;
            const float w = (score == -CUDART_INF_F) ? 0.0f : expf(score - max_score);
            sum_exp += w;
            const bf16 *v_row = v + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_VO + (size_t)kv_head * HEAD_DIM_VO;
            acc = fmaf(w, load_bf16(v_row + d), acc);
        }
        store_bf16(o + (size_t)token * num_qo_heads * HEAD_DIM_VO + (size_t)head * HEAD_DIM_VO + d,
                   (max_score == -CUDART_INF_F) ? 0.0f : acc / sum_exp);
    }

    if (threadIdx.x == 0) {
        lse[(size_t)head * total_qo_tokens + token] =
            (max_score == -CUDART_INF_F) ? CUDART_INF_F : (max_score + logf(sum_exp));
    }
}

// ---- backward ---------------------------------------------------------------------------

// di[t, h] = sum_dim dO[t, h, dim] * O[t, h, dim]
__global__ void __launch_bounds__(NUM_THREADS)
fmha_dense_bwd_di_kernel_sm89(
    const bf16 *__restrict__ d_o, const bf16 *__restrict__ o, float *__restrict__ di,
    int total_qo_tokens, int num_qo_heads, int head_dim_v
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    if (token >= total_qo_tokens) return;

    const bf16 *do_row = d_o + (size_t)token * num_qo_heads * head_dim_v + (size_t)head * head_dim_v;
    const bf16 *o_row = o + (size_t)token * num_qo_heads * head_dim_v + (size_t)head * head_dim_v;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim_v; d += NUM_THREADS) {
        acc = fmaf(load_bf16(do_row + d), load_bf16(o_row + d), acc);
    }
    __shared__ float smem[NUM_WARPS];
    acc = block_reduce_sum(acc, smem);
    if (threadIdx.x == 0) di[(size_t)head * total_qo_tokens + token] = acc;
}

// One CTA per (query token, head). dQ is written directly (the CTA owns the whole row); dK and dV
// go into an fp32 workspace through float atomics and are cast to bf16 afterwards, which keeps the
// accumulation accurate (a bf16 atomicAdd over thousands of query tokens would not be).
template<int HEAD_DIM_QK, int HEAD_DIM_VO>
__global__ void __launch_bounds__(NUM_THREADS)
fmha_dense_bwd_kernel_sm89(
    const bf16 *__restrict__ d_o, const bf16 *__restrict__ q, const bf16 *__restrict__ k,
    const bf16 *__restrict__ v, const float *__restrict__ lse, const float *__restrict__ di,
    const int *__restrict__ cu_seqlens_q, const int *__restrict__ cu_seqlens_kv,
    bf16 *__restrict__ dq, float *__restrict__ dk_accum, float *__restrict__ dv_accum,
    int mask_mode_code, float softmax_scale, int num_batches,
    int total_qo_tokens, int num_qo_heads, int num_kv_heads
) {
    extern __shared__ float s_p[];    // [MAX_CACHED_KV] then [MAX_CACHED_KV] for dS

    const int token = blockIdx.x;
    const int head = blockIdx.y;
    if (token >= total_qo_tokens) return;

    const Row row = get_row(cu_seqlens_q, cu_seqlens_kv, num_batches, token);
    const int kv_head = head / (num_qo_heads / num_kv_heads);
    const int causal_limit = row.kv_len - row.q_pos;
    const int cached = row.kv_len;      // <= MAX_CACHED_KV, enforced host-side
    float *s_ds = s_p + MAX_CACHED_KV;

    const float lse_val = lse[(size_t)head * total_qo_tokens + token];
    const float di_val = di[(size_t)head * total_qo_tokens + token];
    const bf16 *q_row = q + (size_t)token * num_qo_heads * HEAD_DIM_QK + (size_t)head * HEAD_DIM_QK;
    const bf16 *do_row = d_o + (size_t)token * num_qo_heads * HEAD_DIM_VO + (size_t)head * HEAD_DIM_VO;
    __shared__ float smem[NUM_WARPS];

    // 1) P = exp(S - LSE), cached per key
    for (int j0 = 0; j0 < cached; j0 += NUM_THREADS) {
        const int j = j0 + threadIdx.x;
        const bool attend = (j < cached) && !(mask_mode_code == 1 && j > causal_limit);
        float p = 0.0f;
        if (attend) {
            const bf16 *k_row = k + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_QK + (size_t)kv_head * HEAD_DIM_QK;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < HEAD_DIM_QK; d += NUM_THREADS) {
                dot = fmaf(load_bf16(q_row + d), load_bf16(k_row + d), dot);
            }
            const float score = dot * softmax_scale;
            p = (lse_val == CUDART_INF_F) ? 0.0f : expf(score - lse_val);
        }
        if (attend) s_p[j] = p;
    }
    __syncthreads();

    // 2) dP, dS, and the dK / dV scatter
    for (int j = 0; j < cached; ++j) {
        if (mask_mode_code == 1 && j > causal_limit) continue;
        const float p = s_p[j];
        const bf16 *k_row = k + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_QK + (size_t)kv_head * HEAD_DIM_QK;
        const bf16 *v_row = v + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_VO + (size_t)kv_head * HEAD_DIM_VO;

        float dp = 0.0f;
        for (int d = threadIdx.x; d < HEAD_DIM_VO; d += NUM_THREADS) {
            dp = fmaf(load_bf16(do_row + d), load_bf16(v_row + d), dp);
        }
        dp = block_reduce_sum(dp, smem);
        const float ds = p * (dp - di_val) * softmax_scale;
        if (threadIdx.x == 0) s_ds[j] = ds;

        float *dk_row = dk_accum + ((size_t)(row.kv_start + j) * num_kv_heads + kv_head) * HEAD_DIM_QK;
        float *dv_row = dv_accum + ((size_t)(row.kv_start + j) * num_kv_heads + kv_head) * HEAD_DIM_VO;
        for (int d = threadIdx.x; d < HEAD_DIM_QK; d += NUM_THREADS) {
            atomicAdd(&dk_row[d], ds * load_bf16(q_row + d));
        }
        for (int d = threadIdx.x; d < HEAD_DIM_VO; d += NUM_THREADS) {
            atomicAdd(&dv_row[d], p * load_bf16(do_row + d));
        }
    }
    __syncthreads();

    // 3) dQ from the cached dS
    for (int idx = 0; idx < (HEAD_DIM_QK + NUM_THREADS - 1) / NUM_THREADS; ++idx) {
        const int d = threadIdx.x + idx * NUM_THREADS;
        if (d >= HEAD_DIM_QK) continue;
        float acc = 0.0f;
        for (int j = 0; j < cached; ++j) {
            if (mask_mode_code == 1 && j > causal_limit) continue;
            const bf16 *k_row = k + (size_t)(row.kv_start + j) * num_kv_heads * HEAD_DIM_QK + (size_t)kv_head * HEAD_DIM_QK;
            acc = fmaf(s_ds[j], load_bf16(k_row + d), acc);
        }
        store_bf16(dq + (size_t)token * num_qo_heads * HEAD_DIM_QK + (size_t)head * HEAD_DIM_QK + d, acc);
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
fmha_dense_cast_kernel_sm89(const float *__restrict__ src, bf16 *__restrict__ dst, int64_t numel) {
    const int64_t i = (int64_t)blockIdx.x * NUM_THREADS + threadIdx.x;
    if (i < numel) store_bf16(dst + i, src[i]);
}

// ---- launchers --------------------------------------------------------------------------

template<int HEAD_DIM_QK, int HEAD_DIM_VO>
void run_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
             const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_kv,
             const at::Tensor &o, const at::Tensor &lse,
             int mask_mode_code, float softmax_scale, int num_batches, cudaStream_t stream)
{
    dim3 grid((unsigned)q.size(0), (unsigned)q.size(1));
    fmha_dense_fwd_kernel_sm89<HEAD_DIM_QK, HEAD_DIM_VO><<<grid, NUM_THREADS, 0, stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(),
        cu_seqlens_q.data_ptr<int>(), cu_seqlens_kv.data_ptr<int>(),
        (bf16*)o.data_ptr(), lse.data_ptr<float>(),
        mask_mode_code, softmax_scale, num_batches,
        (int)q.size(0), (int)q.size(1), (int)k.size(1));
    KU_CHECK_KERNEL_LAUNCH();
}

template<int HEAD_DIM_QK, int HEAD_DIM_VO>
void run_bwd(const at::Tensor &d_o, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
             const at::Tensor &lse, const at::Tensor &di,
             const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_kv,
             const at::Tensor &dq, const at::Tensor &dk_accum, const at::Tensor &dv_accum,
             int mask_mode_code, float softmax_scale, int num_batches, cudaStream_t stream)
{
    const size_t smem_bytes = 2 * MAX_CACHED_KV * sizeof(float);
    static const bool configured = [&] {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            fmha_dense_bwd_kernel_sm89<HEAD_DIM_QK, HEAD_DIM_VO>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
        return true;
    }();
    (void)configured;

    dim3 grid((unsigned)q.size(0), (unsigned)q.size(1));
    fmha_dense_bwd_kernel_sm89<HEAD_DIM_QK, HEAD_DIM_VO><<<grid, NUM_THREADS, smem_bytes, stream>>>(
        (const bf16*)d_o.data_ptr(), (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(),
        (const bf16*)v.data_ptr(), lse.data_ptr<float>(), di.data_ptr<float>(),
        cu_seqlens_q.data_ptr<int>(), cu_seqlens_kv.data_ptr<int>(),
        (bf16*)dq.data_ptr(), dk_accum.data_ptr<float>(), dv_accum.data_ptr<float>(),
        mask_mode_code, softmax_scale, num_batches,
        (int)q.size(0), (int)q.size(1), (int)k.size(1));
    KU_CHECK_KERNEL_LAUNCH();
}

void cast_to_bf16(const at::Tensor &src, const at::Tensor &dst, cudaStream_t stream) {
    const int64_t numel = src.numel();
    if (numel == 0) return;
    fmha_dense_cast_kernel_sm89<<<(unsigned)((numel + NUM_THREADS - 1) / NUM_THREADS), NUM_THREADS, 0, stream>>>(
        src.data_ptr<float>(), (bf16*)dst.data_ptr(), numel);
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace

void FMHACutlassSM89FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                           at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_kv,
                           at::Tensor o, at::Tensor lse,
                           int mask_mode_code, float softmax_scale, int max_seqlen_q,
                           int max_seqlen_kv, bool is_varlen) {
    (void)workspace_buffer; (void)max_seqlen_q; (void)is_varlen;
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int num_batches = (int)cu_seqlens_q.size(0) - 1;
    const int head_dim_qk = (int)q.size(2);
    const int head_dim_vo = (int)v.size(2);

    TORCH_CHECK(head_dim_vo == 128, "[FlashMLA] SM89 dense fwd requires head_dim_v == 128, got ", head_dim_vo);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "[FlashMLA] SM89 dense fwd requires contiguous q/k/v");
    TORCH_CHECK(lse.stride(0) == 1, "[FlashMLA] SM89 dense fwd requires lse contiguous on the token dim");

    if (head_dim_qk == 128) {
        run_fwd<128, 128>(q, k, v, cu_seqlens_q, cu_seqlens_kv, o, lse, mask_mode_code, softmax_scale, num_batches, stream);
    } else if (head_dim_qk == 192) {
        run_fwd<192, 128>(q, k, v, cu_seqlens_q, cu_seqlens_kv, o, lse, mask_mode_code, softmax_scale, num_batches, stream);
    } else {
        TORCH_CHECK(false, "[FlashMLA] SM89 dense fwd supports head_dim_qk in {128, 192}, got ", head_dim_qk);
    }
}

void FMHACutlassSM89BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                           at::Tensor v, at::Tensor o, at::Tensor lse,
                           at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_kv,
                           at::Tensor dq, at::Tensor dk, at::Tensor dv,
                           int mask_mode_code, float softmax_scale, int max_seqlen_q,
                           int max_seqlen_kv, bool is_varlen) {
    (void)workspace_buffer; (void)max_seqlen_q; (void)is_varlen;
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int num_batches = (int)cu_seqlens_q.size(0) - 1;
    const int total_q = (int)q.size(0);
    const int num_qo_heads = (int)q.size(1);
    const int head_dim_qk = (int)q.size(2);
    const int head_dim_vo = (int)v.size(2);

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && d_o.is_contiguous(),
        "[FlashMLA] SM89 dense bwd requires contiguous q/k/v/d_o");
    TORCH_CHECK(max_seqlen_kv <= MAX_CACHED_KV, "[FlashMLA] SM89 dense bwd supports max_seqlen_kv <= ",
        MAX_CACHED_KV, ", got ", max_seqlen_kv);

    at::Tensor di = torch::empty({num_qo_heads, total_q}, q.options().dtype(torch::kFloat));
    {
        dim3 grid((unsigned)total_q, (unsigned)num_qo_heads);
        fmha_dense_bwd_di_kernel_sm89<<<grid, NUM_THREADS, 0, stream>>>(
            (const bf16*)d_o.data_ptr(), (const bf16*)o.data_ptr(), di.data_ptr<float>(),
            total_q, num_qo_heads, head_dim_vo);
        KU_CHECK_KERNEL_LAUNCH();
    }

    at::Tensor dk_accum = torch::zeros({k.size(0), k.size(1), head_dim_qk}, q.options().dtype(torch::kFloat));
    at::Tensor dv_accum = torch::zeros({v.size(0), v.size(1), head_dim_vo}, q.options().dtype(torch::kFloat));

    if (head_dim_qk == 128) {
        run_bwd<128, 128>(d_o, q, k, v, lse, di, cu_seqlens_q, cu_seqlens_kv, dq, dk_accum, dv_accum,
                          mask_mode_code, softmax_scale, num_batches, stream);
    } else if (head_dim_qk == 192) {
        run_bwd<192, 128>(d_o, q, k, v, lse, di, cu_seqlens_q, cu_seqlens_kv, dq, dk_accum, dv_accum,
                          mask_mode_code, softmax_scale, num_batches, stream);
    } else {
        TORCH_CHECK(false, "[FlashMLA] SM89 dense bwd supports head_dim_qk in {128, 192}, got ", head_dim_qk);
    }

    cast_to_bf16(dk_accum, dk, stream);
    cast_to_bf16(dv_accum, dv, stream);
}

}  // namespace sm89::dense