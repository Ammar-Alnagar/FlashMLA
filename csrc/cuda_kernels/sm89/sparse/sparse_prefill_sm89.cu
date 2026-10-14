/*
SM89 (Ada Lovelace) sparse attention prefill.

Q is [s_q, h_q, 512] bf16, KV is [s_kv, h_kv, 512] bf16, and `indices` [s_q, h_kv, topk] int32
lists, per query token, the KV tokens to attend to. Out is [s_q, h_q, 512] bf16, together with
the per-head max logit and the natural-log LSE.

One CTA owns BLOCK_H = 32 query heads of one query token; the head-block index varies fastest so
that the CTAs sharing a query token - and therefore its KV - are co-scheduled and hit in L2.

See ../../common/flash_attn.cuh for the tensor-core core and its shared-memory budget.
*/
#include "sparse_prefill_sm89.h"

#include "cuda_kernels/sm89/common/flash_attn.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <kerutils/kerutils.cuh>
#include <math_constants.h>
#include <torch/extension.h>

namespace sm89::sparse_prefill {

namespace {

constexpr int D = 512;
constexpr int BLOCK_H = 32;
constexpr int BLOCK_N = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_STAGES = 2;

using Core = FlashAttnCore<BLOCK_H, BLOCK_N, NUM_WARPS, D, NUM_STAGES>;
using Dim = Core::Dim;

constexpr int NUM_THREADS = Core::NUM_THREADS;
constexpr int ROW = Dim::ROW;
constexpr int D_PER_WARP = Dim::D_PER_WARP;
constexpr int CHUNKS = D / 8;      // 16-byte chunks per 512-wide row

__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_prefill_kernel_sm89(__grid_constant__ const SparseAttnFwdParams params) {
    extern __shared__ __align__(16) char smem_raw[];

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Adjacent CTAs handle the head blocks of the same query token so that they share KV in L2.
    const int num_head_blocks = params.h_q / BLOCK_H;
    const int s_q_idx = blockIdx.x / num_head_blocks;
    const int head_base = (blockIdx.x % num_head_blocks) * BLOCK_H;

    Core core;
    core.q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    core.k_smem = core.q_smem + BLOCK_H * ROW;
    core.valid_smem = reinterpret_cast<int*>(core.k_smem + NUM_STAGES * BLOCK_N * ROW);
    core.init(warp_id, tid, params.sm_scale_div_log2);

    // A row of Q / KV is 16-byte loadable iff all the involved strides are multiples of 8 elements.
    const bool q_aligned = ((params.stride_q_s_q | params.stride_q_h_q) & 7) == 0;
    const bool kv_aligned = ((params.stride_kv_s_kv | params.stride_kv_h_kv) & 7) == 0;

    for (int c = tid; c < BLOCK_H * CHUNKS; c += NUM_THREADS) {
        const int r = c / CHUNKS;
        const int chunk = c % CHUNKS;
        const bf16 *src = params.q + s_q_idx * params.stride_q_s_q + (head_base + r) * params.stride_q_h_q;
        __nv_bfloat16 *dst = core.q_smem + r * ROW + chunk * 8;
        if (q_aligned) {
            mma::cp_async_16(dst, src + chunk * 8, 16);
        } else {
            dst[0] = mma::load_bf16(src + chunk * 8);
            #pragma unroll
            for (int i = 1; i < 8; ++i) dst[i] = mma::load_bf16(src + chunk * 8 + i);
        }
    }
    mma::cp_async_commit();

    int topk_eff = params.topk;
    if (params.topk_length != nullptr) {
        const int user_topk = params.topk_length[s_q_idx];
        if (user_topk < topk_eff) topk_eff = user_topk;
    }
    const int num_key_tiles = (topk_eff + BLOCK_N - 1) / BLOCK_N;
    const int *idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;

    auto load_k_tile = [&](int stage, int tile_idx) {
        __nv_bfloat16 *ks = core.k_stage(stage);
        int *valid = core.valid_stage(stage);
        const int base = tile_idx * BLOCK_N;

        for (int c = tid; c < BLOCK_N * CHUNKS; c += NUM_THREADS) {
            const int r = c / CHUNKS;
            const int chunk = c % CHUNKS;
            const int j = base + r;
            const int kv_idx = (j < topk_eff) ? idx_ptr[j] : 0;
            const bool ok = (j < topk_eff) && (kv_idx >= 0) && (kv_idx < params.s_kv);
            // A masked-out row must still be handed a dereferenceable source address even when
            // cp.async is asked to transfer zero bytes.
            const bf16 *src = params.kv + (ok ? kv_idx : 0) * params.stride_kv_s_kv;
            __nv_bfloat16 *dst = ks + r * ROW + chunk * 8;
            if (kv_aligned) {
                mma::cp_async_16(dst, src + chunk * 8, ok ? 16 : 0);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = mma::load_bf16(src + chunk * 8 + i);
            }
        }

        for (int r = tid; r < BLOCK_N; r += NUM_THREADS) {
            const int j = base + r;
            const int kv_idx = (j < topk_eff) ? idx_ptr[j] : 0;
            valid[r] = (j < topk_eff && kv_idx >= 0 && kv_idx < params.s_kv) ? 1 : 0;
        }
    };

    core.run(num_key_tiles, load_k_tile);

    // ---- epilogue ----
    // Each warp owns 16 heads x D_PER_WARP dims of O. Stage the normalised bf16 result in the
    // (now dead) Q tile so that the global stores are 16-byte and fully coalesced.
    const int q_head = head_base + core.m_tile * 16 + core.lane_id / 4;
    const float sink2_lo = params.attn_sink == nullptr ? -CUDART_INF_F
        : params.attn_sink[q_head] * CUDART_L2E_F;
    const float sink2_hi = params.attn_sink == nullptr ? -CUDART_INF_F
        : params.attn_sink[q_head + 8] * CUDART_L2E_F;

    #pragma unroll
    for (int j = 0; j < Core::D_TILES; ++j) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int half = e >> 1;
            const int row = (half == 0) ? core.lane_id / 4 : core.lane_id / 4 + 8;
            const int d = core.d_tile * D_PER_WARP + j * 8 + (core.lane_id & 3) * 2 + (e & 1);
            const float sum = core.row_sum[half];
            float val = 0.0f;
            if (sum > 0.0f) {
                const float lse2 = core.row_max[half] + __log2f(sum);
                const float sink2 = (half == 0) ? sink2_lo : sink2_hi;
                // attn_sink adds exp(sink) to the softmax denominator but contributes no value
                val = core.o[j][e] / (sum * (1.0f + exp2f(sink2 - lse2)));
            }
            core.q_smem[(core.m_tile * 16 + row) * ROW + d] = __float2bfloat16(val);
        }
    }
    __syncthreads();

    for (int c = tid; c < BLOCK_H * CHUNKS; c += NUM_THREADS) {
        const int r = c / CHUNKS;
        const int chunk = c % CHUNKS;
        bf16 *dst = params.out + s_q_idx * params.h_q * D + (head_base + r) * D + chunk * 8;
        uint4 v = mma::ld_shared_u128(core.q_smem + r * ROW + chunk * 8);
        *reinterpret_cast<uint4*>(dst) = v;
    }

    // max_logits / lse (natural log). Only the lanes that own a row write them.
    if (core.lane_id % 4 == 0) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = core.lane_id / 4 + half * 8;
            const int global_index = s_q_idx * params.h_q + head_base + core.m_tile * 16 + row;
            const float m = core.row_max[half];
            const float sum = core.row_sum[half];
            if (m == -CUDART_INF_F || sum == 0.0f) {
                params.max_logits[global_index] = -CUDART_INF_F;
                params.lse[global_index] = CUDART_INF_F;
            } else {
                params.max_logits[global_index] = m * CUDART_LN2_F;
                params.lse[global_index] = (m + __log2f(sum)) * CUDART_LN2_F;
            }
        }
    }
}

}  // namespace

void run_sparse_prefill_sm89(const SparseAttnFwdParams &params) {
    TORCH_CHECK(params.h_q % BLOCK_H == 0,
        "[FlashMLA] SM89 sparse prefill requires h_q to be a multiple of ", BLOCK_H, ", got ", params.h_q);
    TORCH_CHECK(params.d_qk == D && params.d_v == D,
        "[FlashMLA] SM89 sparse prefill requires d_qk == d_v == 512, got ", params.d_qk, " and ", params.d_v);

    constexpr size_t smem_bytes =
        (size_t)(BLOCK_H + NUM_STAGES * BLOCK_N) * ROW * sizeof(__nv_bfloat16)
        + (size_t)NUM_STAGES * BLOCK_N * sizeof(int);

    // sm_89 allows up to 101376 bytes of dynamic shared memory per block
    TORCH_CHECK(smem_bytes <= 101376, "[FlashMLA] SM89 sparse prefill needs ", smem_bytes,
        " bytes of shared memory, more than sm_89 provides");

    static const bool configured = [&] {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            sparse_prefill_kernel_sm89, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
        return true;
    }();
    (void)configured;

    const int num_head_blocks = params.h_q / BLOCK_H;
    const unsigned grid = (unsigned)params.s_q * (unsigned)num_head_blocks;
    sparse_prefill_kernel_sm89<<<grid, NUM_THREADS, smem_bytes, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm89::sparse_prefill