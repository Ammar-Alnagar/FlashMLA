/*
SM89 (Ada Lovelace) sparse attention decoding over a paged, quantized KV cache.

This is the main FlashMLA decode path. Q is [b, s_q, h_q, 512] bf16; the KV cache is
[num_blocks, page_block_size, 1, bytes_per_token], where a token stores 512 quantized dims
followed by its scales:

  * V4.1 fp8 (528 B/token): 512 x e4m3 + 16 x ue8m0, one scale per 32 dims.
  * V4.1 fp4 (288 B/token): 256 B of e2m1 (two values per byte, the even index in the low nibble)
    + 32 x e4m3, one scale per 16 dims. Only valid as the `extra_kv` next to an fp8 main cache.

A KV token is dequantized to bf16 straight into the shared-memory K tile that the tensor-core core
(see ../../common/flash_attn.cuh) streams over, so V never needs a second copy.

Split-KV reuses the architecture-independent scheduler (csrc/cuda_kernels/smxx) exactly like the
sm100 kernels: `gridDim.z` is the SM partition, `DecodingSchedMeta` gives its
[begin_req, end_req] x [begin_block, end_block) share, and a partition that owns only part of a
request writes its partial (log2-domain) LSE and unnormalised output into `lse_accum` /
`o_accum`; the shared combine kernel then reduces them. A partition that owns a whole request
writes `out` and `lse` itself. Note that on this path - unlike the sm100 TMA path - an index is
additionally bounds-checked against the cache size, so an out-of-range positive index is skipped
instead of dereferencing it.
*/
#include "sparse_decode_sm89.h"

#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm89/common/flash_attn.cuh"
#include "cuda_kernels/sm89/common/paged_kv_sm89.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <kerutils/kerutils.cuh>
#include <math_constants.h>
#include <torch/extension.h>

namespace sm89::sparse_decode {

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
constexpr int CHUNKS = D / 8;

constexpr size_t SMEM_BYTES =
    (size_t)(BLOCK_H + NUM_STAGES * BLOCK_N) * ROW * sizeof(__nv_bfloat16)
    + (size_t)NUM_STAGES * BLOCK_N * sizeof(int);

// ---- epilogue ---------------------------------------------------------------------------

// Write the result of one request. A partition that only owns part of the request contributes to
// the split-KV accumulators (log2-domain LSE, unnormalised output); otherwise the final bf16
// output and the natural-log LSE are written directly.
template<bool IS_SPLIT>
__device__ __forceinline__ void write_result(
    const SparseAttnDecodeParams &params, const Core &core, int req_idx, int s_q_idx,
    int head_base, int n_split_idx, int tid
) {
    if constexpr (IS_SPLIT) {
        #pragma unroll
        for (int j = 0; j < Core::D_TILES; ++j) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = core.lane_id / 4 + half * 8;
                const int h = head_base + core.m_tile * 16 + row;
                const int d0 = core.d_tile * D_PER_WARP + j * 8 + (core.lane_id & 3) * 2;
                float *dst = params.o_accum
                    + n_split_idx * params.stride_o_accum_split
                    + s_q_idx * params.stride_o_accum_s_q
                    + h * params.stride_o_accum_h_q
                    + d0;
                *reinterpret_cast<float2*>(dst) = make_float2(core.o[j][half * 2 + 0], core.o[j][half * 2 + 1]);
            }
        }
        if (core.lane_id % 4 == 0) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = core.lane_id / 4 + half * 8;
                const int h = head_base + core.m_tile * 16 + row;
                float *dst = params.lse_accum
                    + n_split_idx * params.stride_lse_accum_split
                    + s_q_idx * params.stride_lse_accum_s_q
                    + h;
                const float m = core.row_max[half];
                const float sum = core.row_sum[half];
                *dst = (m == -CUDART_INF_F || sum == 0.0f) ? -CUDART_INF_F : (m + __log2f(sum));
            }
        }
    } else {
        const int q_head = head_base + core.m_tile * 16 + core.lane_id / 4;
        const float sink2_lo = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[q_head] * CUDART_L2E_F;
        const float sink2_hi = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[q_head + 8] * CUDART_L2E_F;

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
                    val = core.o[j][e] / (sum * (1.0f + exp2f(sink2 - lse2)));
                }
                core.q_smem[(core.m_tile * 16 + row) * ROW + d] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        for (int c = tid; c < BLOCK_H * CHUNKS; c += NUM_THREADS) {
            const int r = c / CHUNKS;
            const int chunk = c % CHUNKS;
            bf16 *dst = params.out + req_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q
                + (head_base + r) * params.stride_o_h_q + chunk * 8;
            *reinterpret_cast<uint4*>(dst) = mma::ld_shared_u128(core.q_smem + r * ROW + chunk * 8);
        }

        if (core.lane_id % 4 == 0) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = core.lane_id / 4 + half * 8;
                const int h = head_base + core.m_tile * 16 + row;
                const float m = core.row_max[half];
                const float sum = core.row_sum[half];
                float *dst = params.lse + req_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + h;
                *dst = (m == -CUDART_INF_F || sum == 0.0f)
                    ? CUDART_INF_F
                    : (m + __log2f(sum)) * CUDART_LN2_F;
            }
        }
    }
}

// ---- kernel -----------------------------------------------------------------------------

template<ModelType ORIG_FORMAT, ModelType EXTRA_FORMAT>
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_decode_kernel_sm89(__grid_constant__ const SparseAttnDecodeParams params) {
    extern __shared__ __align__(16) char smem_raw[];

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;
    const int head_base = blockIdx.x * BLOCK_H;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;

    Core core;
    core.q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    core.k_smem = core.q_smem + BLOCK_H * ROW;
    core.valid_smem = reinterpret_cast<int*>(core.k_smem + NUM_STAGES * BLOCK_N * ROW);
    core.init(warp_id, tid, params.sm_scale_div_log2);

    DecodingSchedMeta sched_meta;
    if (params.enable_split_kv) {
        sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    } else {
        sched_meta.begin_req_idx = sched_meta.end_req_idx = partition_idx;
    }
    if (sched_meta.begin_req_idx >= params.b) return;

    const bool q_aligned = ((params.stride_q_s_q | params.stride_q_h_q) & 7) == 0;

    #pragma unroll 1
    for (int req_idx = sched_meta.begin_req_idx; req_idx <= sched_meta.end_req_idx; ++req_idx) {
        const int topk_length = params.topk_length ? __ldg(params.topk_length + req_idx) : params.topk;
        const int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + req_idx) : params.extra_topk;
        // Both caches are padded to a multiple of BLOCK_N so that a block never straddles the
        // main/extra boundary - the same invariant the shared scheduler relies on.
        const int num_orig_blocks = ((topk_length + BLOCK_N - 1) / BLOCK_N);
        const int total_blocks = num_orig_blocks + ((extra_topk_length + BLOCK_N - 1) / BLOCK_N);

        int start_block = 0, end_block = total_blocks;
        bool is_split = false;
        int n_split_idx = 0;
        if (params.enable_split_kv) {
            start_block = (req_idx == sched_meta.begin_req_idx) ? sched_meta.begin_block_idx : 0;
            end_block = (req_idx == sched_meta.end_req_idx) ? sched_meta.end_block_idx : total_blocks;
            is_split = (req_idx == sched_meta.begin_req_idx) ? sched_meta.is_first_req_splitted
                : ((req_idx == sched_meta.end_req_idx) ? sched_meta.is_last_req_splitted : false);
            n_split_idx = (req_idx == sched_meta.begin_req_idx)
                ? (__ldg(params.num_splits_ptr + req_idx) + sched_meta.begin_split_idx)
                : __ldg(params.num_splits_ptr + req_idx);
        }

        core.reset();

        // Q -> shared memory for this request
        for (int c = tid; c < BLOCK_H * CHUNKS; c += NUM_THREADS) {
            const int r = c / CHUNKS;
            const int chunk = c % CHUNKS;
            const bf16 *src = params.q
                + req_idx * params.stride_q_b + s_q_idx * params.stride_q_s_q
                + (head_base + r) * params.stride_q_h_q;
            __nv_bfloat16 *dst = core.q_smem + r * ROW + chunk * 8;
            if (q_aligned) {
                mma::cp_async_16(dst, src + chunk * 8, 16);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = mma::load_bf16(src + chunk * 8 + i);
            }
        }
        mma::cp_async_commit();

        const paged_kv::CacheView main_view = paged_kv::get_cache_view(params, req_idx, s_q_idx, false);
        const paged_kv::CacheView extra_view = paged_kv::get_cache_view(params, req_idx, s_q_idx, true);

        auto load_extra_tile = [&](int stage, int block_idx) {
            const int local_base = (block_idx - num_orig_blocks) * BLOCK_N;
            if constexpr (EXTRA_FORMAT == ModelType::V41_FP4) {
                paged_kv::load_k_tile_from_cache<ModelType::V41_FP4>(params, extra_view,
                    local_base, ROW, core.k_stage(stage), core.valid_stage(stage), tid);
            } else {
                paged_kv::load_k_tile_from_cache<ModelType::V41>(params, extra_view,
                    local_base, ROW, core.k_stage(stage), core.valid_stage(stage), tid);
            }
        };
        auto load_k_tile = [&](int stage, int tile_idx) {
            // `run()` numbers key tiles from 0, but a split partition starts at `start_block`.
            const int block_idx = start_block + tile_idx;
            if (block_idx < num_orig_blocks) {
                paged_kv::load_k_tile_from_cache<ORIG_FORMAT>(params, main_view,
                    block_idx * BLOCK_N, ROW, core.k_stage(stage), core.valid_stage(stage), tid);
            } else {
                load_extra_tile(stage, block_idx);
            }
        };

        core.run(end_block > start_block ? end_block - start_block : 0, load_k_tile);

        if (is_split) {
            write_result<true>(params, core, req_idx, s_q_idx, head_base, n_split_idx, tid);
        } else {
            write_result<false>(params, core, req_idx, s_q_idx, head_base, n_split_idx, tid);
        }
        __syncthreads();
    }
}

}  // namespace

void run_sparse_decode_sm89(const SparseAttnDecodeParams &params) {
    TORCH_CHECK(params.h_q % BLOCK_H == 0,
        "[FlashMLA] SM89 sparse decode requires h_q to be a multiple of ", BLOCK_H, ", got ", params.h_q);
    TORCH_CHECK(params.d_qk == D && params.d_v == D,
        "[FlashMLA] SM89 sparse decode requires d_qk == d_v == 512, got ", params.d_qk, " and ", params.d_v);
    TORCH_CHECK(SMEM_BYTES <= 101376, "[FlashMLA] SM89 sparse decode needs ", SMEM_BYTES,
        " bytes of shared memory, more than sm_89 provides");

    static const bool configured = [&] {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            sparse_decode_kernel_sm89<ModelType::V41, ModelType::V41>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            sparse_decode_kernel_sm89<ModelType::V41, ModelType::V41_FP4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        return true;
    }();
    (void)configured;

    const bool has_extra = params.extra_topk > 0 || params.extra_topk_length != nullptr;
    const int num_partitions = params.enable_split_kv ? params.num_sm_parts : params.b;

    dim3 grid(params.h_q / BLOCK_H, params.s_q, num_partitions);
    if (!has_extra || params.extra_model_type != ModelType::V41_FP4) {
        sparse_decode_kernel_sm89<ModelType::V41, ModelType::V41>
            <<<grid, NUM_THREADS, SMEM_BYTES, params.stream>>>(params);
    } else {
        sparse_decode_kernel_sm89<ModelType::V41, ModelType::V41_FP4>
            <<<grid, NUM_THREADS, SMEM_BYTES, params.stream>>>(params);
    }
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm89::sparse_decode