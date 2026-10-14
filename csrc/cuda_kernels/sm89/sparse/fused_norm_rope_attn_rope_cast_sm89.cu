/*
SM89 (Ada Lovelace) fused Norm + RoPE + Core Attention + RoPE + Cast.

Per query token (and per query head), in one pass:

  1. optional per-head RMSNorm over all 512 dims of Q,
  2. non-neox RoPE (rope_dim = 64) on the *last* 64 dims of Q, from `cos_sin_cache[token_position]`,
  3. sparse core attention (bf16 KV for prefill, paged fp8/fp4 KV cache for decoding),
  4. the conjugate RoPE on the last 64 dims of O,
  5. a per-32-dim e4m3 cast of O into the DeepGEMM "TMA-aligned, column-major, packed ue8m0"
     scale-factor layout that the following `deep_gemm.fp8_einsum` expects.

Steps 1-2 are folded into the Q tile that the shared tensor-core core streams over
(../../common/flash_attn.cuh); step 3 is that core; steps 4-5 are the epilogue. The output
permutation and the scale-factor exponent follow the sm100 kernel exactly:

    out_fp8 + s_q*(H_Q*D_VO) + wv_group*(WV_GROUP_SIZE*D_VO) + head_in_group*32 + dim_chunk*32*WV_GROUP_SIZE
    sf_block_idx = head_in_group + dim_chunk*WV_GROUP_SIZE
    out_sf byte  = *(uint8_t*)(out_sf + s_q + wv_group*stride_out_sf_wv_group
                                + (sf_block_idx/4)*stride_out_sf_head_dim) + sf_block_idx%4
    sf exponent  = 127 + floor(log2(max(abs_max, 1e-4) / 448))
*/
#include "fused_norm_rope_attn_rope_cast_sm89.h"

#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm89/common/flash_attn.cuh"
#include "cuda_kernels/sm89/common/paged_kv_sm89.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <kerutils/kerutils.cuh>
#include <math_constants.h>
#include <torch/extension.h>

namespace sm89::fused {

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
constexpr int NUM_DIM_CHUNKS = D / 32;

constexpr size_t SMEM_BYTES =
    (size_t)(BLOCK_H + NUM_STAGES * BLOCK_N) * ROW * sizeof(__nv_bfloat16)
    + (size_t)NUM_STAGES * BLOCK_N * sizeof(int);

constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4f;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// The rotation the reference applies; `conjugate` flips the sin sign (used for O).
__device__ __forceinline__ void rotate_pair(float &x, float &y, float c, float s, bool conjugate) {
    const float sv = conjugate ? -s : s;
    const float nx = x * c - y * sv;
    const float ny = x * sv + y * c;
    x = nx;
    y = ny;
}

// The ue8m0 exponent byte the sm100 kernel stores.
__device__ __forceinline__ uint8_t quant_ue8m0_exponent(float abs_max) {
    abs_max = fmaxf(abs_max, O_QUANT_CLAMP_MIN_VALUE);
    float sf = abs_max / 448.0f;
    uint32_t bits;
    memcpy(&bits, &sf, sizeof(bits));
    const int exp_sf = (int32_t)((bits - 1) >> 23) + (1 - 127);
    return (uint8_t)(exp_sf + 127);
}

// ---- Q preprocessing --------------------------------------------------------------------

// Optional RMSNorm over all 512 dims of each of the CTA's heads, followed by the non-neox RoPE on
// the last `rope_dim` dims. Operates in place on the Q tile that the core streams over.
template<typename ParamT>
__device__ __forceinline__ void preprocess_q(
    const ParamT &params, __nv_bfloat16 *q_smem, int token_position, int rope_dim, int tid
) {
    if (params.enable_q_norm) {
        __shared__ float red[NUM_WARPS];
        for (int r = 0; r < BLOCK_H; ++r) {
            float ssq = 0.0f;
            for (int c = tid; c < D; c += NUM_THREADS) {
                const float v = __bfloat162float(q_smem[r * ROW + c]);
                ssq = fmaf(v, v, ssq);
            }
            ssq = warp_reduce_sum(ssq);
            if ((tid & 31) == 0) red[tid / 32] = ssq;
            __syncthreads();
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) total += red[w];
            const float inv = rsqrtf(total / (float)D + params.rms_norm_eps);
            for (int c = tid; c < D; c += NUM_THREADS) {
                q_smem[r * ROW + c] = __float2bfloat16(__bfloat162float(q_smem[r * ROW + c]) * inv);
            }
            __syncthreads();
        }
    }

    const float *cos_sin = params.cos_sin_cache + (size_t)token_position * rope_dim;
    for (int r = 0; r < BLOCK_H; ++r) {
        for (int p = tid; p < rope_dim / 2; p += NUM_THREADS) {
            const int d0 = D - rope_dim + p * 2;
            float x = __bfloat162float(q_smem[r * ROW + d0]);
            float y = __bfloat162float(q_smem[r * ROW + d0 + 1]);
            rotate_pair(x, y, cos_sin[p], cos_sin[rope_dim / 2 + p], false);
            q_smem[r * ROW + d0] = __float2bfloat16(x);
            q_smem[r * ROW + d0 + 1] = __float2bfloat16(y);
        }
        __syncthreads();
    }
}

// ---- Q load -----------------------------------------------------------------------------

// `q` is the result of the q_b_proj GEMM run over the *permuted* weight, so its rows follow the
// layout that `permute_q_b_proj` produces: permuted row p holds the (head, dim) pair
//     p = d % 16 + h * 16 + (d / 16) * (h_q * 16),
// i.e. each run of 16 consecutive rows is one head's 16 dim-chunks. The attention core below wants
// q_smem row r to be head `head_base + r` with column c = dim c, so the permutation is undone here.
template<typename ParamT>
__device__ __forceinline__ void load_q_tile(
    const ParamT &params, __nv_bfloat16 *q_smem, int token_base, int head_base, int tid
) {
    const bool q_aligned = (params.stride_q_h_q & 7) == 0;
    for (int c = tid; c < BLOCK_H * CHUNKS; c += NUM_THREADS) {
        const int r = c / CHUNKS;
        const int chunk = c % CHUNKS;
        const int d = chunk * 8;
        const int h = head_base + r;
        const int p = (d % 16) + h * 16 + (d / 16) * (params.h_q * 16);
        const bf16 *src = params.q + token_base
            + (size_t)(p / D) * params.stride_q_h_q + (p % D);
        __nv_bfloat16 *dst = q_smem + r * ROW + d;
        if (q_aligned) {
            mma::cp_async_16(dst, src, 16);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) dst[i] = mma::load_bf16(src + i);
        }
    }
    mma::cp_async_commit();
}

// ---- epilogue ---------------------------------------------------------------------------

// Normalise the accumulated O, apply the conjugate RoPE on the last `rope_dim` dims, and write the
// result as e4m3 + packed ue8m0 scale factors in the layout the sm100 kernel produces.
template<typename ParamT>
__device__ __forceinline__ void write_quantized_out(
    const ParamT &params, const Core &core, int s_q_idx, int head_base,
    int token_position, int rope_dim, int tid
) {
    const float *cos_sin = params.cos_sin_cache + (size_t)token_position * rope_dim;
    const int q_head = head_base + core.m_tile * 16 + core.lane_id / 4;
    const float sink2_lo = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + q_head) * CUDART_L2E_F;
    const float sink2_hi = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + q_head + 8) * CUDART_L2E_F;

    // 1) normalise (+ attention sink) and stage the result in the (dead) Q tile
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

    // 2) conjugate RoPE on the last rope_dim dims
    if (rope_dim > 0) {
        for (int r = 0; r < BLOCK_H; ++r) {
            for (int p = tid; p < rope_dim / 2; p += NUM_THREADS) {
                const int d0 = D - rope_dim + p * 2;
                float x = __bfloat162float(core.q_smem[r * ROW + d0]);
                float y = __bfloat162float(core.q_smem[r * ROW + d0 + 1]);
                rotate_pair(x, y, cos_sin[p], cos_sin[rope_dim / 2 + p], true);
                core.q_smem[r * ROW + d0] = __float2bfloat16(x);
                core.q_smem[r * ROW + d0 + 1] = __float2bfloat16(y);
            }
            __syncthreads();
        }
    }

    // 3) quantize: one 32-dim chunk per (head, dim chunk) -> 32 e4m3 bytes + one scale byte
    const int wv_group_size = params.wv_group_size;
    const int num_chunks = BLOCK_H * NUM_DIM_CHUNKS;
    for (int c = tid; c < num_chunks; c += NUM_THREADS) {
        const int local_head = c / NUM_DIM_CHUNKS;
        const int dim_chunk = c - local_head * NUM_DIM_CHUNKS;
        const int h = head_base + local_head;
        const int wv_group = h / wv_group_size;
        const int head_in_group = h - wv_group * wv_group_size;

        __nv_bfloat16 vals[32];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i += 8) {
            const uint4 packed = mma::ld_shared_u128(core.q_smem + local_head * ROW + dim_chunk * 32 + i);
            *reinterpret_cast<uint4*>(vals + i) = packed;
        }
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(vals[i])));

        const uint8_t exp_byte = quant_ue8m0_exponent(amax);
        const float sf_inv = exp2f(-(static_cast<int>(exp_byte) - 127));

        __nv_fp8x2_storage_t fp8_vals[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float2 f;
            f.x = __bfloat162float(vals[i * 2 + 0]) * sf_inv;
            f.y = __bfloat162float(vals[i * 2 + 1]) * sf_inv;
            fp8_vals[i] = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
        }

        // 32 contiguous e4m3 bytes, at (head_in_group, dim_chunk) of this wv group
        fp8_e4m3 *fp8_dst = params.out_fp8
            + (size_t)s_q_idx * (params.h_q * D)
            + (size_t)wv_group * (wv_group_size * D)
            + head_in_group * 32
            + (size_t)dim_chunk * 32 * wv_group_size;
        uint4 *fp8_dst_u4 = reinterpret_cast<uint4*>(fp8_dst);
        const uint4 *fp8_vals_u4 = reinterpret_cast<const uint4*>(fp8_vals);
        fp8_dst_u4[0] = fp8_vals_u4[0];
        fp8_dst_u4[1] = fp8_vals_u4[1];

        const int sf_block_idx = head_in_group + dim_chunk * wv_group_size;
        uint8_t *sf_dst = reinterpret_cast<uint8_t*>(
            params.out_sf + s_q_idx
            + (uint32_t)wv_group * params.stride_out_sf_wv_group
            + (sf_block_idx / 4) * params.stride_out_sf_head_dim)
            + (sf_block_idx % 4);
        *sf_dst = exp_byte;
    }
}

// `max_logits` only exists for the prefill parameters; the decode variant returns just the LSE.
template<typename ParamT>
__device__ __forceinline__ void write_lse(const ParamT &params, const Core &core, int s_q_idx, int head_base) {
    if (core.lane_id % 4 != 0) return;
    constexpr bool IS_PREFILL = std::is_base_of_v<SparseAttnFwdParams, ParamT>;
    int lse_base;
    if constexpr (IS_PREFILL) {
        lse_base = s_q_idx * params.h_q;
    } else {
        lse_base = s_q_idx * params.stride_lse_s_q;
    }
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = core.lane_id / 4 + half * 8;
        const int idx = lse_base + head_base + core.m_tile * 16 + row;
        const float m = core.row_max[half];
        const float sum = core.row_sum[half];
        params.lse[idx] = (m == -CUDART_INF_F || sum == 0.0f) ? CUDART_INF_F : (m + __log2f(sum)) * CUDART_LN2_F;
        if constexpr (IS_PREFILL) {
            params.max_logits[idx] = (m == -CUDART_INF_F) ? -CUDART_INF_F : m * CUDART_LN2_F;
        }
    }
}

// ---- prefill ----------------------------------------------------------------------------

__global__ void __launch_bounds__(NUM_THREADS, 1)
fused_prefill_kernel_sm89(__grid_constant__ const PrefillParams params) {
    extern __shared__ __align__(16) char smem_raw[];

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;
    const int num_head_blocks = params.h_q / BLOCK_H;
    const int s_q_idx = blockIdx.x / num_head_blocks;
    const int head_base = (blockIdx.x % num_head_blocks) * BLOCK_H;

    Core core;
    core.q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    core.k_smem = core.q_smem + BLOCK_H * ROW;
    core.valid_smem = reinterpret_cast<int*>(core.k_smem + NUM_STAGES * BLOCK_N * ROW);
    core.init(warp_id, tid, params.sm_scale_div_log2);

    load_q_tile(params, core.q_smem, s_q_idx * params.stride_q_s_q, head_base, tid);
    mma::cp_async_wait_all();
    __syncthreads();

    preprocess_q(params, core.q_smem, __ldg(params.token_positions + s_q_idx), params.rope_dim, tid);

    int topk_eff = params.topk;
    if (params.topk_length != nullptr) {
        const int user_topk = __ldg(params.topk_length + s_q_idx);
        if (user_topk < topk_eff) topk_eff = user_topk;
    }
    const int num_key_tiles = (topk_eff + BLOCK_N - 1) / BLOCK_N;
    const int *idx_ptr = params.indices + s_q_idx * params.stride_indices_s_q;
    const bool kv_aligned = ((params.stride_kv_s_kv | params.stride_kv_h_kv) & 7) == 0;

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

    write_quantized_out(params, core, s_q_idx, head_base,
        __ldg(params.token_positions + s_q_idx), params.rope_dim, tid);
    write_lse(params, core, s_q_idx, head_base);
}

// ---- decode -----------------------------------------------------------------------------

template<ModelType EXTRA_FORMAT>
__global__ void __launch_bounds__(NUM_THREADS, 1)
fused_decode_kernel_sm89(__grid_constant__ const DecodeParams params) {
    extern __shared__ __align__(16) char smem_raw[];

    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;
    const int num_head_blocks = params.h_q / BLOCK_H;
    const int s_q_idx = blockIdx.x / num_head_blocks;
    const int head_base = (blockIdx.x % num_head_blocks) * BLOCK_H;

    Core core;
    core.q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    core.k_smem = core.q_smem + BLOCK_H * ROW;
    core.valid_smem = reinterpret_cast<int*>(core.k_smem + NUM_STAGES * BLOCK_N * ROW);
    core.init(warp_id, tid, params.sm_scale_div_log2);

    load_q_tile(params, core.q_smem, s_q_idx * params.stride_q_s_q, head_base, tid);
    mma::cp_async_wait_all();
    __syncthreads();

    preprocess_q(params, core.q_smem, __ldg(params.token_positions + s_q_idx), params.rope_dim, tid);

    const int topk_length = params.topk_length ? __ldg(params.topk_length) : params.topk;
    const int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length) : params.extra_topk;
    const int num_orig_blocks = (topk_length + BLOCK_N - 1) / BLOCK_N;
    const int total_blocks = num_orig_blocks + (extra_topk_length + BLOCK_N - 1) / BLOCK_N;

    const paged_kv::CacheView main_view = paged_kv::get_cache_view(params, 0, s_q_idx, false);
    const paged_kv::CacheView extra_view = paged_kv::get_cache_view(params, 0, s_q_idx, true);

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
    auto load_k_tile = [&](int stage, int block_idx) {
        if (block_idx < num_orig_blocks) {
            paged_kv::load_k_tile_from_cache<ModelType::V41>(params, main_view,
                block_idx * BLOCK_N, ROW, core.k_stage(stage), core.valid_stage(stage), tid);
        } else {
            load_extra_tile(stage, block_idx);
        }
    };

    core.run(total_blocks, load_k_tile);

    write_quantized_out(params, core, s_q_idx, head_base,
        __ldg(params.token_positions + s_q_idx), params.rope_dim, tid);
    write_lse(params, core, s_q_idx, head_base);
}

}  // namespace

void run_fused_prefill_sm89(const PrefillParams &params) {
    TORCH_CHECK(params.h_q % BLOCK_H == 0, "[FlashMLA] SM89 fused prefill requires h_q to be a multiple of ", BLOCK_H, ", got ", params.h_q);
    TORCH_CHECK(params.d_qk == D && params.d_v == D, "[FlashMLA] SM89 fused prefill requires d_qk == d_v == 512");
    TORCH_CHECK(params.is_rope_neox_style == false && params.rope_dim == 64,
        "[FlashMLA] SM89 fused prefill only supports non-neox RoPE with rope_dim == 64");
    TORCH_CHECK(params.wv_group_size == 8, "[FlashMLA] SM89 fused prefill requires wv_group_size == 8");
    TORCH_CHECK(SMEM_BYTES <= 101376, "[FlashMLA] SM89 fused prefill needs too much shared memory");

    static const bool configured = [&] {
        KU_CUDA_CHECK(cudaFuncSetAttribute(fused_prefill_kernel_sm89,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        KU_CUDA_CHECK(cudaFuncSetAttribute(fused_decode_kernel_sm89<ModelType::V41>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        KU_CUDA_CHECK(cudaFuncSetAttribute(fused_decode_kernel_sm89<ModelType::V41_FP4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        return true;
    }();
    (void)configured;

    fused_prefill_kernel_sm89<<<(unsigned)params.s_q * (unsigned)(params.h_q / BLOCK_H),
        NUM_THREADS, SMEM_BYTES, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

void run_fused_decode_sm89(const DecodeParams &params) {
    TORCH_CHECK(params.h_q % BLOCK_H == 0, "[FlashMLA] SM89 fused decode requires h_q to be a multiple of ", BLOCK_H, ", got ", params.h_q);
    TORCH_CHECK(params.d_qk == D && params.d_v == D, "[FlashMLA] SM89 fused decode requires d_qk == d_v == 512");
    TORCH_CHECK(params.is_rope_neox_style == false && params.rope_dim == 64,
        "[FlashMLA] SM89 fused decode only supports non-neox RoPE with rope_dim == 64");
    TORCH_CHECK(params.wv_group_size == 8, "[FlashMLA] SM89 fused decode requires wv_group_size == 8");

    const unsigned grid = (unsigned)params.s_q * (unsigned)(params.h_q / BLOCK_H);
    if (params.extra_model_type == ModelType::V41_FP4) {
        fused_decode_kernel_sm89<ModelType::V41_FP4><<<grid, NUM_THREADS, SMEM_BYTES, params.stream>>>(params);
    } else {
        fused_decode_kernel_sm89<ModelType::V41><<<grid, NUM_THREADS, SMEM_BYTES, params.stream>>>(params);
    }
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm89::fused