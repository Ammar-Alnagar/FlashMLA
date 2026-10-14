/*
Paged quantized KV cache -> bf16 shared-memory tile, for the SM89 (Ada Lovelace) kernels.

A V4.1 KV token stores 512 quantized dims followed by its scales:

  * fp8 (528 B/token): 512 x e4m3, then 16 x ue8m0, one scale per 32 dims.
  * fp4 (288 B/token): 256 B of e2m1 (two values per byte, the even index in the low nibble),
    then 32 x e4m3, one scale per 16 dims.

See tests/quant.py for the reference quantizer / dequantizer that this must agree with, and
`kv_cache_format.h` for the byte counts. The dequantized values are written straight into the
padded shared-memory K tile consumed by the tensor-core core, so MLA's V == K means no second copy
is needed.
*/
#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/common/types.h"

namespace sm89::paged_kv {

__device__ __forceinline__ float ue8m0_to_float(uint8_t b) {
    return exp2f(static_cast<float>(static_cast<int>(b) - 127));
}

__device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
    float v;
    switch (nibble & 7) {
        case 0: v = 0.0f; break;
        case 1: v = 0.5f; break;
        case 2: v = 1.0f; break;
        case 3: v = 1.5f; break;
        case 4: v = 2.0f; break;
        case 5: v = 3.0f; break;
        case 6: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return (nibble & 8) ? -v : v;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    // NOTE `__nv_fp8_e4m3(byte)` would select the *float* constructor and convert the numeric value
    // of the byte (e.g. 11 -> 11.0) instead of reinterpreting its bit pattern, so the storage type
    // has to go through the explicit raw converter. (sm_89 has `cvt.e4m3x2.f32` for this.)
    __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

// How one quantized token is spread over the threads of the CTA that dequantizes it.
template<ModelType MT>
struct PackedToken;

template<>
struct PackedToken<ModelType::V41> {
    static constexpr int THREADS_PER_TOKEN = 32;    // 32 * 16 B = 512 B of e4m3
    static constexpr int DIMS_PER_THREAD = 16;
    static constexpr int SCALE_OFFSET = 512;
};

template<>
struct PackedToken<ModelType::V41_FP4> {
    static constexpr int THREADS_PER_TOKEN = 16;    // 16 * 16 B = 256 B of e2m1, 32 dims each
    static constexpr int DIMS_PER_THREAD = 32;
    static constexpr int SCALE_OFFSET = 256;
};

// Dequantize one token into `dst`, a padded 512-wide shared-memory row. `lane_in_token` selects
// this thread's slice of the token: `DIMS_PER_THREAD` dims, which is 16 fp8 bytes (32 threads per
// token) or 32 fp4 dims packed two-per-byte (16 threads per token) - 16 input bytes either way. It
// expands to `DIMS_PER_THREAD` bf16, i.e. `DIMS_PER_THREAD / 8` 16-byte output chunks. `src` must
// be 16-byte aligned, which holds because every token stride is a multiple of 16 bytes (528 and
// 288) and the cache base comes from torch.
//
// The loops below are indexed by *output dimension* rather than by input byte, because the two
// formats pack differently: an fp8 output chunk of 8 dims consumes 8 input bytes, an fp4 chunk of
// 8 dims consumes only 4.
template<ModelType MT>
__device__ __forceinline__ void dequantize_token(const uint8_t *src, __nv_bfloat16 *dst, int lane_in_token) {
    using P = PackedToken<MT>;
    constexpr int OUT_CHUNKS = P::DIMS_PER_THREAD / 8;    // 16-byte bf16 chunks this thread writes

    const uint4 raw = *reinterpret_cast<const uint4*>(src + lane_in_token * 16);
    const uint8_t *bytes = reinterpret_cast<const uint8_t*>(&raw);
    const int first_dim = lane_in_token * P::DIMS_PER_THREAD;

    __nv_bfloat16 out[OUT_CHUNKS][8];
    if constexpr (MT == ModelType::V41) {
        // One ue8m0 scale per 32 dims, so two lanes share a scale.
        const float scale = ue8m0_to_float(src[P::SCALE_OFFSET + first_dim / 32]);
        #pragma unroll
        for (int i = 0; i < OUT_CHUNKS; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = i * 8 + j;    // dimension within this thread's slice
                out[i][j] = __float2bfloat16(e4m3_to_float(bytes[d]) * scale);
            }
        }
    } else {
        // One e4m3 scale per 16 dims, so this thread's 32 dims span exactly two scales.
        const int first_tile = first_dim / 16;
        const float scale_lo = e4m3_to_float(src[P::SCALE_OFFSET + first_tile]);
        const float scale_hi = e4m3_to_float(src[P::SCALE_OFFSET + first_tile + 1]);
        #pragma unroll
        for (int i = 0; i < OUT_CHUNKS; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = i * 8 + j;
                const uint8_t byte = bytes[d / 2];
                const uint8_t nibble = (d & 1) ? (byte >> 4) : (byte & 0x0F);
                out[i][j] = __float2bfloat16(e2m1_to_float(nibble) * (d < 16 ? scale_lo : scale_hi));
            }
        }
    }


    __nv_bfloat16 *dst_p = dst + lane_in_token * P::DIMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < OUT_CHUNKS; ++i) {
        uint4 packed;
        packed.x = reinterpret_cast<uint32_t&>(out[i][0]);
        packed.y = reinterpret_cast<uint32_t&>(out[i][2]);
        packed.z = reinterpret_cast<uint32_t&>(out[i][4]);
        packed.w = reinterpret_cast<uint32_t&>(out[i][6]);
        *reinterpret_cast<uint4*>(dst_p + i * 8) = packed;
    }
}

// Everything the sparse decode and fused kernels need to address one of the two caches.
struct CacheView {
    const uint8_t *bytes;
    int num_blocks;
    int page_block_size;
    int stride_block;
    int stride_row;
    const int *indices;
    int length;
};

template<typename ParamsT>
__device__ __forceinline__ CacheView get_cache_view(
    const ParamsT &params, int req_idx, int s_q_idx, bool is_extra
) {
    CacheView v;
    v.bytes = is_extra
        ? reinterpret_cast<const uint8_t*>(params.extra_kv)
        : reinterpret_cast<const uint8_t*>(params.kv);
    v.num_blocks = is_extra ? params.extra_num_blocks : params.num_blocks;
    v.page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
    v.stride_block = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    v.stride_row = is_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    v.indices = is_extra
        ? params.extra_indices + req_idx * params.stride_extra_indices_b + s_q_idx * params.stride_extra_indices_s_q
        : params.indices + req_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
    v.length = is_extra
        ? (params.extra_topk_length ? __ldg(params.extra_topk_length + req_idx) : params.extra_topk)
        : (params.topk_length ? __ldg(params.topk_length + req_idx) : params.topk);
    return v;
}

// Dequantize `block_n` tokens starting at logical index `local_base` into the shared-memory K tile
// and record which of them are attendable.
//
// An index is bounds-checked against the cache size (and not merely against -1), because the token
// address is dereferenced directly rather than through a TMA descriptor. A masked-out key still gets
// a row in the tile - `O += P @ V` multiplies it by P = 0, and `0 * NaN` is NaN, so the row must be
// *zeroed* rather than filled with whatever the token behind a clamped index happened to be. (The
// test suite deliberately poisons unselected cache tokens with NaN.) Zeroing also means an invalid
// index is never dereferenced at all.
template<ModelType MT, typename ParamsT, int BLOCK_N = 32, int NUM_THREADS = 256>
__device__ __forceinline__ void load_k_tile_from_cache(
    const ParamsT &params, const CacheView &v, int local_base, int row_stride,
    __nv_bfloat16 *ks, int *valid, int tid
) {
    using P = PackedToken<MT>;
    constexpr int OUT_CHUNKS = P::DIMS_PER_THREAD / 8;
    const int total_tokens = v.num_blocks * v.page_block_size;

    #pragma unroll 1
    for (int slot = tid; slot < BLOCK_N * P::THREADS_PER_TOKEN; slot += NUM_THREADS) {
        const int r = slot / P::THREADS_PER_TOKEN;
        const int lane_in_token = slot - r * P::THREADS_PER_TOKEN;
        const int token = (local_base + r < v.length) ? v.indices[local_base + r] : -1;
        const bool ok = (token >= 0) && (token < total_tokens);
        // NOTE `dequantize_token` applies the lane offset itself, so it takes the row base.
        __nv_bfloat16 *dst_row = ks + r * row_stride;
        if (ok) {
            const uint8_t *src = v.bytes + (token / v.page_block_size) * v.stride_block
                + (token % v.page_block_size) * v.stride_row;
            dequantize_token<MT>(src, dst_row, lane_in_token);
        } else {
            __nv_bfloat16 *dst = dst_row + lane_in_token * P::DIMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < OUT_CHUNKS; ++i) {
                *reinterpret_cast<uint4*>(dst + i * 8) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }

    for (int r = tid; r < BLOCK_N; r += NUM_THREADS) {
        const int token = (local_base + r < v.length) ? v.indices[local_base + r] : -1;
        valid[r] = (token >= 0) && (token < total_tokens) ? 1 : 0;
    }
}

}  // namespace sm89::paged_kv