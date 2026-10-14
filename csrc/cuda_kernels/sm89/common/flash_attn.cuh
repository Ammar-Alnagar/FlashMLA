/*
SM89 (Ada Lovelace) flash-attention core, shared by the sparse prefill / decode / fused kernels.

All SM89 sparse kernels share exactly the same mathematical core - MLA attention over a gathered
set of KV tokens, with Q always 512-dimensional and V == K - so they only differ in *where* a KV
token comes from and what is done to the result afterwards:

  - sparse_prefill          : KV is bf16 [s_kv, h_kv, 512], addressed by an index list
  - sparse_decode           : KV is a paged, per-32-dim fp8 (or per-16-dim fp4) quantized cache
  - fused_norm_rope_...     : same as above, plus RMSNorm/RoPE on Q and RoPE + fp8 cast on O

The core itself:

  * A CTA owns `BLOCK_H` query heads of one query token (`BLOCK_H = 32`, i.e. two 16-row mma
    tiles) and streams the token's KV in tiles of `BLOCK_N = 32` keys. Heads share KV, so a KV
    token is read once per `h_q / BLOCK_H` CTAs (4x for h_q = 128, 2x for h_q = 64).
  * `NUM_WARPS = 8` warps are laid out as (2 M-tiles) x (4 slices of 128 output dims). A warp
    computes the scores of the M-tile it owns over all `BLOCK_N` keys and then immediately
    consumes them as the A operand of `O[16 heads][its 128 dims] += P @ V`. Because the score
    accumulator layout of `mma.m16n8k16` *is* the A-operand layout of the next `mma`, P never has
    to move through shared memory, and O never has to be reduced across warps. With 512 value
    dims this keeps the register footprint at `16 * 128 / 32 = 64` floats per lane.
  * Softmax is online (a single streaming pass over the KV, no separate "max pass" / "sum pass"),
    in base 2 (`exp2f`), so that split-KV partial results are also in log2 - the convention of
    `csrc/cuda_kernels/smxx/decode/combine`.

Shared-memory budget (sm_89 allows 101376 bytes of dynamic shared memory per block):

    Q tile           BLOCK_H * (D + 8) * 2         = 32 * 520 * 2 =  33280 B
    K/V tile (x2)    2 * BLOCK_N * (D + 8) * 2    = 2 * 32 * 520 * 2 = 66560 B
    key validity     2 * BLOCK_N * 4              = 256 B
                                                     total 100096 B
*/
#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>

#include "mma.cuh"

namespace sm89 {

template<int BLOCK_H, int BLOCK_N, int NUM_WARPS, int D>
struct FlashAttnDims {
    static_assert(BLOCK_H % 16 == 0, "BLOCK_H must be a multiple of the 16-row mma tile");
    static_assert(NUM_WARPS % (BLOCK_H / 16) == 0, "one d-slice group per m-tile");
    static_assert(D % 16 == 0 && BLOCK_N % 16 == 0, "head dim and key tile must be mma-aligned");

    // 8 padding elements per row make the B-operand access bank-conflict free: with a row stride
    // of 260 bf16 (= 520 B), the 32 lanes of an ldmatrix always land on 32 distinct banks.
    static constexpr int PAD = 8;
    static constexpr int ROW = D + PAD;
    static constexpr int M_TILES = BLOCK_H / 16;
    static constexpr int D_SPLIT = NUM_WARPS / M_TILES;
    static constexpr int D_PER_WARP = D / D_SPLIT;
    static constexpr int D_TILES = D_PER_WARP / 8;
    static constexpr int N_TILES = BLOCK_N / 8;
    static constexpr int K_STEPS = D / 16;
    static constexpr int KV_CHUNKS = BLOCK_N / 16;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
};

template<int BLOCK_H, int BLOCK_N, int NUM_WARPS, int D, int NUM_STAGES>
struct FlashAttnCore {
    using Dim = FlashAttnDims<BLOCK_H, BLOCK_N, NUM_WARPS, D>;

    static constexpr int M_TILES = Dim::M_TILES;
    static constexpr int D_TILES = Dim::D_TILES;
    static constexpr int N_TILES = Dim::N_TILES;
    static constexpr int KV_CHUNKS = Dim::KV_CHUNKS;
    static constexpr int ROW = Dim::ROW;
    static constexpr int NUM_THREADS = Dim::NUM_THREADS;

    __nv_bfloat16 *q_smem;     // [BLOCK_H][ROW]
    __nv_bfloat16 *k_smem;     // [NUM_STAGES][BLOCK_N][ROW], doubles as the V tile
    int *valid_smem;           // [NUM_STAGES][BLOCK_N], 0 = masked-out key

    float sm_scale_div_log2;

    int m_tile, d_tile;
    int warp_id, lane_id, tid;

    // Per-warp state. `o[j]` is the 4-accumulator result of one `mma.m16n8k16` on output dims
    // [8j, 8j + 8); the accumulator layout gives lane l the rows l/4 and l/4 + 8 of the tile.
    float o[D_TILES][4];
    float row_max[2];
    float row_sum[2];
    // Optional per-row multiplier on the raw QK scores, applied in fp32 alongside sm_scale. Used by
    // the fused kernel to fold in an RMS norm over Q: scaling the bf16 Q tile instead would round
    // the normalised values back to bf16 and lose the precision. Both default to 1.0f, which makes
    // the multiply below exact.
    float row_scale[2];

    __device__ __forceinline__ void reset() {
        #pragma unroll
        for (int j = 0; j < D_TILES; ++j) {
            o[j][0] = 0.0f; o[j][1] = 0.0f; o[j][2] = 0.0f; o[j][3] = 0.0f;
        }
        row_max[0] = -CUDART_INF_F; row_max[1] = -CUDART_INF_F;
        row_sum[0] = 0.0f;         row_sum[1] = 0.0f;
        row_scale[0] = 1.0f;       row_scale[1] = 1.0f;
    }

    __device__ __forceinline__ void init(int warp_id_, int tid_, float scale) {
        sm_scale_div_log2 = scale;
        warp_id = warp_id_;
        tid = tid_;
        lane_id = tid_ % 32;
        m_tile = warp_id_ % M_TILES;
        d_tile = warp_id_ / M_TILES;
        reset();
    }

    __device__ __forceinline__ __nv_bfloat16 *k_stage(int stage) const {
        return k_smem + stage * BLOCK_N * ROW;
    }
    __device__ __forceinline__ int *valid_stage(int stage) const {
        return valid_smem + stage * BLOCK_N;
    }

    // Address for a 4-matrix `ldmatrix`: matrices 0/1 are the two halves of the row range
    // [row0, row0 + 16) at columns [col0, col0 + 8), matrices 2/3 the same rows at columns
    // [col0 + 8, col0 + 16). That is exactly the pair layout of the m16n8k16 A and B fragments.
    __device__ __forceinline__ uint32_t ldmatrix_addr(const __nv_bfloat16 *base, int row0, int col0) const {
        int g = lane_id >> 3;        // which of the four 8x8 matrices this thread addresses
        int r = lane_id & 7;         // its row inside that matrix
        int row = row0 + (g & 1) * 8 + r;
        int col = col0 + (g >> 1) * 8;
        return mma::smem_addr(base) + static_cast<uint32_t>((row * ROW + col) * sizeof(__nv_bfloat16));
    }

    __device__ __forceinline__ void load_q_frag(uint32_t (&a)[4], int k_step) const {
        uint32_t addr = ldmatrix_addr(q_smem + m_tile * 16 * ROW, 0, k_step * 16);
        mma::ldmatrix_x4(a, addr);
    }

    // `load_k_tile(stage, key_tile_idx)` must fill the K/V rows and the validity flags of `stage`.
    template<typename Loader>
    __device__ __forceinline__ void run(int num_key_tiles, Loader &&load_k_tile) {
        if (num_key_tiles <= 0) {
            // The caller prefetched Q with cp.async and reuses the Q tile as the output staging
            // buffer in its epilogue. `cp.async.wait_all` only covers the calling thread's own
            // copies, so the barrier is what stops a fast thread from overwriting a row that
            // another thread is still filling.
            mma::cp_async_wait_all();
            __syncthreads();
            return;
        }

        load_k_tile(0, 0);
        mma::cp_async_commit();

        for (int tile = 0; tile < num_key_tiles; ++tile) {
            const int cur = tile % NUM_STAGES;
            if (tile + 1 < num_key_tiles) {
                load_k_tile((tile + 1) % NUM_STAGES, tile + 1);
                mma::cp_async_commit();
                mma::cp_async_wait<NUM_STAGES - 1>();
            } else {
                // Nothing left to prefetch: `cp.async.wait_group N` returns as soon as *at most*
                // N groups are in flight, so the tail needs an explicit wait-all.
                mma::cp_async_wait_all();
            }
            __syncthreads();

            accumulate_tile(cur);

            // Every warp must be done reading the tile before its stage slot is recycled, which
            // happens NUM_STAGES iterations later.
            __syncthreads();
        }
    }

    __device__ __forceinline__ void accumulate_tile(int stage) {
        const __nv_bfloat16 *ks = k_stage(stage);
        const int *valid = valid_stage(stage);

        float acc[N_TILES][4];
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            acc[n][0] = 0.0f; acc[n][1] = 0.0f; acc[n][2] = 0.0f; acc[n][3] = 0.0f;
        }

        // ---- S = (Q @ K^T) * log2(e) ----
        #pragma unroll
        for (int k_step = 0; k_step < Dim::K_STEPS; ++k_step) {
            uint32_t a[4];
            load_q_frag(a, k_step);
            #pragma unroll
            for (int n_pair = 0; n_pair < N_TILES / 2; ++n_pair) {
                uint32_t b[4];
                mma::ldmatrix_x4(b, ldmatrix_addr(ks, n_pair * 16, k_step * 16));
                // b[0], b[2] are the B fragment of key tile 2*n_pair; b[1], b[3] of 2*n_pair + 1
                uint32_t b0[2] = {b[0], b[2]};
                uint32_t b1[2] = {b[1], b[3]};
                mma::mma_m16n8k16_bf16(acc[n_pair * 2 + 0], a, b0, acc[n_pair * 2 + 0]);
                mma::mma_m16n8k16_bf16(acc[n_pair * 2 + 1], a, b1, acc[n_pair * 2 + 1]);
            }
        }

        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            acc[n][0] *= sm_scale_div_log2 * row_scale[0];
            acc[n][1] *= sm_scale_div_log2 * row_scale[0];
            acc[n][2] *= sm_scale_div_log2 * row_scale[1];
            acc[n][3] *= sm_scale_div_log2 * row_scale[1];
        }

        // ---- online softmax ----
        // A lane holds, for rows lane/4 and lane/4 + 8, key columns 8n + 2*(lane % 4) + {0, 1}
        // of every key tile n, i.e. all BLOCK_N keys of its two rows.
        bool v0[N_TILES], v1[N_TILES];
        float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            const int key0 = n * 8 + (lane_id & 3) * 2;
            v0[n] = valid[key0] != 0;
            v1[n] = valid[key0 + 1] != 0;
            acc[n][0] = v0[n] ? acc[n][0] : -CUDART_INF_F;
            acc[n][1] = v1[n] ? acc[n][1] : -CUDART_INF_F;
            acc[n][2] = v0[n] ? acc[n][2] : -CUDART_INF_F;
            acc[n][3] = v1[n] ? acc[n][3] : -CUDART_INF_F;
            tile_max[0] = fmaxf(tile_max[0], fmaxf(acc[n][0], acc[n][1]));
            tile_max[1] = fmaxf(tile_max[1], fmaxf(acc[n][2], acc[n][3]));
        }
        // butterfly-reduce the row max over the four lanes that hold the same row
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 1));
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 2));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 1));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 2));

        float new_max[2], corr[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            new_max[r] = fmaxf(row_max[r], tile_max[r]);
            corr[r] = (new_max[r] == -CUDART_INF_F) ? 0.0f : exp2f(row_max[r] - new_max[r]);
        }

        float sum_p[2] = {0.0f, 0.0f};
        uint32_t p_frag[KV_CHUNKS][4];
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            float p0 = (acc[n][0] == -CUDART_INF_F) ? 0.0f : exp2f(acc[n][0] - new_max[0]);
            float p1 = (acc[n][1] == -CUDART_INF_F) ? 0.0f : exp2f(acc[n][1] - new_max[0]);
            float p2 = (acc[n][2] == -CUDART_INF_F) ? 0.0f : exp2f(acc[n][2] - new_max[1]);
            float p3 = (acc[n][3] == -CUDART_INF_F) ? 0.0f : exp2f(acc[n][3] - new_max[1]);
            sum_p[0] += p0 + p1;
            sum_p[1] += p2 + p3;
            // The score accumulator layout is exactly the A-operand layout of the PV mma, so P
            // is packed straight out of the registers with no data movement.
            p_frag[n / 2][(n % 2) * 2 + 0] = mma::pack_bf16x2(p0, p1);
            p_frag[n / 2][(n % 2) * 2 + 1] = mma::pack_bf16x2(p2, p3);
        }
        sum_p[0] += __shfl_xor_sync(0xffffffff, sum_p[0], 1);
        sum_p[1] += __shfl_xor_sync(0xffffffff, sum_p[1], 1);
        sum_p[0] += __shfl_xor_sync(0xffffffff, sum_p[0], 2);
        sum_p[1] += __shfl_xor_sync(0xffffffff, sum_p[1], 2);

        row_max[0] = new_max[0];
        row_max[1] = new_max[1];
        row_sum[0] = row_sum[0] * corr[0] + sum_p[0];
        row_sum[1] = row_sum[1] * corr[1] + sum_p[1];

        // rescale the running output accumulator before accumulating this tile
        #pragma unroll
        for (int j = 0; j < D_TILES; ++j) {
            o[j][0] *= corr[0]; o[j][1] *= corr[0];
            o[j][2] *= corr[1]; o[j][3] *= corr[1];
        }

        // ---- O += P @ V (V is the very same K tile; ldmatrix.trans reads it in place) ----
        const int d_base = d_tile * Dim::D_PER_WARP;
        #pragma unroll
        for (int kc = 0; kc < KV_CHUNKS; ++kc) {
            #pragma unroll
            for (int j_pair = 0; j_pair < D_TILES / 2; ++j_pair) {
                uint32_t b[4];
                mma::ldmatrix_x4_trans(b, ldmatrix_addr(ks, kc * 16, d_base + j_pair * 16));
                // NOTE ldmatrix.trans pairs (r0, r1) and (r2, r3) into the B fragments (it is
                // ldmatrix - and only ldmatrix - that pairs (r0, r2) / (r1, r3)); see mma.cuh.
                uint32_t b0[2] = {b[0], b[1]};
                uint32_t b1[2] = {b[2], b[3]};
                mma::mma_m16n8k16_bf16(o[j_pair * 2 + 0], p_frag[kc], b0, o[j_pair * 2 + 0]);
                mma::mma_m16n8k16_bf16(o[j_pair * 2 + 1], p_frag[kc], b1, o[j_pair * 2 + 1]);
            }
        }
    }
};

}  // namespace sm89