/*
SM89 (Ada Lovelace) tensor-core helpers.

SM89 has no tcgen05 / TMA / cluster support, so the Blackwell helpers in kerutils are not
available here. What Ada does have is the Ampere-style warp-level `mma.sync` plus `ldmatrix`,
which is everything the SM89 kernels below need:

    Q @ K^T   and   P @ V      with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

Both operands are read out of shared memory with `ldmatrix` in the layout the `mma` fragment
layout expects, i.e. *no* separate shared-memory staging or transposing of the K/V tiles:

  - The A fragment of `m16n8k16` wants `A[row = lane/4 (+8)][col = (lane%4)*2 + {0,1} (+8)]`,
    which is exactly what a plain `ldmatrix.x4` of a row-major 8x8 bf16 tile produces. It is
    used for Q and for P (P is produced in the mma accumulator layout, so it needs no shuffle at
    all - see `flash_attn.cuh`).
  - The B fragment of `m16n8k16` wants `B[k = (lane%4)*2 + {0,1} (+8)][n = lane/4]`. For
    `S = Q @ K^T` (k = head dim, n = key) that is `Ks[key][dim]`, i.e. the natural row-major K
    tile, so a plain `ldmatrix.x4` works. For `O = P @ V` (k = key, n = head dim) it is the
    transpose, which `ldmatrix.x4.trans` produces directly from the very same `Ks[key][dim]`
    tile - this is why V never needs its own shared-memory buffer (V is K for MLA).
*/
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

#include "cuda_kernels/common/types.h"

namespace sm89::mma {

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// d[16x8] += a[16x16] @ b[16x8], with a row-major and b column-major (k-contiguous)
__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// Load four 8x8 bf16 tiles of shared memory. Thread t supplies the address of row t%8 of
// matrix t/8. `r[i]` holds the i-th 8x8 matrix in the mma A/B fragment layout of one thread.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr)
    );
}

// Same, but each 8x8 tile is transposed on the fly. Note the difference against the plain variant,
// which is the whole reason both helpers exist:
//
//   plain ldmatrix : r0 = tile[l/4][2*(l%4) + {0,1}],  r1 = tile[8 + l/4][...same cols...]
//   ldmatrix.trans : r0 = tile[2*(l%4) + {0,1}][l/4],  r1 = tile[8 + 2*(l%4) + {0,1}][l/4]
//
// i.e. with the address pattern of `ldmatrix_x4`'s companion helpers below (matrices 0/1 span rows
// [row0, row0+16) at column col0, matrices 2/3 the same rows at column col0+8), the plain form
// pairs (r0, r2) and (r1, r3) into the two m16n8k16 B fragments, while the transposed form pairs
// (r0, r1) and (r2, r3).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr)
    );
}

// 16-byte asynchronous global -> shared copy. `src_bytes` < 16 zero-fills the remainder, which
// is how invalid (masked-out) K/V rows are neutralised without a separate pass.
__device__ __forceinline__ void cp_async_16(void *dst_smem, const void *src_gmem, int src_bytes) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(smem_addr(dst_smem)), "l"(src_gmem), "r"(src_bytes)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

// Wait until at most `N` cp.async groups are still in flight.
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// 128-bit shared-memory load / store helpers (the addresses are already 16-byte aligned by
// construction, see the smem layouts in flash_attn.cuh).
__device__ __forceinline__ uint4 ld_shared_u128(const void *p) {
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_addr(p)));
    return v;
}

__device__ __forceinline__ void st_shared_u128(void *p, const uint4 &v) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_addr(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void st_shared_u32(void *p, uint32_t v) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr(p)), "r"(v));
}

// Pack two bf16 into a 32-bit word (the mma A/B fragment element type).
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    uint32_t r;
    __builtin_memcpy(&r, &v, sizeof(r));
    return r;
}

// `cutlass::bfloat16_t` (the kernel-side `bf16`) and `__nv_bfloat16` (what the shared-memory
// tiles and mma fragments use) are layout compatible; go through memcpy to avoid relying on
// conversions between the two vendor types.
__device__ __forceinline__ __nv_bfloat16 load_bf16(const bf16 *p) {
    __nv_bfloat16 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

__device__ __forceinline__ void store_bf16(bf16 *p, __nv_bfloat16 v) {
    __builtin_memcpy(p, &v, sizeof(v));
}

}  // namespace sm89::mma