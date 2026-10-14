#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace sm89::dequant {

__inline__ __device__ float e4m3_to_float(__nv_fp8_e4m3 val) {
    return __half2float(__half(val));
}

__inline__ __device__ float e8m0_to_scale(uint8_t byte_val) {
    int exp_val = static_cast<int>(byte_val) - 127;
    return exp2f(static_cast<float>(exp_val));
}

__inline__ __device__ float fp4_e2m1_to_float(uint8_t nibble) {
    uint8_t sign = (nibble >> 3) & 1;
    uint8_t exp = (nibble >> 1) & 3;
    uint8_t mant = nibble & 1;

    float val = 0.0f;
    if (exp == 0) {
        val = mant * 0.25f;
    } else if (exp == 1) {
        val = 1.0f + mant * 0.5f;
    } else if (exp == 2) {
        val = 2.0f + mant * 1.0f;
    } else { // exp == 3
        val = 4.0f + mant * 2.0f;
    }
    return sign ? -val : val;
}

// Dequantize 1 element from V4.1 FP8 token buffer (528 bytes per token)
// token_ptr points to the 528 bytes of this token. dim_idx in [0, 512).
__inline__ __device__ float dequant_v41_fp8(const uint8_t* token_ptr, int dim_idx) {
    const __nv_fp8_e4m3* raw_data = reinterpret_cast<const __nv_fp8_e4m3*>(token_ptr);
    const uint8_t* scales = token_ptr + 512;

    int scale_idx = dim_idx / 32;
    float raw_val = e4m3_to_float(raw_data[dim_idx]);
    float scale = e8m0_to_scale(scales[scale_idx]);
    return raw_val * scale;
}

// Dequantize 1 element from V4.1 FP4 token buffer (288 bytes per token)
// token_ptr points to the 288 bytes of this token. dim_idx in [0, 512).
__inline__ __device__ float dequant_v41_fp4(const uint8_t* token_ptr, int dim_idx) {
    const uint8_t* raw_bytes = token_ptr;
    const __nv_fp8_e4m3* scales = reinterpret_cast<const __nv_fp8_e4m3*>(token_ptr + 256);

    int byte_idx = dim_idx / 2;
    uint8_t byte_val = raw_bytes[byte_idx];
    uint8_t nibble = (dim_idx % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);

    int scale_idx = dim_idx / 16;
    float raw_val = fp4_e2m1_to_float(nibble);
    float scale = e4m3_to_float(scales[scale_idx]);
    return raw_val * scale;
}

}  // namespace sm89::dequant
