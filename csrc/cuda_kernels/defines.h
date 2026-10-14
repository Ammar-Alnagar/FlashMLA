#pragma once

#include "cuda_kernels/defines.h"
#include <cutlass/arch/barrier.h>

#include "cuda_kernels/common/types.h"

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;
using fp8_e4m3 = cutlass::float_e4m3_t;