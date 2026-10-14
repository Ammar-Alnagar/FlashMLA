/*
Architecture-independent type aliases shared by every backend.

`cuda_kernels/defines.h` additionally pulls in the Blackwell cluster-barrier helpers and must only
be included by sm100 sources; everything here is usable from an sm89-only build.
*/
#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

using bf16 = cutlass::bfloat16_t;
using fp8_e4m3 = cutlass::float_e4m3_t;