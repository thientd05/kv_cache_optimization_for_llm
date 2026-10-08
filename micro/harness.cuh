#pragma once

// Shared scaffolding for the two attention-kernel microbenchmarks (§7.1 of the paper).
// The two binaries each link the kernels of their OWN source tree, so neither copies a kernel
// and the two cannot drift apart; separate binaries also keep the identically-named symbols
// from colliding.

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// The grid from Fig 18a: "Latency of attention kernels" at batch sizes 8 and 32 over context
// lengths 64, 128 and 256.
static const int kContextLens[] = {64, 128, 256};
static const int kBatchSizes[] = {8, 32};
static const int kWarmup = 20;
static const int kIters = 200;

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t _e = (call);                                                      \
        if (_e != cudaSuccess) {                                                      \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

// Deterministic filler so both builds see identical numbers; the kernel is memory-bound and
// the values do not affect timing, but identical input removes one thing to wonder about.
inline std::vector<__nv_bfloat16> fillPattern(size_t n)
{
    std::vector<__nv_bfloat16> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = (__nv_bfloat16)(((float)(i % 97) - 48.0f) / 64.0f);
    }
    return v;
}

inline void report(const char *mechanism, int context_len, int batch, float latency_us,
                   unsigned long long digest)
{
    printf("{\"mechanism\":\"%s\",\"context_len\":%d,\"batch\":%d,\"latency_us\":%.3f,"
           "\"output_digest\":\"%llx\"}\n",
           mechanism, context_len, batch, latency_us, digest);
    fflush(stdout);
}

// Logical-coordinate fill. The two builds lay the same logical (slot, layer, K/V, token, dim)
// element out at completely different physical offsets, so filling by linear index would give
// them different *values* and their outputs could not be compared. Filling by logical
// coordinate instead means both kernels see identical inputs, so their output checksums must
// agree - which is what proves the paged kernel is doing the same work and not less of it.
__host__ __device__ inline float kvValue(int slot, int layer, int is_v, int token, int dim)
{
    int h = slot * 31 + layer * 17 + is_v * 7 + token * 3 + dim;
    return ((float)(((h % 97) + 97) % 97) - 48.0f) / 64.0f;
}

inline float qValue(int row, int dim)
{
    int h = row * 13 + dim * 5;
    return ((float)(((h % 89) + 89) % 89) - 44.0f) / 64.0f;
}

// Order-insensitive but value-sensitive digest of the attention output.
inline unsigned long long outputDigest(const std::vector<__nv_bfloat16> &out)
{
    unsigned long long d = 1469598103934665603ULL;
    for (size_t i = 0; i < out.size(); ++i)
    {
        // quantise: the two builds accumulate in the same order, but round to be robust to
        // the last bit rather than to a genuine difference in what was summed
        long long q = (long long)llrintf((float)out[i] * 4096.0f);
        d = (d ^ (unsigned long long)q) * 1099511628211ULL;
    }
    return d;
}
