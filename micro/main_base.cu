// §7.1 microbenchmark, contiguous side: contiguousAttentionKernel reading one unbroken run of
// token slots per (slot, layer) with a fixed KV_DIM stride.

#include "harness.cuh"

#include "config.h"
#include "kernels.cuh"

// config.h declares these; the engine defines them. The kernels do not read them, but the
// linker wants them if anything in the translation units refers to them.
int g_kv_pool_tokens = 0;
ReservePolicy g_reserve_policy = ReservePolicy::Max;
const char *reservePolicyName() { return "max"; }

int main()
{
    for (int batch : kBatchSizes)
    {
        for (int context_len : kContextLens)
        {
            // One run per (slot, layer), cap == the context it holds. Same bytes as the paged
            // side allocates for the same batch and context.
            const int cap = context_len;
            const size_t cache_elems = (size_t)batch * cap * N_LAYERS * 2 * KV_DIM;

            __nv_bfloat16 *d_cache = nullptr;
            CHECK(cudaMalloc(&d_cache, cache_elems * sizeof(__nv_bfloat16)));
            std::vector<__nv_bfloat16> host_cache(cache_elems);
            for (int slot = 0; slot < batch; ++slot)
            {
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    const size_t k_off = kvKOffset(slot * cap, cap, layer);
                    for (int token = 0; token < context_len; ++token)
                    {
                        for (int d = 0; d < KV_DIM; ++d)
                        {
                            host_cache[k_off + (size_t)token * KV_DIM + d] =
                                (__nv_bfloat16)kvValue(slot, layer, 0, token, d);
                            host_cache[k_off + (size_t)cap * KV_DIM + (size_t)token * KV_DIM + d] =
                                (__nv_bfloat16)kvValue(slot, layer, 1, token, d);
                        }
                    }
                }
            }
            CHECK(cudaMemcpy(d_cache, host_cache.data(), cache_elems * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice));

            std::vector<int> slot_base(batch), slot_cap(batch, cap), seq_lens(batch, context_len),
                active(batch);
            for (int i = 0; i < batch; ++i)
            {
                slot_base[i] = i * cap;
                active[i] = i;
            }
            int *d_slot_base, *d_slot_cap, *d_seq_lens, *d_active;
            CHECK(cudaMalloc(&d_slot_base, batch * sizeof(int)));
            CHECK(cudaMalloc(&d_slot_cap, batch * sizeof(int)));
            CHECK(cudaMalloc(&d_seq_lens, batch * sizeof(int)));
            CHECK(cudaMalloc(&d_active, batch * sizeof(int)));
            CHECK(cudaMemcpy(d_slot_base, slot_base.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_slot_cap, slot_cap.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_seq_lens, seq_lens.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_active, active.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

            __nv_bfloat16 *d_q, *d_out;
            CHECK(cudaMalloc(&d_q, (size_t)batch * EMBEDDING_LENGTH * sizeof(__nv_bfloat16)));
            CHECK(cudaMalloc(&d_out, (size_t)batch * EMBEDDING_LENGTH * sizeof(__nv_bfloat16)));
            std::vector<__nv_bfloat16> host_q((size_t)batch * EMBEDDING_LENGTH);
            for (int r = 0; r < batch; ++r)
                for (int d = 0; d < EMBEDDING_LENGTH; ++d)
                    host_q[(size_t)r * EMBEDDING_LENGTH + d] = (__nv_bfloat16)qValue(r, d);
            CHECK(cudaMemcpy(d_q, host_q.data(), host_q.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice));

            const dim3 grid(batch, NUM_Q_HEADS);
            for (int i = 0; i < kWarmup; ++i)
            {
                contiguousAttentionKernel<<<grid, HEAD_DIM>>>(0, batch, d_q, d_cache, d_slot_base,
                                                              d_slot_cap, d_seq_lens, d_active, d_out);
            }
            CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            CHECK(cudaEventRecord(start));
            for (int i = 0; i < kIters; ++i)
            {
                contiguousAttentionKernel<<<grid, HEAD_DIM>>>(0, batch, d_q, d_cache, d_slot_base,
                                                              d_slot_cap, d_seq_lens, d_active, d_out);
            }
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            std::vector<__nv_bfloat16> host_out((size_t)batch * EMBEDDING_LENGTH);
            CHECK(cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
            report("contiguous", context_len, batch, ms * 1000.0f / kIters, outputDigest(host_out));

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_cache); cudaFree(d_slot_base); cudaFree(d_slot_cap);
            cudaFree(d_seq_lens); cudaFree(d_active); cudaFree(d_q); cudaFree(d_out);
        }
    }
    return 0;
}
