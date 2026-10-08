// §7.1 microbenchmark, paged side: pagedAttentionKernel walking a block table, one page per
// (slot, layer, logical block). Same bytes of cache and same thread mapping as the contiguous
// side; the only difference is the indirection, which is the cost the paper measures here.

#include "harness.cuh"

#include "config.h"
#include "kernels.cuh"

int g_num_blocks = 0;

int main()
{
    for (int batch : kBatchSizes)
    {
        for (int context_len : kContextLens)
        {
            const int pages_per_layer = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int total_pages = batch * N_LAYERS * pages_per_layer;
            g_num_blocks = total_pages;
            const size_t cache_bytes = (size_t)total_pages * BLOCK_BYTES;

            __nv_bfloat16 *d_cache = nullptr;
            CHECK(cudaMalloc(&d_cache, cache_bytes));
            std::vector<__nv_bfloat16> host_cache(cache_bytes / sizeof(__nv_bfloat16));

            // Map pages the way the engine does: free_blocks is initialised 0..N-1 and
            // allocation pops from the back, so a fresh pool hands them out in descending
            // order. Using the engine's own order keeps this from measuring a layout the
            // engine never produces.
            std::vector<int> block_table((size_t)batch * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);
            int next = total_pages - 1;
            for (int slot = 0; slot < batch; ++slot)
            {
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    for (int p = 0; p < pages_per_layer; ++p)
                    {
                        block_table[(size_t)slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                                    (size_t)layer * MAX_BLOCKS_PER_SEQ + p] = next--;
                    }
                }
            }
            // now that the mapping is known, write the same logical values the contiguous
            // build writes, so the two outputs are comparable
            for (int slot = 0; slot < batch; ++slot)
            {
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    for (int token = 0; token < context_len; ++token)
                    {
                        const int page = block_table[(size_t)slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                                                     (size_t)layer * MAX_BLOCKS_PER_SEQ +
                                                     token / BLOCK_SIZE];
                        const size_t k_off = (size_t)page * (BLOCK_BYTES / sizeof(__nv_bfloat16)) +
                                             (size_t)(token % BLOCK_SIZE) * KV_DIM;
                        const size_t v_off = k_off + V_OFFSET / sizeof(__nv_bfloat16);
                        for (int d = 0; d < KV_DIM; ++d)
                        {
                            host_cache[k_off + d] = (__nv_bfloat16)kvValue(slot, layer, 0, token, d);
                            host_cache[v_off + d] = (__nv_bfloat16)kvValue(slot, layer, 1, token, d);
                        }
                    }
                }
            }
            CHECK(cudaMemcpy(d_cache, host_cache.data(), cache_bytes, cudaMemcpyHostToDevice));

            int *d_block_table;
            CHECK(cudaMalloc(&d_block_table, block_table.size() * sizeof(int)));
            CHECK(cudaMemcpy(d_block_table, block_table.data(), block_table.size() * sizeof(int),
                             cudaMemcpyHostToDevice));

            std::vector<int> seq_lens(batch, context_len), active(batch);
            for (int i = 0; i < batch; ++i)
            {
                active[i] = i;
            }
            int *d_seq_lens, *d_active;
            CHECK(cudaMalloc(&d_seq_lens, batch * sizeof(int)));
            CHECK(cudaMalloc(&d_active, batch * sizeof(int)));
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
                pagedAttentionKernel<<<grid, HEAD_DIM>>>(0, batch, d_q, d_cache, d_block_table,
                                                         d_seq_lens, d_active, d_out);
            }
            CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            CHECK(cudaEventRecord(start));
            for (int i = 0; i < kIters; ++i)
            {
                pagedAttentionKernel<<<grid, HEAD_DIM>>>(0, batch, d_q, d_cache, d_block_table,
                                                         d_seq_lens, d_active, d_out);
            }
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            std::vector<__nv_bfloat16> host_out((size_t)batch * EMBEDDING_LENGTH);
            CHECK(cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
            report("paged", context_len, batch, ms * 1000.0f / kIters, outputDigest(host_out));

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_cache); cudaFree(d_block_table); cudaFree(d_seq_lens);
            cudaFree(d_active); cudaFree(d_q); cudaFree(d_out);
        }
    }
    return 0;
}
