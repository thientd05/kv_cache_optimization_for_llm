#include "engine.cuh"

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include "kernels.cuh"
#include "utils.cuh"

using json = nlohmann::json;

SlotState::SlotState()
    : is_slot_free(BATCH_SIZE, true),
      generated_tokens(BATCH_SIZE),
      last_generated_tokens(BATCH_SIZE),
      current_prompt_len(BATCH_SIZE, 0),
      slot_request_id(BATCH_SIZE, -1),
      decode_start(BATCH_SIZE)
{
}

DeviceBuffers allocateDeviceBuffers()
{
    // Everything per-token is sized for one packed prefill pass (MAX_BUFFER_SIZE ==
    // MAX_BATCH_TOKENS), which also covers decode, where only num_active_slots rows are used.
    DeviceBuffers buf{};

    buf.input_tokens = (int *)allocDevice(MAX_BATCH_TOKENS * sizeof(int), "input tokens");

    // index arrays that describe the packed prefill batch, one entry per packed token
    buf.token_positions = (int *)allocDevice(MAX_BATCH_TOKENS * sizeof(int), "token positions");
    buf.token_seq_start = (int *)allocDevice(MAX_BATCH_TOKENS * sizeof(int), "token seq starts");
    buf.token_slot_ids = (int *)allocDevice(MAX_BATCH_TOKENS * sizeof(int), "token slot ids");
    buf.last_token_rows = (int *)allocDevice(BATCH_SIZE * sizeof(int), "last token rows");
    // worst case one partial tile per prompt on top of the full ones
    buf.tile_token_begin = (int *)allocDevice(MAX_PREFILL_ATTN_TILES * sizeof(int), "attn tile begins");
    buf.tile_token_count = (int *)allocDevice(MAX_PREFILL_ATTN_TILES * sizeof(int), "attn tile counts");

    // embeddingGather writes straight in here now - the old input_embeddings buffer was
    // copied into hidden_state and then never read again
    buf.hidden_state = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, "hidden_state");
    buf.rms_norms = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, "rms_norms");
    buf.buf_2048_1 = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, "buf_2048_1");
    buf.buf_2048_2 = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, "buf_2048_2");

    // K and V of the packed batch, before they are scattered into the paged cache.
    // Prefill attention reads them straight from here instead of walking the block table.
    buf.k_proj_temp_buf = (__nv_bfloat16 *)allocDevice(MAX_BATCH_TOKENS * KV_DIM * sizeof(__nv_bfloat16), "k_proj_temp_buf");
    buf.v_proj_temp_buf = (__nv_bfloat16 *)allocDevice(MAX_BATCH_TOKENS * KV_DIM * sizeof(__nv_bfloat16), "v_proj_temp_buf");

    buf.gate = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM, "gate");
    buf.up = (__nv_bfloat16 *)allocDevice(MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM, "up");

    // Last token of every prompt in a prefill pass, gathered so the lm_head GEMM only sees
    // the rows whose logits are actually used.
    buf.last_hidden = (__nv_bfloat16 *)allocDevice(BATCH_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH, "last_hidden");

    // Logits, BATCH_SIZE rows. It used to be MAX_PROMPT_LEN rows (125.25 MiB) because
    // prefill ran the lm_head over every prompt token; at 16 rows it is 3.91 MiB, and that
    // is most of what pays for the packed per-token buffers. See MACHINE.md.
    buf.embed_proj = (__nv_bfloat16 *)allocDevice(sizeof(__nv_bfloat16) * BATCH_SIZE * VOCAB_SIZE, "embed_proj");

    buf.last_tokens = (int *)allocDevice(BATCH_SIZE * sizeof(int), "decode last tokens");
    buf.active_slots = (int *)allocDevice(BATCH_SIZE * sizeof(int), "decode active slots");
    buf.seq_lens = (int *)allocDevice(BATCH_SIZE * sizeof(int), "decode seq lens");

    // Sampling scratch. The partials are tiny (31 x 128 pairs) and the penalty mask is
    // BATCH_SIZE * VOCAB_SIZE bytes = 3.79 MiB, which is half of what the host-side logits
    // copy it replaces used to cost in pinned-free pageable memory anyway.
    buf.logit_row_slots = (int *)allocDevice(BATCH_SIZE * sizeof(int), "logit row slots");
    buf.argmax_values = (float *)allocDevice(BATCH_SIZE * ARGMAX_CHUNKS_PER_ROW * sizeof(float), "argmax values");
    buf.argmax_indices = (int *)allocDevice(BATCH_SIZE * ARGMAX_CHUNKS_PER_ROW * sizeof(int), "argmax indices");
    buf.sampled_tokens = (int *)allocDevice(BATCH_SIZE * sizeof(int), "sampled tokens");
    buf.penalty_mask = (unsigned char *)allocDevice((size_t)BATCH_SIZE * VOCAB_SIZE, "repetition penalty mask");
    cudaMemset(buf.penalty_mask, 0, (size_t)BATCH_SIZE * VOCAB_SIZE);

    return buf;
}

KVCacheState allocateKVCache()
{
    KVCacheState kv{};
    kv.cache = (__nv_bfloat16 *)allocDevice(KV_CACHE_SIZE_BYTES, "KV cache");
    kv.free_blocks.resize(NUM_BLOCKS);
    std::iota(kv.free_blocks.begin(), kv.free_blocks.end(), 0);
    kv.block_table.assign(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);
    kv.block_table_gpu = (int *)allocDevice(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), "block table");
    return kv;
}

void reportEngineConfig()
{
    // The scratch buffers scale with MAX_BATCH_TOKENS and this card has very little
    // slack, so say out loud what is left once everything is allocated.
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cerr << "Scratch allocated for MAX_BATCH_TOKENS=" << MAX_BATCH_TOKENS
              << " prefill tokens, VRAM left: " << free_mem / B_TO_MB << " MiB\n"
              << std::flush;

    // BATCH_SIZE is derived, so the client cannot hardcode it
    json config_j;
    config_j["type"] = "engine_config";
    config_j["batch_size"] = BATCH_SIZE;
    config_j["max_batch_tokens"] = MAX_BATCH_TOKENS;
    config_j["max_prompt_len"] = MAX_PROMPT_LEN;
    std::cout << config_j.dump() << "\n" << std::flush;
}

std::vector<PrefillBatchItem> admitQueuedRequests(std::deque<Request> &queue, SlotState &slots)
{
    std::vector<PrefillBatchItem> items;
    for (int slot = 0; slot < BATCH_SIZE; ++slot)
    {
        if (!slots.is_slot_free[slot])
        {
            continue;
        }
        Request request;
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            if (queue.empty())
            {
                break; // nothing left to admit, the remaining free slots stay free
            }
            request = std::move(queue.front());
            queue.pop_front();
        }
        if ((int)request.tokens.size() > MAX_PROMPT_LEN)
        {
            std::cerr << "Request " << request.id << ": prompt of " << request.tokens.size()
                      << " tokens exceeds MAX_PROMPT_LEN (" << MAX_PROMPT_LEN << "), truncating\n";
            request.tokens.resize(MAX_PROMPT_LEN);
        }
        slots.is_slot_free[slot] = false;
        slots.slot_request_id[slot] = request.id;
        slots.generated_tokens[slot].clear();
        // start the clock now, so a sequence that is retired during prefill (EOS on its
        // first token, or a failure) still reports a sane duration instead of whatever
        // the previous owner of this slot left behind
        slots.decode_start[slot] = std::chrono::high_resolution_clock::now();
        items.push_back({slot, request.id, std::move(request.tokens)});
    }
    return items;
}

void finishSequence(int slot, int request_id, const char *error, SlotState &slots, KVCacheState &kv)
{
    auto decode_end = std::chrono::high_resolution_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(decode_end - slots.decode_start[slot]).count();
    size_t tokens = slots.generated_tokens[slot].size();

    json done_j;
    done_j["id"] = request_id;
    done_j["slot"] = slot;
    done_j["done"] = true;
    done_j["tokens"] = tokens;
    done_j["decode_time"] = decode_ms;
    done_j["decode_speed"] = decode_ms > 0.0f ? tokens / (decode_ms / 1000.0f) : 0.0f;
    if (error != nullptr)
    {
        done_j["error"] = error;
    }
    std::cout << done_j.dump() << "\n" << std::flush;

    slots.is_slot_free[slot] = true;
    slots.slot_request_id[slot] = -1;
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        for (int logical_block_idx = 0; logical_block_idx < MAX_BLOCKS_PER_SEQ; ++logical_block_idx)
        {
            int block_idx = slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
            if (kv.block_table[block_idx] != -1)
            {
                kv.free_blocks.push_back(kv.block_table[block_idx]);
                kv.block_table[block_idx] = -1;
            }
        }
    }
    cudaMemcpy(kv.block_table_gpu, kv.block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);
}

void enforcePageBudget(SlotState &slots, KVCacheState &kv)
{
    for (int slot = 0; slot < BATCH_SIZE; ++slot)
    {
        if (slots.is_slot_free[slot])
        {
            continue;
        }
        if (slots.current_prompt_len[slot] / BLOCK_SIZE >= MAX_BLOCKS_PER_SEQ)
        {
            std::cerr << "Slot " << slot << " ran past the block table (" << MAX_SEQ_LEN
                      << " tokens), retiring it\n";
            finishSequence(slot, slots.slot_request_id[slot], "sequence_too_long", slots, kv);
        }
    }

    // a sequence that starts a new page this step needs one page per layer
    auto pages_needed_now = [&]()
    {
        int pages = 0;
        for (int slot = 0; slot < BATCH_SIZE; ++slot)
        {
            if (!slots.is_slot_free[slot] && slots.current_prompt_len[slot] % BLOCK_SIZE == 0)
            {
                pages += N_LAYERS;
            }
        }
        return pages;
    };
    // evict newest first: the older sequences are closer to finishing on their own
    // and giving their pages back
    for (int slot = BATCH_SIZE; slot-- > 0 && pages_needed_now() > (int)kv.free_blocks.size();)
    {
        if (slots.is_slot_free[slot] || slots.current_prompt_len[slot] % BLOCK_SIZE != 0)
        {
            continue;
        }
        std::cerr << "KV cache exhausted during decode, evicting slot " << slot << "\n";
        finishSequence(slot, slots.slot_request_id[slot], "kv_cache_exhausted", slots, kv);
    }
}

void prefillBatch(std::vector<PrefillBatchItem> &items,
                  DeviceBuffers &buf,
                  const Weights &weights,
                  cublasHandle_t cublas_handle,
                  SlotState &slots,
                  KVCacheState &kv,
                  std::deque<Request> &queue)
{
    // every GEMM in prefill is a plain C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A slot is recycled between sequences, so the repetition penalty mask of every slot
    // admitted here has to start empty - otherwise the new sequence inherits the previous
    // tenant's token history. One VOCAB_SIZE row is 125 KiB of device memset.
    for (const PrefillBatchItem &item : items)
    {
        cudaMemset(buf.penalty_mask + (size_t)item.slot * VOCAB_SIZE, 0, VOCAB_SIZE);
    }

    size_t next_item = 0;
    while (next_item < items.size())
    {
        // ---- pack as many prompts as fit into one pass ----
        // token t of the pass lives at row t of every [num_tokens, dim] buffer
        std::vector<int> packed_tokens;
        std::vector<int> token_positions;
        std::vector<int> token_seq_start;
        std::vector<int> token_slot_ids;
        std::vector<int> last_token_rows;
        std::vector<int> logit_row_slots;
        std::vector<int> tile_token_begin;
        std::vector<int> tile_token_count;
        std::vector<size_t> pass_items;

        while (next_item < items.size())
        {
            int prompt_len = (int)items[next_item].tokens.size();
            if (!packed_tokens.empty() && (int)packed_tokens.size() + prompt_len > MAX_BATCH_TOKENS)
            {
                break;
            }
            int seq_start = (int)packed_tokens.size();
            for (int position = 0; position < prompt_len; ++position)
            {
                packed_tokens.push_back(items[next_item].tokens[position]);
                token_positions.push_back(position);
                token_seq_start.push_back(seq_start);
                token_slot_ids.push_back(items[next_item].slot);
            }
            last_token_rows.push_back(seq_start + prompt_len - 1);
            // row i of the logits belongs to this prompt's slot; the sampling kernels need
            // it to find the slot's penalty mask row
            logit_row_slots.push_back(items[next_item].slot);
            // cut this prompt into query tiles for prefillAttention; a tile never straddles
            // a prompt boundary, which is what keeps attention from looking across prompts
            for (int tile_begin = 0; tile_begin < prompt_len; tile_begin += PREFILL_ATTN_QUERIES_PER_TILE)
            {
                tile_token_begin.push_back(seq_start + tile_begin);
                tile_token_count.push_back(std::min(PREFILL_ATTN_QUERIES_PER_TILE, prompt_len - tile_begin));
            }
            pass_items.push_back(next_item);
            ++next_item;
        }

        int num_tokens = (int)packed_tokens.size();
        int num_seqs = (int)pass_items.size();
        int num_attn_tiles = (int)tile_token_begin.size();

        // ---- reserve every KV page the pass needs, for all prompts and all layers at once ----
        // Doing it up front means the layer loop only has to launch the scatter kernel, and
        // kv.block_table_gpu is synchronised exactly once per pass instead of once per page.
        int pages_needed = 0;
        for (size_t i = 0; i < pass_items.size(); ++i)
        {
            int prompt_len = (int)items[pass_items[i]].tokens.size();
            pages_needed += N_LAYERS * ((prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
        }
        if ((int)kv.free_blocks.size() < pages_needed)
        {
            // Unreachable by construction: BATCH_SIZE is derived so that every slot the
            // scheduler can admit into has its worst-case page budget reserved
            // (BATCH_SIZE * PAGES_PER_SEQUENCE <= NUM_BLOCKS). If it happens anyway the
            // budget maths is wrong somewhere, and the one thing we must not do is drop the
            // prompt on the floor: the client is blocked waiting for it. Hand every
            // not-yet-prefilled prompt of this call back to the front of the queue so it is
            // retried once a running sequence returns its pages - unless nothing is running,
            // in which case no pages will ever come back and the request genuinely failed.
            std::cerr << "KV cache exhausted: need " << pages_needed << " pages, "
                      << kv.free_blocks.size() << " free. Deferring " << (items.size() - pass_items.front())
                      << " prompt(s).\n";

            // pass_items is contiguous, so everything from its first entry on is unprefilled
            const size_t first_unfilled = pass_items.front();
            for (size_t i = first_unfilled; i < items.size(); ++i)
            {
                slots.is_slot_free[items[i].slot] = true;
            }

            bool anything_running = false;
            for (size_t slot = 0; slot < slots.is_slot_free.size(); ++slot)
            {
                if (!slots.is_slot_free[slot])
                {
                    anything_running = true;
                    break;
                }
            }

            // reverse order, so push_front leaves the queue in its original order
            for (size_t i = items.size(); i-- > first_unfilled;)
            {
                if (anything_running)
                {
                    std::lock_guard<std::mutex> lock(g_queue_mutex);
                    queue.push_front({items[i].request_id, std::move(items[i].tokens)});
                }
                else
                {
                    finishSequence(items[i].slot, items[i].request_id, "kv_cache_exhausted", slots, kv);
                }
            }
            return;
        }

        for (size_t i = 0; i < pass_items.size(); ++i)
        {
            int slot = items[pass_items[i]].slot;
            int prompt_len = (int)items[pass_items[i]].tokens.size();
            int num_pages = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
                for (int logical_block_idx = 0; logical_block_idx < num_pages; ++logical_block_idx)
                {
                    int entry = slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
                    // slots are released (and their pages returned) when a sequence finishes,
                    // so everything we touch here must still be unmapped
                    assert(kv.block_table[entry] == -1 && "page must be free before prefill - what happened?");
                    kv.block_table[entry] = kv.free_blocks.back();
                    kv.free_blocks.pop_back();
                }
            }
        }
        cudaMemcpy(kv.block_table_gpu, kv.block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

        auto prefill_start = std::chrono::high_resolution_clock::now();

        cudaMemcpy(buf.input_tokens, packed_tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.token_positions, token_positions.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.token_seq_start, token_seq_start.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.token_slot_ids, token_slot_ids.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.last_token_rows, last_token_rows.data(), num_seqs * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.logit_row_slots, logit_row_slots.data(), num_seqs * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.tile_token_begin, tile_token_begin.data(), num_attn_tiles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(buf.tile_token_count, tile_token_count.data(), num_attn_tiles * sizeof(int), cudaMemcpyHostToDevice);

        // the gather can write straight into hidden_state: the old separate input_embeddings
        // buffer was copied into hidden_state and never read again
        embeddingGatherKernel<<<num_tokens, EMBED_HALF>>>(buf.input_tokens, buf.hidden_state, weights.embed_tokens, num_tokens);

        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNormKernel<<<num_tokens, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.input_layernorm[layer], num_tokens);

            // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
            // it perceives my matrices as transposed
            // there's a trick where C = A * B == C^T = B^T * A^T
            // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
            // so I need to do: Q^T = wq ^T * inputs
            // the beauty is that we don't need to transpose Q^T back to Q
            // because cublas sees the output as column-major
            // so it's in fact transposed
            // final dim (num_tokens, EMBEDDING_LENGTH), num_tokens now being the whole batch
            __nv_bfloat16 *q_proj = buf.buf_2048_1;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_tokens,       // n
                         EMBEDDING_LENGTH, // k
                         &alpha,
                         weights.w_q[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda
                         buf.rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb
                         &beta,
                         q_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldc
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
            // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tokens, KV_DIM)
            // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_tokens,
                         EMBEDDING_LENGTH,
                         &alpha,
                         weights.w_k[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf.rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &beta,
                         buf.k_proj_temp_buf,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // same as K projection
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_tokens,
                         EMBEDDING_LENGTH,
                         &alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf.rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &beta,
                         buf.v_proj_temp_buf,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // RoPE now - positions restart at 0 for every prompt in the pack, so the angle
            // has to come from token_positions rather than from the row index
            ropePackedKernel<<<num_tokens, EMBEDDING_LENGTH / 2>>>(q_proj, num_tokens, EMBEDDING_LENGTH,
                                                                   buf.token_positions, d_cos_table, d_sin_table);
            ropePackedKernel<<<num_tokens, KV_DIM / 2>>>(buf.k_proj_temp_buf, num_tokens, KV_DIM,
                                                         buf.token_positions, d_cos_table, d_sin_table);

            // PagedAttention - scatter K and V of the entire batch into their pages
            // KV_DIM is 512, so one thread per K element and per V element fits in a block
            scatterKVPackedKernel<<<num_tokens, KV_DIM>>>(layer, num_tokens, buf.k_proj_temp_buf, buf.v_proj_temp_buf,
                                                          kv.cache, kv.block_table_gpu, buf.token_slot_ids, buf.token_positions);

            // Attention, in place over q_proj. Reads the packed K/V we just computed rather
            // than the pages, which is the same data and needs no block table walk.
            // Causality and prompt boundaries both fall out of token_seq_start/token_positions:
            // token t only ever looks at packed rows seq_start .. seq_start + position.
            prefillAttentionKernel<<<dim3(num_attn_tiles, NUM_Q_HEADS), PREFILL_ATTN_THREADS>>>(
                q_proj, buf.k_proj_temp_buf, buf.v_proj_temp_buf,
                buf.tile_token_begin, buf.tile_token_count,
                buf.token_seq_start, buf.token_positions, q_proj);

            // output projection, it will be an input for MLP blocks
            // attn_output * w_o^T
            // (num_tokens, 2048) * (2048, 2048) -> (num_tokens, 2048)
            // same as Q projection, so copy paste
            __nv_bfloat16 *o_proj = buf.buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH,
                         num_tokens,
                         EMBEDDING_LENGTH,
                         &alpha,
                         weights.w_o[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         q_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &beta,
                         o_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (num_tokens, 2048) + (num_tokens, 2048) -> (num_tokens, 2048)
            residualKernel<<<num_tokens, EMBED_HALF>>>(buf.hidden_state, o_proj);
            // post attention RMS Norm
            rmsNormKernel<<<num_tokens, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.post_attn_layernorms[layer], num_tokens);

            // SwiGLU time - just MLP + SiLU
            // gate = hidden_state (rms-normed) * mlp_gate_proj ^ T
            // HIDDEN_DIM = 8192
            // (num_tokens, 2048) * (2048, 8192) -> (num_tokens, 8192)
            // my data is row major so transpose trick
            // gate ^T = (mlp_gate_proj ^ T)^T * hidden_state^T
            // gate ^T = mlp_gate_proj * hidden_state^T
            // (num_tokens, 8192)^T = (8192, 2048) * (2048, num_tokens)
            // but data is perceived as column major so I need to transpose mlp_gate_proj
            // to make it work
            // m 8192 n num_tokens k 2048 lda 2048 ldb 2048 ldc 8192
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,
                         num_tokens,
                         EMBEDDING_LENGTH,
                         &alpha,
                         weights.mlp_gate_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf.rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &beta,
                         buf.gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // up, the same dims as gate
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,
                         num_tokens,
                         EMBEDDING_LENGTH,
                         &alpha,
                         weights.mlp_up_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf.rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &beta,
                         buf.up,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // SiLU
            // after_silu = SiLU(gate) * up (element-wise multication)
            // after_silu = gate * (1 / (1 + e^(-gate))) * up
            // gate is dim (num_tokens, 8192), up too
            siluKernel<<<num_tokens, MAX_THREADS_PER_BLOCK>>>(buf.gate, buf.up); // gate = after_silu now

            // down projection
            // output = post-silu * down_proj^T
            // dims: (num_tokens, 8192) * (2048, 8192) ^ T = (num_tokens, 8192) * (8192, 2048) = (num_tokens, 2048)
            // output^T = (down_proj^T)^T * post-silu^T
            // output^T = down_proj * post-silu^T
            // cublas sees them already as transposed so only down_proj I need to transpose
            // dims = (2048, 8192) * (8192, num_tokens) = (2048, num_tokens)
            // m: 2048 n: num_tokens, k: 8192
            // lda: 8192, ldb: 8192, ldc: 2048
            __nv_bfloat16 *down = buf.buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH,
                         num_tokens,
                         HIDDEN_DIM,
                         &alpha,
                         weights.mlp_down_proj[layer],
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         buf.gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         &beta,
                         down,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (num_tokens, 2048) + (num_tokens, 2048) -> (num_tokens, 2048)
            residualKernel<<<num_tokens, EMBED_HALF>>>(buf.hidden_state, down);
        }
        rmsNormKernel<<<num_tokens, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.norm, num_tokens);

        // Only the last token of each prompt produces the first generated token, so gather
        // those num_seqs rows and run the lm_head on them alone. The sequential path ran
        // this GEMM over all prompt_len rows and threw away everything but the last one.
        gatherRowsKernel<<<num_seqs, EMBED_HALF>>>(buf.rms_norms, buf.last_hidden, buf.last_token_rows, num_seqs);

        // logits = last_hidden * weights.embed_tokens^T
        // dim last_hidden: (num_seqs, 2048), dim embed_tokens: (128256, 2048)
        // because of the cublas trick the logits come out transposed, so m and n swap:
        // logits^T = weights.embed_tokens * last_hidden^T
        // m 128256, n num_seqs, k 2048, lda 2048, ldb 2048, ldc 128256
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,
                     num_seqs,
                     EMBEDDING_LENGTH,
                     &alpha,
                     weights.embed_tokens,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     buf.last_hidden,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &beta,
                     buf.embed_proj,
                     CUDA_R_16BF,
                     VOCAB_SIZE,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        // Greedy sampling on the device, so the only thing that crosses PCIe is one token id
        // per prompt. No penalty mask here: a prompt being prefilled has emitted nothing yet,
        // which is exactly what the host-side scan did (it never applied the penalty either).
        argmaxPartialKernel<<<dim3(ARGMAX_CHUNKS_PER_ROW, num_seqs), ARGMAX_BLOCK_THREADS>>>(
            buf.embed_proj, num_seqs, nullptr, nullptr, buf.argmax_values, buf.argmax_indices);
        argmaxFinalizeKernel<<<num_seqs, ARGMAX_CHUNKS_PER_ROW>>>(
            buf.argmax_values, buf.argmax_indices, num_seqs, buf.sampled_tokens);
        // the first generated token counts towards the repetition penalty of the steps that
        // follow, same as when the host pushed it into generated_tokens
        markSampledTokensKernel<<<1, BATCH_SIZE>>>(buf.sampled_tokens, buf.logit_row_slots, num_seqs, buf.penalty_mask);

        std::vector<int> sampled(num_seqs);
        cudaMemcpy(sampled.data(), buf.sampled_tokens, num_seqs * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        auto prefill_end = std::chrono::high_resolution_clock::now();
        float prefill_ms = std::chrono::duration<float, std::milli>(prefill_end - prefill_start).count();

        for (int seq = 0; seq < num_seqs; ++seq)
        {
            int slot = items[pass_items[seq]].slot;
            int request_id = items[pass_items[seq]].request_id;
            int prompt_len = (int)items[pass_items[seq]].tokens.size();
            const int max_token_idx = sampled[seq];

            const bool is_eos = (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID);

            // One message per generated token, carrying no text: the client only counts
            // tokens and reports speeds, so decoding ids back into strings - and keeping a
            // VOCAB_SIZE table of them around to do it - bought nothing.
            if (!is_eos)
            {
                json out_j;
                out_j["id"] = request_id;
                out_j["slot"] = slot;
                std::cout << out_j.dump() << "\n" << std::flush;
            }

            if (!is_eos)
            {
                slots.generated_tokens[slot].push_back(max_token_idx);
            }
            slots.last_generated_tokens[slot] = max_token_idx;
            slots.current_prompt_len[slot] = prompt_len;

            // time_ms is the latency of the shared pass, speed its aggregate throughput -
            // every prompt in the pass waited exactly as long as the slowest one
            json stat_j;
            stat_j["type"] = "prefill_stats";
            stat_j["id"] = request_id;
            stat_j["slot"] = slot;
            stat_j["speed"] = num_tokens / (prefill_ms / 1000.0f);
            stat_j["time_ms"] = prefill_ms;
            stat_j["prompt_tokens"] = prompt_len;
            stat_j["batch_tokens"] = num_tokens;
            stat_j["batch_prompts"] = num_seqs;
            std::cout << stat_j.dump() << "\n" << std::flush;

            // A prompt whose first sampled token is already EOS is finished here. Letting it
            // through to decode used to run it for the full MAX_NEW_TOKENS_GENERATED steps,
            // burning a slot (and its pages) on a sequence that had nothing left to say.
            if (is_eos)
            {
                finishSequence(slot, request_id, nullptr, slots, kv);
            }
        }
    }
}

// Advances every running sequence by one token.
int decodeStep(DeviceBuffers &buf,
               const Weights &weights,
               cublasHandle_t cublas_handle,
               SlotState &slots,
               KVCacheState &kv)
{
    // every GEMM in decode is a plain C = A * B, same as in prefill
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // decode needs the active sequences packed contiguously, one row per slot
    std::vector<int> active_slots;
    std::vector<int> active_tokens;
    active_slots.reserve(BATCH_SIZE);
    active_tokens.reserve(BATCH_SIZE);
    for (int slot = 0; slot < BATCH_SIZE; ++slot)
    {
        if (slots.is_slot_free[slot])
        {
            continue;
        }
        active_slots.push_back(slot);
        active_tokens.push_back(slots.last_generated_tokens[slot]);
    }
    int num_active_slots = (int)active_slots.size();
    if (num_active_slots == 0)
    {
        return 0;
    }

    // copy useful data to gpu
    cudaMemcpy(buf.last_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buf.active_slots, active_slots.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<int> seq_lens(num_active_slots);
    for (int slot = 0; slot < num_active_slots; ++slot)
    {
        int active_slot = active_slots[slot];
        seq_lens[slot] = slots.current_prompt_len[active_slot] + 1;
    }
    cudaMemcpy(buf.seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice);

    embeddingGatherDecodeKernel<<<num_active_slots, EMBED_HALF>>>(buf.last_tokens, num_active_slots, buf.hidden_state, weights.embed_tokens);
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        rmsNormKernel<<<num_active_slots, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.input_layernorm[layer], num_active_slots);
        __nv_bfloat16 *q_proj = buf.buf_2048_1;
        // q proj (num_prompts, 2048)
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     EMBEDDING_LENGTH, // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &alpha,
                     weights.w_q[layer], // A
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH, // lda
                     buf.rms_norms,        // B
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH, // ldb
                     &beta,
                     q_proj, // C
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH, // ldc
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
        // k proj (1, 512), writing output to next position in current layer's K cache
        // K proj = rms_norms (num_prompt, 2048) * W_k (512, 2048)
        // W_k is actually stored as 512, 2048 (out features, in features)
        // so that's why we need to transpose it
        // all the data is stored in row major and cublas reads it as column major
        // so all the data appears as transposed
        // so data actually apppears as (2048, num_prompt) * (2048, 512)
        // the output of matmul will also be produced as transposed, so we can say that
        // in our mental model we talk about K_proj^T
        // and to get K_proj^T we can do transposition trick and write the cublas call as
        // W_k^T * rms_nroms
        // so we end up with: K_proj^T = W_k^T (512, 2048) * rms_norms (2048, num_prompt)
        // result dim is K_proj^T = (512, num_prompt)
        // but it's transposed, so in fact we get correct output dimension (num_prompt, 512)
        // it was great for num_prompt=1, but the problem is that prompts have different length
        // that's why we have vector of current_prompt_len, but also we can't write to K_proj
        // directly, so I write to temp buffer kv_proj_batched_buffer and the scatter
        // output to K_proj in a loop
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     KV_DIM,           // m = 512
                     num_active_slots, // n = num prompts
                     EMBEDDING_LENGTH, // k = 2048
                     &alpha,
                     weights.w_k[layer], // A
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
                     // so the gap between subsequent elements is 2048
                     buf.rms_norms, // B
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH, // ldb, same reason for rms_norms
                     &beta,
                     buf.k_proj_temp_buf, // TODO C
                     CUDA_R_16BF,
                     KV_DIM, // ldc = 512
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        // same
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     KV_DIM,
                     num_active_slots,
                     EMBEDDING_LENGTH,
                     &alpha,
                     weights.w_v[layer],
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     buf.rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &beta,
                     buf.v_proj_temp_buf,
                     CUDA_R_16BF,
                     KV_DIM,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            ropeDecodeKernel<<<1, EMBEDDING_LENGTH / 2>>>(&q_proj[slot * EMBEDDING_LENGTH], slots.current_prompt_len[active_slot],
                                                         EMBEDDING_LENGTH, d_cos_table, d_sin_table);
            ropeDecodeKernel<<<1, KV_DIM / 2>>>(buf.k_proj_temp_buf + slot * KV_DIM, slots.current_prompt_len[active_slot],
                                                KV_DIM, d_cos_table, d_sin_table);
        }

        // PagedAttn scatter k and v from a temp buffer, like in the prefill
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            int seq_len = slots.current_prompt_len[active_slot]; // + generated tokens?
            int logical_block_idx = seq_len / BLOCK_SIZE;
            int token_in_block_idx = seq_len % BLOCK_SIZE;
            int block = kv.block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
            if (token_in_block_idx == 0)
            {
                int physical_block_idx = kv.free_blocks.back();
                kv.free_blocks.pop_back();
                block = physical_block_idx;
                kv.block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
            }
            __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv.cache + block * BLOCK_BYTES + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
            __nv_bfloat16 *k_proj_ptr = buf.k_proj_temp_buf + slot * KV_DIM;
            cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

            __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv.cache + block * BLOCK_BYTES + V_OFFSET + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
            __nv_bfloat16 *v_proj_ptr = buf.v_proj_temp_buf + slot * KV_DIM;
            cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        }

        // synchronize block table on cpu with block table on gpu (for attention)
        cudaMemcpy(kv.block_table_gpu, kv.block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

        pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(
            layer, num_active_slots, q_proj, kv.cache, kv.block_table_gpu, buf.seq_lens, buf.active_slots, buf.buf_2048_1);

        __nv_bfloat16 *o_proj = buf.buf_2048_2;
        // (1, 2048) * (2048, 2048) -> (1, 2048)
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     EMBEDDING_LENGTH, // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &alpha,
                     weights.w_o[layer],
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     buf.buf_2048_1,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &beta,
                     o_proj,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        residualKernel<<<num_active_slots, EMBED_HALF>>>(buf.hidden_state, o_proj);

        rmsNormKernel<<<num_active_slots, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.post_attn_layernorms[layer], num_active_slots);

        // MLP
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     HIDDEN_DIM,       // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &alpha,
                     weights.mlp_gate_proj[layer],
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     buf.rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &beta,
                     buf.gate,
                     CUDA_R_16BF,
                     HIDDEN_DIM,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        // (1, 2048) * (2048, 8192) -> (1, 8192)
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     HIDDEN_DIM,       // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &alpha,
                     weights.mlp_up_proj[layer],
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     buf.rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &beta,
                     buf.up,
                     CUDA_R_16BF,
                     HIDDEN_DIM,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        siluKernel<<<num_active_slots, MAX_THREADS_PER_BLOCK>>>(buf.gate, buf.up);

        __nv_bfloat16 *down = buf.buf_2048_2;
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     EMBEDDING_LENGTH, // m
                     num_active_slots, // n
                     HIDDEN_DIM,       // k
                     &alpha,
                     weights.mlp_down_proj[layer],
                     CUDA_R_16BF,
                     HIDDEN_DIM,
                     buf.gate,
                     CUDA_R_16BF,
                     HIDDEN_DIM,
                     &beta,
                     down,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        residualKernel<<<num_active_slots, EMBED_HALF>>>(buf.hidden_state, down);
    }

    rmsNormKernel<<<num_active_slots, EMBED_HALF>>>(buf.hidden_state, buf.rms_norms, weights.norm, num_active_slots);

    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 VOCAB_SIZE,       // m
                 num_active_slots, // n
                 EMBEDDING_LENGTH, // k
                 &alpha,
                 weights.embed_tokens,
                 CUDA_R_16BF,
                 EMBEDDING_LENGTH,
                 buf.rms_norms,
                 CUDA_R_16BF,
                 EMBEDDING_LENGTH,
                 &beta,
                 buf.embed_proj,
                 CUDA_R_16BF,
                 VOCAB_SIZE,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);

    // Penalise and argmax on the device. This used to be the single most expensive
    // host-side step of a decode iteration: a 7.58 MiB logits download, then per slot a
    // VOCAB_SIZE bf16->float conversion into a fresh std::vector plus a serial scan, i.e.
    // ~4M elements touched three times on one core while the GPU sat idle. buf.active_slots
    // already holds the slot that owns each logits row, so it doubles as the mask index.
    argmaxPartialKernel<<<dim3(ARGMAX_CHUNKS_PER_ROW, num_active_slots), ARGMAX_BLOCK_THREADS>>>(
        buf.embed_proj, num_active_slots, buf.penalty_mask, buf.active_slots,
        buf.argmax_values, buf.argmax_indices);
    argmaxFinalizeKernel<<<num_active_slots, ARGMAX_CHUNKS_PER_ROW>>>(
        buf.argmax_values, buf.argmax_indices, num_active_slots, buf.sampled_tokens);
    markSampledTokensKernel<<<1, BATCH_SIZE>>>(buf.sampled_tokens, buf.active_slots, num_active_slots, buf.penalty_mask);

    std::vector<int> sampled(num_active_slots);
    cudaMemcpy(sampled.data(), buf.sampled_tokens, num_active_slots * sizeof(int), cudaMemcpyDeviceToHost);

    for (int slot = 0; slot < num_active_slots; ++slot)
    {
        int active_slot = active_slots[slot];
        const int max_token_idx = sampled[slot];
        const int request_id = slots.slot_request_id[active_slot];
        const bool is_eos = (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID);

        if (!is_eos)
        {
            json out_j;
            out_j["id"] = request_id;
            out_j["slot"] = active_slot;
            std::cout << out_j.dump() << "\n" << std::flush;
        }

        if (is_eos || slots.current_prompt_len[active_slot] == MAX_SEQ_LEN - 1 || slots.generated_tokens[active_slot].size() >= MAX_NEW_TOKENS_GENERATED)
        {
            finishSequence(active_slot, request_id, nullptr, slots, kv);
        }
        else
        {
            slots.last_generated_tokens[active_slot] = max_token_idx;
            slots.generated_tokens[active_slot].push_back(max_token_idx);
            slots.current_prompt_len[active_slot] = slots.current_prompt_len[active_slot] + 1;
        }
    }

    return num_active_slots;
}
