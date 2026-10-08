#include "engine.cuh"

#include <cassert>
#include <cstdlib>
#include <chrono>
#include <iostream>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include "kernels.cuh"
#include "utils.cuh"

using json = nlohmann::json;

// Benchmark knob, set from IGNORE_EOS at startup. With it on, a sequence runs for exactly
// its output budget instead of stopping at <|eot_id|>. That is what vLLM's own serving
// benchmark does, and it is what makes the comparison fair here: every request generates
// the same number of tokens in every build, so the two engines do identical work and the
// "oracle" reservation policy really is an oracle rather than an estimate.
bool g_ignore_eos = false;

// Set from the ORCA_POLICY environment variable at startup; see config.h for what the
// three policies mean and why all three are worth measuring.
ReservePolicy g_reserve_policy = ReservePolicy::Max;

const char *reservePolicyName()
{
    switch (g_reserve_policy)
    {
    case ReservePolicy::Pow2:
        return "pow2";
    case ReservePolicy::Oracle:
        return "oracle";
    default:
        return "max";
    }
}

// Serving telemetry, not scheduler state.
static const std::chrono::high_resolution_clock::time_point g_engine_start =
    std::chrono::high_resolution_clock::now();

SlotState::SlotState()
    : is_slot_free(BATCH_SIZE, true),
      generated_tokens(BATCH_SIZE),
      last_generated_tokens(BATCH_SIZE),
      current_prompt_len(BATCH_SIZE, 0),
      slot_request_id(BATCH_SIZE, -1),
      remaining_budget(BATCH_SIZE, 0),
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

    // K and V of the packed batch, before they are scattered into the KV cache.
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

int g_kv_pool_tokens = 0;

// ---- binary buddy allocator over the token-slot pool ----
// The paper's Orca "uses the buddy allocation algorithm to determine the memory address to store
// KV cache", so this is what the baseline must use. A plain first-fit free list with coalescing -
// what this code used to do - is a *stronger* allocator than the paper's baseline: it rounds to a
// 16-token granularity instead of to a power of two, and it fragments less. Using it would have
// flattered the baseline against the paper's own description of it.

static int ceilLog2(int n)
{
    int order = 0;
    while ((1 << order) < n)
    {
        ++order;
    }
    return order;
}

// Places every self-aligned power-of-two run that lies entirely inside the real pool into its
// free list. The pool is viewed as one buddy tree over the next power of two above its size; the
// straddling and past-the-end parts are simply never made free, which is what keeps them from
// ever being allocated or merged into without any extra bookkeeping.
static void buddyInitRange(KVCacheState &kv, int start, int order, int pool_tokens)
{
    if (start >= pool_tokens)
    {
        return; // past the end of real memory: not a block, never free
    }
    const int size = 1 << order;
    if (start + size <= pool_tokens)
    {
        kv.free_lists[order].push_back(start);
        return; // fully inside the pool
    }
    if (order == 0)
    {
        return;
    }
    buddyInitRange(kv, start, order - 1, pool_tokens);
    buddyInitRange(kv, start + size / 2, order - 1, pool_tokens);
}

static void buddyInit(KVCacheState &kv)
{
    int root_order = ceilLog2(g_kv_pool_tokens);
    if (root_order > KV_MAX_ORDER)
    {
        std::cerr << "KV pool of " << g_kv_pool_tokens << " tokens exceeds KV_MAX_ORDER\n";
        std::exit(1);
    }
    buddyInitRange(kv, 0, root_order, g_kv_pool_tokens);
}

// Smallest free run of at least 2^want_order slots, split down to exactly that order.
// Returns -1 if no run of that order can be formed, which is the external-fragmentation case.
static int buddyAlloc(KVCacheState &kv, int want_order)
{
    int k = want_order;
    while (k <= KV_MAX_ORDER && kv.free_lists[k].empty())
    {
        ++k;
    }
    if (k > KV_MAX_ORDER)
    {
        return -1;
    }
    int start = kv.free_lists[k].back();
    kv.free_lists[k].pop_back();
    // split the surplus half away, order by order, keeping the lower half
    while (k > want_order)
    {
        --k;
        kv.free_lists[k].push_back(start + (1 << k));
    }
    return start;
}

static void buddyFree(KVCacheState &kv, int start, int order)
{
    while (order < KV_MAX_ORDER)
    {
        const int buddy = start ^ (1 << order);
        auto &list = kv.free_lists[order];
        auto it = std::find(list.begin(), list.end(), buddy);
        if (it == list.end())
        {
            break; // buddy is in use, straddles the end of the pool, or is past it: no merge
        }
        list.erase(it);
        start = std::min(start, buddy);
        ++order;
    }
    kv.free_lists[order].push_back(start);
}


KVCacheState allocateKVCache()
{
    // One slab of token slots plus a free list that starts as a single run covering all of it.
    //
    // The size is not chosen, it is whatever is left. In the paper both systems run the same model
    // on the same GPU, so the VRAM that remains once the weights and the activations are allocated
    // becomes the KV cache, identically on both sides - "how many MiB does each side get" is not a
    // question that exists there. Hardcoding it here is how the two builds silently drifted to
    // 768 MiB and 1000 MiB, a 30% gap in the one resource the comparison is about. So: allocate
    // everything else first (see main.cu), then claim what is free minus a margin.
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem <= KV_SAFETY_MARGIN_BYTES)
    {
        std::cerr << "Only " << free_mem / B_TO_MB << " MiB of VRAM free, which does not cover the "
                  << KV_SAFETY_MARGIN_BYTES / B_TO_MB << " MiB safety margin\n";
        std::exit(1);
    }
    size_t kv_bytes = free_mem - KV_SAFETY_MARGIN_BYTES;
    g_kv_pool_tokens = (int)(kv_bytes / KV_BYTES_PER_TOKEN);
    if (g_kv_pool_tokens < MAX_TOKENS_PER_SEQUENCE)
    {
        std::cerr << "KV pool holds " << g_kv_pool_tokens << " tokens, too small for even one "
                  << MAX_TOKENS_PER_SEQUENCE << "-token sequence\n";
        std::exit(1);
    }
    std::cerr << "KV pool: " << (size_t)g_kv_pool_tokens * KV_BYTES_PER_TOKEN / B_TO_MB << " MiB = "
              << g_kv_pool_tokens << " tokens of context\n";

    KVCacheState kv{};
    kv.cache = (__nv_bfloat16 *)allocDevice((size_t)g_kv_pool_tokens * KV_BYTES_PER_TOKEN, "KV cache");
    kv.free_lists.assign(KV_MAX_ORDER + 1, {});
    buddyInit(kv);
    kv.slot_base.assign(MAX_SEQUENCES, -1);
    kv.slot_cap.assign(MAX_SEQUENCES, 0);
    kv.slot_base_gpu = (int *)allocDevice(MAX_SEQUENCES * sizeof(int), "slot bases");
    kv.slot_cap_gpu = (int *)allocDevice(MAX_SEQUENCES * sizeof(int), "slot caps");
    kv.reserved_tokens = 0;
    kv.free_tokens = g_kv_pool_tokens;
    kv.fragmentation_failures = 0;
    return kv;
}

int reservationTokens(int prompt_len, int max_tokens)
{
    // The paper's §6.1 wording decides all three of these, and the previous implementation got
    // all three slightly wrong - two of them in the baseline's favour.
    if (g_reserve_policy == ReservePolicy::Max)
    {
        // "always reserves the space up to the maximum sequence length of the model, i.e. 2048
        // tokens". It used to reserve MAX_TOKENS_PER_SEQUENCE, which is only the same number
        // because the generation limits now add up to MAX_SEQ_LEN.
        return MAX_SEQ_LEN;
    }
    if (g_reserve_policy == ReservePolicy::Pow2)
    {
        // "over-reserves the space for outputs by at most 2x. For example, if the true output
        // length is 25, it reserves 32 positions for outputs." The rounding is on the OUTPUT and
        // the prompt is added on top. It used to round the total, which for a 40-token prompt and
        // a 25-token output reserved 128 instead of 72 - nearly twice the paper's figure.
        int p = 1;
        while (p < max_tokens)
        {
            p *= 2;
        }
        int reserved = prompt_len + p;
        return std::min(reserved, MAX_SEQ_LEN);
    }
    // Oracle: "the system has the knowledge of the lengths of the outputs that will be actually
    // generated", so exactly what the sequence will need and not a token more. No rounding here -
    // the buddy allocator applies its own, which is the paper's model of Orca.
    return std::min(prompt_len + max_tokens, MAX_SEQ_LEN);
}

bool kvReserve(KVCacheState &kv, int slot, int tokens)
{
    const int order = ceilLog2(tokens);
    const int size = 1 << order;
    const int start = buddyAlloc(kv, order);
    if (start < 0)
    {
        // Enough free total but no aligned run of this order: contiguity itself is what failed.
        // Worth counting separately from "the pool is simply full", because paging has no such
        // failure mode - any free page fits any sequence.
        if (kv.free_tokens >= size)
        {
            ++kv.fragmentation_failures;
        }
        return false;
    }
    kv.slot_base[slot] = start;
    // The run really is `size` long, not `tokens`: the gap is buddy's rounding, and charging the
    // sequence for it is the point. kvKOffset uses the cap to find V, so it must be the true size.
    kv.slot_cap[slot] = size;
    kv.reserved_tokens += size;
    kv.free_tokens -= size;
    return true;
}

void kvRelease(KVCacheState &kv, int slot)
{
    if (kv.slot_base[slot] < 0)
    {
        return;
    }
    const int size = kv.slot_cap[slot];
    buddyFree(kv, kv.slot_base[slot], ceilLog2(size));
    kv.reserved_tokens -= size;
    kv.free_tokens += size;
    kv.slot_base[slot] = -1;
    kv.slot_cap[slot] = 0;
}

void kvSyncSlotTables(const KVCacheState &kv)
{
    cudaMemcpy(kv.slot_base_gpu, kv.slot_base.data(), MAX_SEQUENCES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kv.slot_cap_gpu, kv.slot_cap.data(), MAX_SEQUENCES * sizeof(int), cudaMemcpyHostToDevice);
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
    config_j["mechanism"] = "contiguous";
    config_j["batch_size"] = BATCH_SIZE;
    config_j["max_batch_tokens"] = MAX_BATCH_TOKENS;
    config_j["max_prompt_len"] = MAX_PROMPT_LEN;
    config_j["max_new_tokens"] = MAX_NEW_TOKENS_GENERATED;
    config_j["reserve_policy"] = reservePolicyName();
    config_j["kv_pool_tokens"] = g_kv_pool_tokens;
    std::cout << config_j.dump() << "\n" << std::flush;
}

std::vector<PrefillBatchItem> admitQueuedRequests(std::deque<Request> &queue, SlotState &slots,
                                                  KVCacheState &kv)
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
            // The reservation has to be placed before the prompt can be taken: the region
            // cannot grow later, so if the whole worst case does not fit now, the request
            // waits. This is the line that decides this build's concurrency, and under
            // ORCA_POLICY=max it is also the line that makes it 31.
            int prompt_len = std::min((int)queue.front().tokens.size(), MAX_PROMPT_LEN);
            int reserved = reservationTokens(prompt_len, queue.front().max_tokens);
            if (!kvReserve(kv, slot, reserved))
            {
                break;
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
        slots.remaining_budget[slot] = request.max_tokens;
        slots.generated_tokens[slot].clear();
        // start the clock now, so a sequence that is retired during prefill (EOS on its
        // first token, or a failure) still reports a sane duration instead of whatever
        // the previous owner of this slot left behind
        slots.decode_start[slot] = std::chrono::high_resolution_clock::now();
        items.push_back({slot, request.id, request.max_tokens, 0, std::move(request.tokens)});
    }
    if (!items.empty())
    {
        kvSyncSlotTables(kv);
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
    // The whole reservation comes back in one piece - that is the one thing contiguity
    // makes cheap. It also comes back only now: however little of it the sequence actually
    // used, the rest was unavailable to everyone else for the sequence's entire lifetime.
    kvRelease(kv, slot);
}

void emitStepTelemetry(const SlotState &slots, const KVCacheState &kv, const std::deque<Request> &queue)
{
    int running = 0;
    long long live_tokens = 0;
    for (int slot = 0; slot < BATCH_SIZE; ++slot)
    {
        if (slots.is_slot_free[slot])
        {
            continue;
        }
        ++running;
        live_tokens += slots.current_prompt_len[slot] + 1;
    }
    size_t waiting = 0;
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        waiting = queue.size();
    }

    json t;
    t["type"] = "step";
    t["t_ms"] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - g_engine_start).count();
    t["running"] = running;
    t["waiting"] = (int)waiting;
    // reserved is what the pool has handed out; live is what holds a real token. Everything
    // between them is reservation the sequence has not reached yet and may never reach.
    t["kv_tokens_reserved"] = kv.reserved_tokens;
    t["kv_tokens_live"] = live_tokens;
    t["kv_tokens_total"] = g_kv_pool_tokens;
    t["fragmentation_failures"] = kv.fragmentation_failures;
    std::cout << t.dump() << "\n" << std::flush;
}

void enforceSeqCapacity(SlotState &slots, KVCacheState &kv)
{
    for (int slot = 0; slot < BATCH_SIZE; ++slot)
    {
        if (slots.is_slot_free[slot])
        {
            continue;
        }
        if (slots.current_prompt_len[slot] >= kv.slot_cap[slot])
        {
            std::cerr << "Slot " << slot << " filled its KV reservation (" << kv.slot_cap[slot]
                      << " tokens), retiring it\n";
            finishSequence(slot, slots.slot_request_id[slot], "sequence_too_long", slots, kv);
        }
    }

    // No eviction pass below this: a slot's region is reserved for its whole lifetime, so a
    // sequence that was admitted can always be served to the end. The paged build has to
    // evict here (newest first) because its per-step page allocation can find the pool
    // empty - free_blocks.back() with no fallback - even for sequences already running.
}

void prefillBatch(std::vector<PrefillBatchItem> &items,
                  DeviceBuffers &buf,
                  const Weights &weights,
                  cublasHandle_t cublas_handle,
                  SlotState &slots,
                  KVCacheState &kv)
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

        // No KV reservation step. Every prompt of the pass was admitted into a slot, and a
        // slot's cache region is already sized for the longest sequence it can hold, so the
        // pass can simply write into it. This is where the paged build reserves
        // N_LAYERS * ceil(prompt_len / BLOCK_SIZE) pages per prompt out of its pool, has to
        // handle the pool coming up short, and uploads the updated block table to the device.

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

            // scatter K and V of the entire batch into the cache, at each token's own position
            // KV_DIM is 512, so one thread per K element and per V element fits in a block
            scatterKVPackedKernel<<<num_tokens, KV_DIM>>>(layer, num_tokens, buf.k_proj_temp_buf, buf.v_proj_temp_buf,
                                                          kv.cache, kv.slot_base_gpu, kv.slot_cap_gpu,
                                                          buf.token_slot_ids, buf.token_positions);

            // Attention, in place over q_proj. Reads the packed K/V we just computed rather
            // than the cache, which is the same data and saves the strided re-read.
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

            const bool is_eos = !g_ignore_eos && (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID);

            // One message per generated token, carrying no text: the client only counts
            // tokens and reports speeds, so decoding ids back into strings - and keeping a
            // VOCAB_SIZE table of them around to do it - bought nothing.
            if (!is_eos)
            {
                json out_j;
                out_j["id"] = request_id;
                out_j["slot"] = slot;
                // the id itself, so the benchmark can checksum a request's output and show
                // that the two cache designs are numerically identical, not just equally fast
                out_j["tok"] = max_token_idx;
                std::cout << out_j.dump() << "\n" << std::flush;
            }

            if (!is_eos)
            {
                slots.generated_tokens[slot].push_back(max_token_idx);
                --slots.remaining_budget[slot];
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
            // ...and so is one that spent its whole output budget on that token.
            if (is_eos || slots.remaining_budget[slot] <= 0)
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
    // token_positions is a prefill buffer (MAX_BATCH_TOKENS ints) and prefill never runs
    // concurrently with decode, so decode borrows it rather than holding its own.
    std::vector<int> positions(num_active_slots);
    for (int row = 0; row < num_active_slots; ++row)
    {
        positions[row] = slots.current_prompt_len[active_slots[row]];
    }
    cudaMemcpy(buf.token_positions, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice);
    // the slot tables only change when a sequence is admitted or retired, but they are
    // 160 ints each, so there is nothing to gain from tracking that
    kvSyncSlotTables(kv);

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

        // One launch per projection instead of one per slot - identical to the paged build,
        // and for the same reason: at 160 slots the per-slot form is thousands of launches
        // a step, which would show up as a difference between the two builds that has
        // nothing to do with how either addresses its cache.
        ropeDecodeBatchKernel<<<num_active_slots, EMBEDDING_LENGTH / 2>>>(
            q_proj, num_active_slots, EMBEDDING_LENGTH, buf.token_positions, d_cos_table, d_sin_table);
        ropeDecodeBatchKernel<<<num_active_slots, KV_DIM / 2>>>(
            buf.k_proj_temp_buf, num_active_slots, KV_DIM, buf.token_positions, d_cos_table, d_sin_table);

        // Scatter k and v from a temp buffer, like in the prefill. One launch, same as the
        // paged build, but the destination is arithmetic: no page to allocate when a
        // sequence crosses a block boundary, and no block table to re-upload to the device.
        scatterKVDecodeKernel<<<num_active_slots, KV_DIM>>>(
            layer, num_active_slots, buf.k_proj_temp_buf, buf.v_proj_temp_buf, kv.cache,
            kv.slot_base_gpu, kv.slot_cap_gpu, buf.active_slots, buf.token_positions);

        contiguousAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(
            layer, num_active_slots, q_proj, kv.cache, kv.slot_base_gpu, kv.slot_cap_gpu,
            buf.seq_lens, buf.active_slots, buf.buf_2048_1);

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
        const bool is_eos = !g_ignore_eos && (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID);

        if (!is_eos)
        {
            json out_j;
            out_j["id"] = request_id;
            out_j["slot"] = active_slot;
            out_j["tok"] = max_token_idx;
            std::cout << out_j.dump() << "\n" << std::flush;
        }

        if (!is_eos)
        {
            slots.last_generated_tokens[active_slot] = max_token_idx;
            slots.generated_tokens[active_slot].push_back(max_token_idx);
            slots.current_prompt_len[active_slot] = slots.current_prompt_len[active_slot] + 1;
            --slots.remaining_budget[active_slot];
        }

        // The output budget is per request now, not one global MAX_NEW_TOKENS_GENERATED for
        // everybody: that is the property this build has to reserve for and the paged one
        // does not, so a benchmark that fixes it measures neither.
        if (is_eos || slots.remaining_budget[active_slot] <= 0 ||
            slots.current_prompt_len[active_slot] >= kv.slot_cap[active_slot])
        {
            finishSequence(active_slot, request_id, nullptr, slots, kv);
        }
    }

    return num_active_slots;
}
