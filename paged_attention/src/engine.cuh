#pragma once

// The inference engine: the scheduler state every step reads and writes, and the two
// forward passes over the model (packed prefill and batched decode) that make up a step.
// main.cu is left with nothing but startup and the step loop.

#include <chrono>
#include <deque>
#include <string>
#include <vector>

#include "config.h"
#include "model.cuh"
#include "request_queue.h"

// One queued prompt waiting to be prefilled, together with the batch slot it was admitted into.
struct PrefillBatchItem
{
    int slot;
    int request_id;
    std::vector<int> tokens;
};

// Device scratch shared by prefill and decode. Every per-token buffer holds
// MAX_BATCH_TOKENS rows, which also covers decode (at most BATCH_SIZE rows).
struct DeviceBuffers
{
    int *input_tokens;    // [MAX_BATCH_TOKENS] packed prompt tokens of the whole batch
    int *token_positions; // [MAX_BATCH_TOKENS] position of each token inside its own prompt
    int *token_seq_start; // [MAX_BATCH_TOKENS] packed index where that token's prompt starts
    int *token_slot_ids;  // [MAX_BATCH_TOKENS] batch slot each token belongs to
    int *last_token_rows; // [BATCH_SIZE]       packed index of every prompt's last token
    // prefillAttention query tiles: runs of at most PREFILL_ATTN_QUERIES_PER_TILE
    // consecutive tokens that belong to the same prompt
    int *tile_token_begin;  // [MAX_PREFILL_ATTN_TILES]
    int *tile_token_count;  // [MAX_PREFILL_ATTN_TILES]
    __nv_bfloat16 *hidden_state;
    __nv_bfloat16 *rms_norms;
    __nv_bfloat16 *buf_2048_1; // shared between q_proj and the attention output
    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    // K/V of the pass before they are scattered into the paged cache; decode reuses them
    // for its one-row-per-slot projections, which never run at the same time as a prefill
    __nv_bfloat16 *k_proj_temp_buf;
    __nv_bfloat16 *v_proj_temp_buf;
    __nv_bfloat16 *gate;
    __nv_bfloat16 *up;
    __nv_bfloat16 *last_hidden; // [BATCH_SIZE, EMBEDDING_LENGTH] last token of every prompt
    __nv_bfloat16 *embed_proj;  // [BATCH_SIZE, VOCAB_SIZE]       logits, last token only
    // ---- decode only ----
    // Decode works on one token per active slot, so these are all BATCH_SIZE-sized.
    int *last_tokens;  // [BATCH_SIZE] token each active slot feeds back in this step
    int *active_slots; // [BATCH_SIZE] slot id of each row, for the block table walk
    int *seq_lens;     // [BATCH_SIZE] length each active sequence has reached
    // TODO: move argmax to GPU and get rid of the CPU<->GPU tokens moves these exist for
};

// Per-slot scheduler state. A slot is a decode seat: it is taken from the moment a prompt is
// admitted until finishSequence retires it, and everything about the sequence occupying it
// lives at the same index across all of these.
struct SlotState
{
    std::vector<bool> is_slot_free;                   // false while a sequence owns the slot
    std::vector<std::vector<int>> generated_tokens;   // what the sequence has emitted so far
    std::vector<int> last_generated_tokens;           // token to feed back in the next step
    std::vector<int> current_prompt_len;              // tokens already in the KV cache
    std::vector<int> slot_request_id;                 // which request owns the slot, -1 if none
    std::vector<std::chrono::high_resolution_clock::time_point> decode_start;

    SlotState();
};

// The paged KV cache and the page table that maps (slot, layer, logical page) to a physical
// page. The table is kept on both sides: the host allocates pages into `block_table` and
// uploads it to `block_table_gpu` for the attention kernels to walk.
struct KVCacheState
{
    __nv_bfloat16 *cache;          // NUM_BLOCKS pages of BLOCK_BYTES
    std::vector<int> block_table;  // [MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ], -1 = unmapped
    int *block_table_gpu;          // device mirror of block_table
    std::vector<int> free_blocks;  // physical pages nobody holds
};

// ---- startup ----

DeviceBuffers allocateDeviceBuffers();
KVCacheState allocateKVCache();

// Writes the engine_config line the client needs (BATCH_SIZE is derived, so it cannot
// hardcode it) and reports how much VRAM the scratch buffers left behind.
void reportEngineConfig();

// ---- one step ----

// Takes prompts off the queue into every free slot. The returned items still have to be
// prefilled; their slots are already marked taken.
std::vector<PrefillBatchItem> admitQueuedRequests(std::deque<Request> &queue, SlotState &slots);

// The single place a sequence is retired. Emitting the final message, handing the slot back
// and returning the sequence's pages used to be open-coded in the decode loop, and the
// KV-exhaustion path in prefill had its own half-version that freed the slot without the
// pages. Routing prefill, decode and the eviction guard through one function means a slot
// can never be released without the client being told which request owned it, and pages can
// never be leaked with it. `error` is null for a normal finish.
void finishSequence(int slot, int request_id, const char *error, SlotState &slots, KVCacheState &kv);

// Retires any running sequence the next decode step could not serve: one that has run past
// its block table, and - newest first - whichever ones have to start a new page while the
// pool cannot cover all of them. decodeStep calls free_blocks.back() with no fallback and
// indexes block_table at seq_len / BLOCK_SIZE with no bound check, both of which hold only
// as long as the page budget does; checking here, before any of the forward pass has run,
// means such a sequence is retired cleanly instead of taking the whole batch down with it.
void enforcePageBudget(SlotState &slots, KVCacheState &kv);

// Prefills every admitted prompt in one pass over the model instead of one pass per prompt.
//
// The batch is packed into a single flat run of tokens, so each layer issues exactly one
// GEMM per projection with n = total tokens. The old per-prompt loop streamed all 2.30 GiB
// of weights once per prompt (16 x 2.30 GiB = 36.8 GiB, ~192 ms of pure traffic at the
// GTX 1650's 192 GB/s); packed, they are streamed once for the whole batch.
//
// Three other costs that dominated the sequential path are gone as well:
//   * the lm_head GEMM ran over every prompt token and discarded all but the last row
//     (269 GFLOP per 512-token prompt, ~100 ms each on 2.7 TFLOP/s) - it now runs on the
//     gathered last token of each prompt only;
//   * K/V were written into the paged cache with two cudaMemcpy calls per page per layer
//     (~16k calls for a full batch) - one scatter kernel per layer now does all of it;
//   * attention materialised a (NUM_Q_HEADS, L, L) score buffer and walked it three times
//     - prefillAttention is a flash-style single pass that never writes it out.
//
// Prompts whose combined length exceeds MAX_BATCH_TOKENS are split across several passes.
void prefillBatch(std::vector<PrefillBatchItem> &items,
                  DeviceBuffers &buf,
                  std::vector<__nv_bfloat16> &embed_proj_cpu,
                  const Weights &weights,
                  cublasHandle_t cublas_handle,
                  SlotState &slots,
                  KVCacheState &kv,
                  std::deque<Request> &queue);

// Advances every running sequence by one token: one forward pass over the model with one
// row per active slot. Returns how many slots were active, 0 meaning there was nothing to
// decode and the caller should back off instead of spinning.
int decodeStep(DeviceBuffers &buf,
               std::vector<__nv_bfloat16> &embed_proj_cpu,
               const Weights &weights,
               cublasHandle_t cublas_handle,
               SlotState &slots,
               KVCacheState &kv);
