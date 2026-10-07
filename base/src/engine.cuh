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
    int max_tokens;         // output budget still owed to this request
    int resumed_tokens;     // unused here: a reservation never has to preempt anything
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
    // K/V of the pass before they are scattered into the KV cache; decode reuses them
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
    // ---- sampling ----
    // The argmax over the logits runs on the device, so all that crosses PCIe after the
    // lm_head is one token id per row instead of a VOCAB_SIZE row of logits.
    int *logit_row_slots;     // [BATCH_SIZE] slot owning each logits row (decode reuses active_slots)
    float *argmax_values;     // [BATCH_SIZE, ARGMAX_CHUNKS_PER_ROW] first-pass partials
    int *argmax_indices;      // [BATCH_SIZE, ARGMAX_CHUNKS_PER_ROW]
    int *sampled_tokens;      // [BATCH_SIZE] the only thing the host reads back per step
    // Repetition penalty state: one byte per (slot, token), 1 once the sequence has emitted
    // that token. Replaces re-deriving the set from generated_tokens on the host every step.
    unsigned char *penalty_mask; // [BATCH_SIZE, VOCAB_SIZE]
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
    std::vector<int> remaining_budget;                // output tokens the request may still get
    std::vector<std::chrono::high_resolution_clock::time_point> decode_start;

    SlotState();
};

// One free run of the contiguous pool: `length` token slots starting at `start`, where a
// "token slot" is the 32 KiB that one token of context costs across all 16 layers.
struct KVFreeRun
{
    int start;
    int length;
};

// The contiguous KV cache and the reservation allocator over it. A sequence gets one run of
// consecutive token slots at admission, sized by the reservation policy, and keeps it until
// it finishes. The run cannot grow, so the size has to be right the first time - that
// constraint is the baseline, and everything the paged build does differently follows from
// not having it.
//
// The free list is kept sorted and coalesced, so a slot's run is as contiguous as the pool
// allows. When a reservation fails even though the free *total* would cover it, the pool is
// externally fragmented: another cost of contiguity that paging does not have, and one the
// engine counts rather than hides.
struct KVCacheState
{
    __nv_bfloat16 *cache;             // KV_POOL_TOKENS token slots' worth of K and V
    std::vector<KVFreeRun> free_runs; // sorted by start, coalesced
    std::vector<int> slot_base;       // [MAX_SEQUENCES] first token slot of the run, -1 if none
    std::vector<int> slot_cap;        // [MAX_SEQUENCES] how many token slots long it is
    int *slot_base_gpu;               // device mirrors, read by the attention and scatter kernels
    int *slot_cap_gpu;
    int reserved_tokens;              // sum of slot_cap over live slots
    long long fragmentation_failures; // reservations the free total could cover but no single run could
};

// Reservation size for a request under the active policy, rounded up to
// KV_ALLOC_GRANULARITY. `final_len` is prompt + the request's output budget.
int reservationTokens(int prompt_len, int max_tokens);

// First-fit allocation of `tokens` consecutive token slots to `slot`. Returns false if the
// pool cannot place it, which is what caps concurrency in this build.
bool kvReserve(KVCacheState &kv, int slot, int tokens);

// Returns a slot's run to the pool and coalesces it with its neighbours.
void kvRelease(KVCacheState &kv, int slot);

// Pushes slot_base / slot_cap to the device. Cheap (160 ints each) and done once per pass.
void kvSyncSlotTables(const KVCacheState &kv);

// Set from the IGNORE_EOS environment variable; see engine.cu.
extern bool g_ignore_eos;

// ---- startup ----

DeviceBuffers allocateDeviceBuffers();
KVCacheState allocateKVCache();

// Writes the engine_config line the client needs (BATCH_SIZE is derived, so it cannot
// hardcode it) and reports how much VRAM the scratch buffers left behind.
void reportEngineConfig();

// ---- one step ----

// Takes prompts off the queue into free slots; the returned items still have to be
// prefilled, and their slots are already marked taken.
//
// Admission is where the reservation is made: a prompt comes off the queue only if the
// allocator can place its whole worst-case run right now. This is the single place the
// baseline differs from the paged build in kind rather than in addressing - over there a
// prompt only has to pay for its prompt, and the rest is allocated a page at a time as it
// is generated.
std::vector<PrefillBatchItem> admitQueuedRequests(std::deque<Request> &queue, SlotState &slots,
                                                  KVCacheState &kv);

// The single place a sequence is retired. Emitting the final message and handing the slot
// back used to be open-coded in the decode loop. Routing prefill, decode and the capacity
// guard through one function means a slot can never be released without the client being
// told which request owned it. `error` is null for a normal finish. The sequence's whole
// reservation is released here in one piece - cheap, where the paged build has to walk the
// block table and push every page back onto the pool individually. The expensive half is
// that it could not have been released any earlier.
void finishSequence(int slot, int request_id, const char *error, SlotState &slots, KVCacheState &kv);

// Retires any running sequence the next decode step could not serve: one that has filled
// its reservation. decodeStep writes K/V at index seq_len with no bound check, which holds
// only as long as the sequence fits inside its run; checking here, before any of the
// forward pass has run, means such a sequence is retired cleanly instead of scribbling
// over the next sequence's region.
//
// There is no eviction pass here, and that is the structural difference from the paged
// build: a reservation cannot run out from under a sequence that was already admitted, so
// nothing ever has to be preempted. The paged build pays for over-subscription with
// preemptSequence; this one pays for the guarantee at admission, in concurrency.
void enforceSeqCapacity(SlotState &slots, KVCacheState &kv);

// How many requests are batched right now, how much of the pool is reserved, and how much
// of what is reserved holds a real token. One JSON line every STEP_TELEMETRY_EVERY decode
// steps; the benchmark turns them into the paper's "batched requests over time" and "KV
// cache utilisation" curves. The gap between reserved and live is this build's waste.
void emitStepTelemetry(const SlotState &slots, const KVCacheState &kv, const std::deque<Request> &queue);

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
//   * K/V were written into the cache with two cudaMemcpy calls per prompt per layer -
//     one scatter kernel per layer now does all of it;
//   * attention materialised a (NUM_Q_HEADS, L, L) score buffer and walked it three times
//     - prefillAttention is a flash-style single pass that never writes it out.
//
// Prompts whose combined length exceeds MAX_BATCH_TOKENS are split across several passes.
void prefillBatch(std::vector<PrefillBatchItem> &items,
                  DeviceBuffers &buf,
                  const Weights &weights,
                  cublasHandle_t cublas_handle,
                  SlotState &slots,
                  KVCacheState &kv);

// Advances every running sequence by one token: one forward pass over the model with one
// row per active slot. Returns how many slots were active, 0 meaning there was nothing to
// decode and the caller should back off instead of spinning.
int decodeStep(DeviceBuffers &buf,
               const Weights &weights,
               cublasHandle_t cublas_handle,
               SlotState &slots,
               KVCacheState &kv);
