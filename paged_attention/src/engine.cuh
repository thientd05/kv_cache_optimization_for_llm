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
    int max_tokens;     // output budget still owed to this request
    int resumed_tokens; // non-zero only for a sequence being recomputed after preemption
    int seq_index;      // which answer of a parallel-sampling request this is; 0 otherwise
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
    // ---- sampling ----
    // The argmax over the logits runs on the device, so all that crosses PCIe after the
    // lm_head is one token id per row instead of a VOCAB_SIZE row of logits.
    int *logit_row_slots;     // [BATCH_SIZE] slot owning each logits row (decode reuses active_slots)
    float *argmax_values;     // [BATCH_SIZE, ARGMAX_CHUNKS_PER_ROW] first-pass partials
    int *argmax_indices;      // [BATCH_SIZE, ARGMAX_CHUNKS_PER_ROW]
    int *sampled_tokens;      // [BATCH_SIZE] the only thing the host reads back per step
    // Random sampling (§6.3). Allocated unconditionally - about 4 KiB at BATCH_SIZE 384 -
    // because making them conditional would make the greedy and sampled builds disagree on
    // how much VRAM is left, and the KV pool is exactly "whatever is left".
    float *row_max;           // [BATCH_SIZE] stable-softmax shift, one per logits row
    float *exp_sums;          // [BATCH_SIZE, ARGMAX_CHUNKS_PER_ROW] second-pass partials
    int *rng_keys;            // [BATCH_SIZE] which random stream this row draws from
    int *rng_positions;       // [BATCH_SIZE] how far along that stream it is
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
    // Which of that request's answers this slot is producing (§6.3). Always 0 outside parallel
    // sampling. It is carried separately from the request id because several slots legitimately
    // share an id, and every message the engine emits has to say which one it is about.
    std::vector<int> slot_seq_index;
    std::vector<int> remaining_budget;                // output tokens the request may still get
    // The prompt as it arrived, kept for the whole sequence's life. A preempted sequence is
    // recomputed from prompt + what it had already generated, so the prompt has to survive the
    // prefill that consumed it. The baseline never needs this: a reservation cannot be taken away.
    std::vector<std::vector<int>> prompt_tokens;
    // True while this slot holds a sequence that was preempted and readmitted, so finishSequence
    // knows when the last one has drained and admission may reopen. See kv.preemption_pending.
    std::vector<bool> was_preempted;
    std::vector<std::chrono::high_resolution_clock::time_point> decode_start;

    SlotState();
};

// The paged KV cache and the page table that maps (slot, layer, logical page) to a physical
// page. The table is kept on both sides: the host allocates pages into `block_table` and
// uploads it to `block_table_gpu` for the attention kernels to walk.
// §7.3. Two ways to recover a sequence whose pages were taken away: rebuild its KV cache with
// one prefill pass (Recompute), or keep the pages' contents in host RAM and copy them back when
// the pool has room (Swap). Recompute is the default, so every result produced before the swap
// path existed still means what it meant.
enum class PreemptionPolicy
{
    Recompute,
    Swap
};
extern PreemptionPolicy g_preemption_policy;

// A sequence whose pages live in host RAM right now. It is NOT in the request queue: unlike a
// recomputed sequence it never goes back through prefill, so it has to carry everything a decode
// slot holds - the tokens it has emitted, how much budget is left, how long its context is - and
// be restored straight into a slot by swapInSequences().
struct SwappedSequence
{
    int request_id;
    int seq_index;
    int remaining_budget;
    int context_len;   // current_prompt_len at the moment it was swapped out
    int last_token;    // the token the next decode step feeds back in
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    // Host page holding each (layer, logical page) of this sequence, -1 where it was unmapped.
    // Same layout as one slot's rows of block_table, so the two map onto each other directly.
    std::vector<int> host_pages;
};

struct KVCacheState
{
    __nv_bfloat16 *cache;          // NUM_BLOCKS pages of BLOCK_BYTES
    std::vector<int> block_table;  // [MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ], -1 = unmapped
    int *block_table_gpu;          // device mirror of block_table
    std::vector<int> free_blocks;  // physical pages nobody holds
    // How many block-table entries point at each physical page (§4.4). One for an ordinary
    // page; more once the answers to a parallel-sampling request share their prompt's pages.
    // A page only goes back on free_blocks when this reaches zero, which is the single
    // invariant the whole sharing scheme rests on - release a shared page early and the
    // sibling still reading it gets another sequence's tokens.
    std::vector<int> block_refcount;
    // Pages that exist once but are pointed at more than once, i.e. the saving. Fig 15.
    long long shared_page_refs;
    long long preemptions;         // sequences sent back for recomputation
    // True from the moment a sequence is preempted until every preempted sequence has finished.
    // §4.5: "Once it preempts a sequence and evicts its blocks, vLLM stops accepting new requests
    // until all preempted sequences are completed." Without it, the sequence that was just
    // readmitted is the newest one again and gets preempted straight back out - thrashing, and the
    // latency numbers would measure that loop instead of paging.
    bool preemption_pending;
    int preempted_in_flight;       // how many are still working their way back out
    // ---- swap-to-host state, only touched under PreemptionPolicy::Swap ----
    __nv_bfloat16 *swap_host;          // pinned host pages, SWAP_HOST_BYTES / BLOCK_BYTES of them
    std::vector<int> swap_free_pages;  // host pages nobody holds
    std::deque<SwappedSequence> swapped; // FIFO: oldest swapped-out sequence comes back first
    long long swap_outs;               // sequences pushed to host
    long long swap_ins;                // sequences pulled back
    // Times the swap policy could not swap and recomputed instead. Reported rather than hidden:
    // a run where this is ~100% is measuring recompute under a different name, and reading it as
    // "swap is slow" would be wrong. See the gate in plan 5.
    long long swap_fallbacks;
    long long swap_bytes;              // total moved in both directions
    double swap_ms;                    // wall time spent inside the copies
    // ---- §6.4 shared prefix ----
    // "In vLLM, this can be conveniently achieved by reserving a set of physical blocks for a
    // set of predefined shared prefixes by the LLM service provider, as how OS handles shared
    // library across processes." The registry is those reserved pages: one row of
    // (layer, logical block) covering the whole-page part of the prefix, pinned for the run.
    std::vector<int> prefix_pages;   // [N_LAYERS * prefix_blocks], -1 until established
    std::vector<int> prefix_tokens;  // the prefix itself, so a prompt can be checked against it
    long long prefix_hits;           // prompts that mapped the registry instead of paying for it
    long long prefix_pinned_pages;   // size of the registry, i.e. the memory it costs once
};

// How many whole pages of the configured shared prefix can be shared, and whether a prompt
// actually carries that prefix. Zero unless SHARED_PREFIX_LEN is set, which is every experiment
// outside §6.4.
extern int g_shared_prefix_len;
int sharedPrefixBlocks();

// Takes a running sequence's pages back and returns it to the front of the queue to be recomputed
// from prompt + the tokens it had already generated. §4.5: "we simply recompute the KV cache when
// the preempted sequences are rescheduled... their KV cache at all positions can be generated in
// one prompt phase iteration." Front of the queue because scheduling is FCFS and this request
// arrived before anything still waiting.
void preemptSequence(int slot, SlotState &slots, KVCacheState &kv, std::deque<Request> &queue);

// True while the page pool adds up: free pages have refcount 0, held pages have refcount > 0,
// and the two sets together are every page exactly once. See the call site in main.cu.
bool kvPageAccountingHolds(const KVCacheState &kv);

// Puts swapped-out sequences back into free slots, pages and all, without a prefill pass. Called
// at the top of the step loop, before new prompts are admitted: a sequence already in flight has
// arrived earlier than anything still queued, and FCFS says it goes first.
void swapInSequences(SlotState &slots, KVCacheState &kv, DeviceBuffers &buf);

// How many requests are batched right now, how much of the pool is mapped, and how much of what
// is mapped holds a real token. One JSON line every STEP_TELEMETRY_EVERY decode steps; the
// benchmark turns them into the paper's "batched requests over time" and "KV cache utilisation"
// curves. Same field names as the baseline build so one parser reads both.
void emitStepTelemetry(const SlotState &slots, const KVCacheState &kv, const std::deque<Request> &queue);

// Set from the IGNORE_EOS environment variable; see engine.cu.
extern bool g_ignore_eos;
extern float g_temperature;
extern unsigned int g_sampling_seed;

// One sampling entry point for both prefill and decode. rng_keys/rng_positions are read only
// when g_temperature > 0; under greedy they are ignored and may hold anything.
void sampleTokens(DeviceBuffers &buf, int num_rows, const unsigned char *penalty_mask,
                  const int *row_slot_ids, const std::vector<int> &rng_keys,
                  const std::vector<int> &rng_positions);

// ---- startup ----

DeviceBuffers allocateDeviceBuffers();
KVCacheState allocateKVCache();

// Writes the engine_config line the client needs (BATCH_SIZE is derived, so it cannot
// hardcode it) and reports how much VRAM the scratch buffers left behind.
void reportEngineConfig();

// ---- one step ----

// Takes prompts off the queue into every free slot. The returned items still have to be
// prefilled; their slots are already marked taken.
// Admission only has to cover the *prompt*: pages for the generated tokens are taken one at a
// time as decode produces them, and the pool is deliberately over-subscribed. That is the whole
// point of paging, and it is the one place this build differs from the baseline in kind rather
// than in addressing - over there a prompt cannot be admitted unless its entire worst-case
// lifetime reservation fits right now.
std::vector<PrefillBatchItem> admitQueuedRequests(std::deque<Request> &queue, SlotState &slots,
                                                  KVCacheState &kv);

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
void enforcePageBudget(SlotState &slots, KVCacheState &kv, std::deque<Request> &queue);

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
                  const Weights &weights,
                  cublasHandle_t cublas_handle,
                  SlotState &slots,
                  KVCacheState &kv,
                  std::deque<Request> &queue);

// Advances every running sequence by one token: one forward pass over the model with one
// row per active slot. Returns how many slots were active, 0 meaning there was nothing to
// decode and the caller should back off instead of spinning.
int decodeStep(DeviceBuffers &buf,
               const Weights &weights,
               cublasHandle_t cublas_handle,
               SlotState &slots,
               KVCacheState &kv);
