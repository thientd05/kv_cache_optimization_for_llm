#pragma once

// Every tunable and every hardcoded model dimension of the engine lives here, so main and
// the kernels read the same numbers instead of each keeping their own copy of them.

#include <algorithm>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---- hardware ----
// GTX 1650 / most consumer NVIDIA parts: a block may not exceed 1024 threads, which is why
// every kernel over an EMBEDDING_LENGTH row has each thread handle two elements.
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
// full-warp mask for the __shfl_*_sync reductions
#define WARP_FULL_MASK 0xffffffff

// ---- llama 3.2 1B-Instruct ----
// TODO: hardcoded for llama 3.2 1B; read them off config.json instead
constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr float SQRT_HEAD_DIM = 8;
constexpr int NUM_Q_HEADS = 32;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int VOCAB_SIZE = 128256;
// double on purpose: the norm is accumulated in float but the epsilon add and the sqrt
// run in double, which is what rmsNormKernel has always done
constexpr double RMS_NORM_EPS = 1.0e-5;

// threads per block for the kernels that walk a whole EMBEDDING_LENGTH row, two elements each
constexpr int EMBED_HALF = EMBEDDING_LENGTH / 2;
static_assert(EMBED_HALF <= MAX_THREADS_PER_BLOCK, "an embedding row must fit two per thread");

// llama 3 RoPE scaling, from the model's config.json rope_scaling section
constexpr float ROPE_THETA = 500000.0f;
constexpr float ROPE_SCALING_FACTOR = 32.0f;
constexpr float ROPE_LOW_FREQ_FACTOR = 1.0f;
constexpr float ROPE_HIGH_FREQ_FACTOR = 4.0f;
constexpr int ROPE_ORIGINAL_MAX_LEN = 8192;

constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>

// ---- generation limits ----
constexpr int MAX_NEW_TOKENS_GENERATED = 512; // TODO: parameterize it with program arguments
constexpr int MAX_SEQ_LEN = 2048;             // TODO: make it tunable
constexpr int MAX_PROMPT_LEN = 512;           // TODO: arbitrary, tunable
constexpr float REPETITION_PENALTY = 1.15f;

// ---- GPU argmax ----
// A step's logits are a (rows, VOCAB_SIZE) bf16 block and all the host needs out of them is
// one token id per row. Bringing the whole block back (BATCH_SIZE * 128256 bf16 = 7.58 MiB
// per step over PCIe) and scanning it on one core cost far more than the reduction itself,
// so the argmax runs on the device in two passes: ARGMAX_CHUNKS_PER_ROW blocks per row each
// reduce a contiguous slice of the vocabulary, then one block per row reduces those
// partials. Only the resulting ints come back, 4 bytes per row instead of 256 KiB.
//
// The chunk count is what keeps the card busy: one block per row would leave a 14-SM GPU
// with BATCH_SIZE blocks of work, so each row is cut into this many independent pieces.
constexpr int ARGMAX_CHUNKS_PER_ROW = 128;
constexpr int ARGMAX_BLOCK_THREADS = 256;
constexpr int ARGMAX_CHUNK_TOKENS = (VOCAB_SIZE + ARGMAX_CHUNKS_PER_ROW - 1) / ARGMAX_CHUNKS_PER_ROW;
// the second pass is a tree reduction over a single block of ARGMAX_CHUNKS_PER_ROW threads
static_assert(ARGMAX_CHUNKS_PER_ROW <= MAX_THREADS_PER_BLOCK, "the finalize pass needs one thread per chunk");
static_assert((ARGMAX_CHUNKS_PER_ROW & (ARGMAX_CHUNKS_PER_ROW - 1)) == 0, "the finalize pass halves the thread count, so it must be a power of two");
static_assert(ARGMAX_BLOCK_THREADS % WARP_SIZE == 0, "the first pass reduces warp by warp");

// ---- contiguous KV cache: the Orca-style reservation baseline ----
// A sequence's K and V live in one contiguous run of token slots per layer, and token
// `position` always sits at `position` inside that run. Addressing is pure arithmetic -
// no page table, no pool, no indirection - which is the point of the baseline.
//
// The price is that the run cannot grow. The scheduler has to decide how long it will be
// *at admission*, before a single token has been generated, and nothing can be handed back
// until the sequence finishes. That is Orca's KV memory manager, and it is what the
// PagedAttention paper compares against. Three reservation policies, selected at runtime
// with the ORCA_POLICY environment variable, bracket what a contiguous engine can do:
//
//   max     reserve MAX_TOKENS_PER_SEQUENCE for everybody. The only policy that needs no
//           information about the request, and the one a real system is stuck with.
//   pow2    reserve the next power of two above the request's actual final length.
//   oracle  reserve exactly the request's actual final length. Not implementable - it
//           needs to know the output length before generating it - and it is here as the
//           lower bound on what *any* reservation scheme could waste.
//
// All three still hold the full reservation for the sequence's whole lifetime, which is
// the part even the oracle cannot fix and the part paging removes.
enum class ReservePolicy
{
    Max,
    Pow2,
    Oracle
};
extern ReservePolicy g_reserve_policy;
extern const char *reservePolicyName();

constexpr int MAX_TOKENS_PER_SEQUENCE = MAX_PROMPT_LEN + MAX_NEW_TOKENS_GENERATED; // 1024
static_assert(MAX_TOKENS_PER_SEQUENCE <= MAX_SEQ_LEN, "the RoPE tables only cover MAX_SEQ_LEN positions");

// the same budget the paged build gives its page pool, so neither side gets more KV VRAM
constexpr size_t KV_CACHE_SIZE_BYTES = 768ULL * 1024 * 1024;
// 16 layers * 2 (K and V) * 512 * 2 B = 32 KiB of cache per token of context
constexpr size_t KV_BYTES_PER_TOKEN = (size_t)N_LAYERS * 2 * KV_DIM * sizeof(__nv_bfloat16);
// 768 MiB / 32 KiB = 24576 token slots, exactly the 24576 pages the paged build's pool
// holds (a page is 16 tokens of one layer, so 16 layers x 16 tokens = one token slot).
// Same bytes, same accounting unit.
constexpr int KV_POOL_TOKENS = (int)(KV_CACHE_SIZE_BYTES / KV_BYTES_PER_TOKEN);
// Reservations are rounded up to this, matching the paged build's BLOCK_SIZE, so that
// neither side gets an allocation-granularity advantage over the other.
constexpr int KV_ALLOC_GRANULARITY = 16;

// K of (layer) for a sequence whose run starts at token `base_token` and is `cap_tokens`
// long starts at this element offset into the cache; V starts cap_tokens * KV_DIM further
// on. This is what replaces the paged build's block table walk. The run is laid out
// layer-major, K then V within a layer - the same order as inside a page over there.
__host__ __device__ inline size_t kvKOffset(int base_token, int cap_tokens, int layer)
{
    return ((size_t)base_token * N_LAYERS * 2 + (size_t)layer * 2 * cap_tokens) * KV_DIM;
}

// ---- how many sequences may decode at once ----
// Not derived from the cache any more, and deliberately the same number the paged build
// uses: concurrency has to be decided by the reservation policy at runtime, not by a
// compile-time worst case, or the comparison answers the wrong question. Under ORCA_POLICY
// =max the allocator will in fact only ever fit 24576 / 1024 = 24 sequences; that is the
// result, not an input.
constexpr int MAX_SEQUENCES = 384;
constexpr int BATCH_SIZE = MAX_SEQUENCES;

static_assert(BATCH_SIZE >= 1, "KV cache too small for even one sequence");

// ---- serving telemetry ----
// One JSON line every this many decode steps: how many requests are batched, how much of
// the cache is reserved, and how much of what is reserved holds a real token. The gap
// between those last two is the waste the PagedAttention paper is about.
constexpr int STEP_TELEMETRY_EVERY = 8;

// ---- packed prefill ----
// Prefill runs every queued prompt in one packed pass instead of one pass per prompt.
// MAX_BATCH_TOKENS caps how many prompt tokens a single pass may carry; a batch that does
// not fit is split into several passes (BATCH_SIZE prompts of MAX_PROMPT_LEN would need 6).
//
// Every per-token prefill buffer scales with it: hidden_state + rms_norms + buf_2048_1 +
// buf_2048_2 (4 x 2048 bf16 = 16 KiB) + gate + up (2 x 8192 bf16 = 32 KiB) + k/v temp
// (2 x 512 bf16 = 2 KiB) + 4 index arrays = ~50 KiB per token, so 3072 tokens is ~150 MiB
// against the ~27 MiB the old 512-token buffers used. Per MACHINE.md that fits only
// because two allocations went away: the logits buffer now holds BATCH_SIZE rows instead
// of MAX_PROMPT_LEN rows (125.25 MiB -> 7.58 MiB, see prefillBatch) and the materialised
// (NUM_Q_HEADS, L, L) attention score buffer (16 MiB) is gone entirely. Raising this eats
// into the VRAM that is left, so check the "Scratch allocated" line the engine prints at
// startup before bumping it.
constexpr int MAX_BATCH_TOKENS = 3072;
constexpr int MAX_BUFFER_SIZE = std::max(MAX_BATCH_TOKENS, BATCH_SIZE);

// ---- prefillAttention tiling ----
// prefillAttentionKernel works on tiles of consecutive query tokens from the same prompt, so
// the host has to cut the packed batch into tiles of at most PREFILL_ATTN_QUERIES_PER_TILE
// tokens. One tile shares a single K/V staging tile in shared memory, which is where its
// bandwidth saving comes from.
constexpr int PREFILL_ATTN_QUERIES_PER_TILE = 8;
constexpr int PREFILL_ATTN_KEYS_PER_TILE = 32;
constexpr int PREFILL_ATTN_LANES = WARP_SIZE;
constexpr int PREFILL_ATTN_THREADS = PREFILL_ATTN_QUERIES_PER_TILE * PREFILL_ATTN_LANES;
// query tiles for prefillAttention: sum of ceil(prompt_len / tile) over the pass, which is
// at most one partial tile per prompt on top of the full ones
constexpr int MAX_PREFILL_ATTN_TILES = MAX_BATCH_TOKENS / PREFILL_ATTN_QUERIES_PER_TILE + BATCH_SIZE;

// ---- misc ----
constexpr int B_TO_MB = 1024 * 1024;

// The HF model snapshot lives in models/ at the repo root, and the engine is launched from
// base/, hence the "../". Override with the MODEL_DIR env var or by passing the
// directory as the first program argument.
constexpr const char *DEFAULT_MODEL_DIR = "../models/Llama-3.2-1B-Instruct";
