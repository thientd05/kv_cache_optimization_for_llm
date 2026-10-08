#pragma once

// Config for the *contiguous reservation* build, i.e. this build being the Orca-style baseline.
//
// Two parts, separated by a marked barrier: first the block both builds share verbatim, then what
// is specific to contiguous reservation. Nothing mechanism-specific may go above the END marker.

#include <algorithm>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ======================== BEGIN SHARED CONFIG - DO NOT LET THIS DRIFT ========================
//
// Everything the contiguous-reservation build (base/) and the paged build (paged_attention/) must
// agree on. This block is kept BYTE-IDENTICAL in base/src/config.h and paged_attention/src/config.h.
// Change one, change the other, then check:
//
//     ./check_shared_config.sh
//
// They did drift once: base claimed its 768 MiB KV pool was "exactly the 24576 pages the paged
// build's pool holds" while the paged build actually had 1000 MiB and 32000 pages - a 30%
// difference in the one resource the whole comparison is about, hidden behind a stale comment.
//
// The rule for what belongs in here: a constant goes above the END marker if the two builds must
// use the same value for the comparison to mean anything. That covers the model, the hardware, the
// generation limits, the scheduler's knobs and the kernel launch geometry. It does NOT cover
// anything that is a *consequence* of how a build manages its KV cache - page sizes, block tables,
// reservation policies, allocation granularity. Those differences are the thing being measured,
// and flattening them would erase the result. See A6 in .claude/plan-1-reproduce-conditions.md.

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
// 1024 + 1024 = 2048 on purpose. The PagedAttention paper serves OPT, whose maximum sequence
// length is 2048, and its chatbot experiment (§6.5) cuts prompts to the last 1024 tokens and lets
// the model generate at most 1024. Matching those numbers matters for two reasons beyond tidiness:
// Orca (Max) is defined as reserving "up to the maximum sequence length of the model, i.e. 2048
// tokens", so the baseline's worst case is only right if that length really is 2048; and ShareGPT
// prompts stop being filtered out wholesale.
//
// This costs no extra VRAM. Every per-token prefill buffer scales with MAX_BATCH_TOKENS, not with
// MAX_PROMPT_LEN, and the RoPE tables already cover MAX_SEQ_LEN positions.
constexpr int MAX_NEW_TOKENS_GENERATED = 1024;
constexpr int MAX_PROMPT_LEN = 1024;
constexpr int MAX_SEQ_LEN = 2048;
constexpr float REPETITION_PENALTY = 1.15f;

// Worst-case lifetime footprint of one sequence, in tokens of context. Both builds need it: the
// baseline reserves against it, the paged build sizes its block table by it.
constexpr int MAX_TOKENS_PER_SEQUENCE = MAX_PROMPT_LEN + MAX_NEW_TOKENS_GENERATED;
static_assert(MAX_TOKENS_PER_SEQUENCE <= MAX_SEQ_LEN,
              "the RoPE tables only cover MAX_SEQ_LEN positions");

// ---- how many sequences may decode at once ----
// A scheduler knob, deliberately the same number in both builds and deliberately generous. The
// paper lets memory decide how many sequences run concurrently rather than imposing an artificial
// cap, so this is set high enough that it should never bind; emitStepTelemetry reports running_max
// so a run that does hit it is visible rather than silent. A slot costs ~380 KiB of VRAM (a
// VOCAB_SIZE row of logits plus its repetition-penalty mask), and both builds pay that same bill.
#ifdef ABLATION_MAX_SEQUENCES
// The §7.2 block-size ablation runs at a smaller, fixed slot count for every block size: the
// block table scales as MAX_SEQUENCES * N_LAYERS * (MAX_TOKENS_PER_SEQUENCE / BLOCK_SIZE), so
// at BLOCK_SIZE=1 and 384 slots it is 25 MiB and does not fit. Holding it fixed keeps the
// ablation to one moving variable. Never set for the main sweep.
constexpr int MAX_SEQUENCES = ABLATION_MAX_SEQUENCES;
#else
constexpr int MAX_SEQUENCES = 384;
#endif
static_assert(MAX_SEQUENCES <= MAX_THREADS_PER_BLOCK,
              "markSampledTokensKernel runs <<<1, BATCH_SIZE>>>, so a slot must fit in one block");

// ---- KV cache budget ----
// Deliberately NOT a size. In the paper the KV cache is not a dial, it is the leftover: same GPU,
// same model, same activations, so whatever VRAM is free once the weights and the scratch buffers
// are allocated becomes the cache, equally on both sides. Each build therefore allocates its
// scratch first and then claims (free - this margin). The margin is the headroom left for the
// display server and for cuBLAS workspaces that appear after startup.
constexpr size_t KV_SAFETY_MARGIN_BYTES = 128ULL * 1024 * 1024;

// ---- packed prefill ----
// Prefill runs every queued prompt in one packed pass instead of one pass per prompt.
// MAX_BATCH_TOKENS caps how many prompt tokens a single pass may carry; a batch that does
// not fit is split into several passes.
//
// Every per-token prefill buffer scales with it: hidden_state + rms_norms + buf_2048_1 +
// buf_2048_2 (4 x 2048 bf16 = 16 KiB) + gate + up (2 x 8192 bf16 = 32 KiB) + k/v temp
// (2 x 512 bf16 = 2 KiB) + 4 index arrays = ~50 KiB per token, so 3072 tokens is ~150 MiB
// against the ~27 MiB the old 512-token buffers used. Per MACHINE.md that fits only
// because two allocations went away: the logits buffer now holds BATCH_SIZE rows instead
// of MAX_PROMPT_LEN rows (125.25 MiB -> 7.58 MiB, see prefillBatch) and the materialised
// (NUM_Q_HEADS, L, L) attention score buffer (16 MiB) is gone entirely. Raising this eats
// into the VRAM that is left - and now also directly out of the KV pool, since the pool is
// whatever is left over - so check the "Scratch allocated" line the engine prints at startup.
constexpr int MAX_BATCH_TOKENS = 3072;

// ---- prefillAttention tiling ----
// prefillAttentionKernel works on tiles of consecutive query tokens from the same prompt, so
// the host has to cut the packed batch into tiles of at most PREFILL_ATTN_QUERIES_PER_TILE
// tokens. One tile shares a single K/V staging tile in shared memory, which is where its
// bandwidth saving comes from.
constexpr int PREFILL_ATTN_QUERIES_PER_TILE = 8;
constexpr int PREFILL_ATTN_KEYS_PER_TILE = 32;
constexpr int PREFILL_ATTN_LANES = WARP_SIZE;
constexpr int PREFILL_ATTN_THREADS = PREFILL_ATTN_QUERIES_PER_TILE * PREFILL_ATTN_LANES;

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

// ---- serving telemetry ----
// One JSON line every this many decode steps: how many requests are batched, how much of
// the cache is reserved, and how much of what is reserved holds a real token. The gap
// between those last two is the waste the PagedAttention paper is about (its Figure 2).
constexpr int STEP_TELEMETRY_EVERY = 8;

// ---- misc ----
constexpr int B_TO_MB = 1024 * 1024;

// =========================== END SHARED CONFIG - mechanism-specific below ===========================

// ---- contiguous KV cache: the Orca-style reservation baseline ----
// A sequence's K and V live in one contiguous run of token slots per layer, and token
// `position` always sits at `position` inside that run. Addressing is pure arithmetic -
// no page table, no pool, no indirection - which is the point of the baseline.
//
// The price is that the run cannot grow. The scheduler has to decide how long it will be
// *at admission*, before a single token has been generated, and nothing can be handed back
// until the sequence finishes. That is Orca's KV memory manager, and it is what the
// PagedAttention paper compares against. Three reservation policies, selected at runtime
// with the ORCA_POLICY environment variable, bracket what a contiguous engine can do. The
// wording below is the paper's own (§6.1):
//
//   max     "always reserves the space up to the maximum sequence length of the model, i.e.
//           2048 tokens". The only policy that needs no information about the request, and
//           the one a real system is stuck with.
//   pow2    "over-reserves the space for outputs by at most 2x. For example, if the true
//           output length is 25, it reserves 32 positions for outputs." Note that the
//           rounding is applied to the *output* length and the prompt is then added on top,
//           not to the total.
//   oracle  "the system has the knowledge of the lengths of the outputs that will be
//           actually generated". Not implementable, and it is here as the lower bound on
//           what *any* reservation scheme could waste.
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

// 16 layers * 2 (K and V) * 512 * 2 B = 32 KiB of cache per token of context
constexpr size_t KV_BYTES_PER_TOKEN = (size_t)N_LAYERS * 2 * KV_DIM * sizeof(__nv_bfloat16);


// How many tokens of context the pool holds. Not a constant: the pool is whatever VRAM is left
// once the weights and every scratch buffer are allocated, which is how the paper's setup works -
// same GPU, same model, so the leftover is the same on both sides and neither build gets to pick.
// See allocateKVCache(). A token slot is the 32 KiB one token of context costs across all layers,
// the same accounting unit the paged build's pages add up to.
extern int g_kv_pool_tokens;

// K of (layer) for a sequence whose run starts at token `base_token` and is `cap_tokens`
// long starts at this element offset into the cache; V starts cap_tokens * KV_DIM further
// on. This is what replaces the paged build's block table walk. The run is laid out
// layer-major, K then V within a layer - the same order as inside a page over there.
__host__ __device__ inline size_t kvKOffset(int base_token, int cap_tokens, int layer)
{
    return ((size_t)base_token * N_LAYERS * 2 + (size_t)layer * 2 * cap_tokens) * KV_DIM;
}

// ---- how many sequences may decode at once ----
// Not derived from the cache, and deliberately the same number the paged build uses:
// concurrency has to be decided by the reservation policy at runtime, not by a compile-time
// worst case, or the comparison answers the wrong question. Under ORCA_POLICY=max the
// allocator will in fact only fit pool / 2048 sequences; that is the result, not an input.
constexpr int BATCH_SIZE = MAX_SEQUENCES;
static_assert(BATCH_SIZE >= 1, "at least one decode slot");

constexpr int MAX_BUFFER_SIZE = std::max(MAX_BATCH_TOKENS, BATCH_SIZE);

// query tiles for prefillAttention: sum of ceil(prompt_len / tile) over the pass, which is
// at most one partial tile per prompt on top of the full ones
constexpr int MAX_PREFILL_ATTN_TILES = MAX_BATCH_TOKENS / PREFILL_ATTN_QUERIES_PER_TILE + BATCH_SIZE;

// The HF model snapshot lives in models/ at the repo root, and the engine is launched from
// base/, hence the "../". Override with the MODEL_DIR env var or by passing the
// directory as the first program argument.
constexpr const char *DEFAULT_MODEL_DIR = "../models/Llama-3.2-1B-Instruct";
