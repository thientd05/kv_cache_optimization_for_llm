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

// ---- paged KV cache ----
constexpr int BLOCK_SIZE = 16; // TODO: tunable as well, defines the size of a single page in pagedattn
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;                     // * 2 because K and V
constexpr size_t KV_CACHE_SIZE_BYTES = 1000ULL * 1024 * 1024; // ~0.98 GiB; safe margin on the GTX 1650's 3.63 GiB usable VRAM (2.30 GiB of that goes to the bf16 weights)
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;  // 2048 / 16 = 128
constexpr int NUM_BLOCKS = KV_CACHE_SIZE_BYTES / BLOCK_BYTES; // 1000*1024*1024/(16*512*2*2) = 32000

// ---- how many sequences may decode at once ----
// A sequence lives for at most MAX_PROMPT_LEN prompt tokens plus MAX_NEW_TOKENS_GENERATED
// decoded ones, so its worst-case KV footprint is a fixed number of pages and the KV cache
// is what really caps concurrency. Deriving BATCH_SIZE from it rather than hardcoding a
// number means the pool can always honour every sequence it admitted - `free_blocks` can
// never run dry - and the limit follows along when KV_CACHE_SIZE_BYTES, BLOCK_SIZE,
// N_LAYERS or either length cap changes.
//
// Decode slots are cheap in VRAM and not what the scratch buffers pay for: one slot costs
// ~265 KiB (a VOCAB_SIZE row of logits, an embedding row, its block table and a few index
// entries), i.e. about what 5 prefill tokens cost. Decode is bandwidth-bound - every step
// streams all 2.30 GiB of weights no matter how many slots are active - so slots are close
// to free throughput: measured 112 tok/s aggregate at 16 slots vs 226 tok/s at 31.
constexpr int MAX_TOKENS_PER_SEQUENCE = MAX_PROMPT_LEN + MAX_NEW_TOKENS_GENERATED;
constexpr int PAGES_PER_SEQUENCE = ((MAX_TOKENS_PER_SEQUENCE + BLOCK_SIZE - 1) / BLOCK_SIZE) * N_LAYERS;
constexpr int BATCH_SIZE = NUM_BLOCKS / PAGES_PER_SEQUENCE; // 32000 / 1024 = 31
constexpr int MAX_SEQUENCES = BATCH_SIZE;

static_assert(MAX_TOKENS_PER_SEQUENCE <= MAX_SEQ_LEN,
              "a sequence must fit in MAX_BLOCKS_PER_SEQ pages per layer");
static_assert(BATCH_SIZE >= 1, "KV cache too small for even one sequence");

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
// paged_attention/, hence the "../". Override with the MODEL_DIR env var or by passing the
// directory as the first program argument.
constexpr const char *DEFAULT_MODEL_DIR = "../models/Llama-3.2-1B-Instruct";
