#pragma once

// Kernel declarations. There are no host-side wrappers: main launches every kernel with
// <<<grid, block>>> directly, so the grid and block shape of each stage is visible at the
// call site next to the GEMMs it sits between. Launch shapes each kernel assumes are
// documented above it.

#include "config.h"

// RoPE cos/sin tables, [MAX_SEQ_LEN, HEAD_DIM] each, filled once by initRopeFrequencies.
// Device pointers held on the host, so the rope kernels take them as arguments.
extern float *d_cos_table;
extern float *d_sin_table;

void initRopeFrequencies();

// ---- shared by prefill and decode ----

// <<<num_tokens, EMBED_HALF>>>; (num_tokens, EMBEDDING_LENGTH) -> (num_tokens, EMBEDDING_LENGTH)
__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens);

// <<<num_tokens, EMBED_HALF>>>; in-place on input
__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds);

// <<<num_tokens, MAX_THREADS_PER_BLOCK>>>; in-place, overwriting a with SiLU(a) * b
__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b);

// ---- batched ("packed") prefill ----
//
// The whole batch of prompts is laid out as one flat run of tokens:
//   token t lives at row t of every [num_tokens, dim] buffer,
//   token_positions[t]  = position of t inside its own prompt (RoPE angle, causal limit),
//   token_seq_start[t]  = flat index of the first token of t's prompt,
//   token_slot_ids[t]   = batch slot that owns t (KV cache paging).
// Everything that is per-token (norms, projections, RoPE, MLP) therefore runs on the
// whole batch in a single launch, and the model weights are streamed once per layer
// instead of once per prompt.

// <<<num_input_tokens, EMBED_HALF>>>
// gpu_input_tokens - N tokens, gpu_input_embeds - (N, EMBEDDING_LENGTH),
// embed_tokens - (VOCAB_SIZE, EMBEDDING_LENGTH)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens);

// <<<num_tokens, proj_dim / 2>>>, proj_dim / 2 must not exceed MAX_THREADS_PER_BLOCK
__global__ void ropePackedKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim,
                                 const int *token_positions, const float *cos_table,
                                 const float *sin_table);

// <<<num_tokens, KV_DIM>>>
// Writes the whole batch's K/V of one layer into the KV cache in a single launch.
// Replaces the old per-prompt cudaMemcpy pair: a 512-token prompt used to cost two
// memcpys per layer, i.e. ~512 cudaMemcpy calls for a 16-prompt batch.
// The destination is pure arithmetic - kvKOffset(slot, layer) plus the token's position -
// where the paged build has to look the page up in block_table_gpu first.
__global__ void scatterKVPackedKernel(int layer, int num_tokens, const __nv_bfloat16 *k_src,
                                      const __nv_bfloat16 *v_src, __nv_bfloat16 *kv_cache,
                                      const int *kv_slot_base, const int *kv_slot_cap,
                                      const int *token_slot_ids, const int *token_positions);

// <<<dim3(num_tiles, NUM_Q_HEADS), PREFILL_ATTN_THREADS>>>
__global__ void prefillAttentionKernel(const __nv_bfloat16 *q_proj, const __nv_bfloat16 *k_packed,
                                       const __nv_bfloat16 *v_packed, const int *tile_token_begin,
                                       const int *tile_token_count, const int *token_seq_start,
                                       const int *token_positions, __nv_bfloat16 *output);

// <<<num_rows, EMBED_HALF>>>; (num_rows, EMBEDDING_LENGTH) gathered out of (num_tokens, EMBEDDING_LENGTH)
__global__ void gatherRowsKernel(const __nv_bfloat16 *src, __nv_bfloat16 *dst, const int *row_indices, int num_rows);

// ---- decode ----

// <<<num_tokens, EMBED_HALF>>>
__global__ void embeddingGatherDecodeKernel(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens);

// <<<1, proj_dim / 2>>>, proj_dim / 2 must not exceed MAX_THREADS_PER_BLOCK
__global__ void ropeDecodeKernel(__nv_bfloat16 *input, int position_in_sequence, int proj_dim,
                                 const float *cos_table, const float *sin_table);

// <<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>
__global__ void contiguousAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj,
                                          __nv_bfloat16 *kv_cache, const int *kv_slot_base,
                                          const int *kv_slot_cap, int *gpu_seq_lens,
                                          int *gpu_active_slots, __nv_bfloat16 *output);

// <<<num_rows, proj_dim / 2>>>, proj_dim / 2 must not exceed MAX_THREADS_PER_BLOCK
// The batched form of ropeDecodeKernel: one launch rotates every active slot's row instead
// of one launch per slot. At 31 slots the per-slot form cost 2 * 31 * N_LAYERS = ~1000
// launches a step, which is invisible; at 160 slots it is ~5000 and it stops being
// invisible, so the shape of the comparison must not depend on it.
__global__ void ropeDecodeBatchKernel(__nv_bfloat16 *input, int num_rows, int proj_dim,
                                      const int *positions, const float *cos_table,
                                      const float *sin_table);

// <<<num_rows, KV_DIM>>>
// Writes one decoded token's K and V per active slot into the cache in a single launch,
// replacing the 2 * num_rows cudaMemcpy pair the decode loop used to issue per layer.
// Pages must already be reserved in block_table_gpu by the host.
__global__ void scatterKVDecodeKernel(int layer, int num_rows, const __nv_bfloat16 *k_src,
                                      const __nv_bfloat16 *v_src, __nv_bfloat16 *kv_cache,
                                      const int *kv_slot_base, const int *kv_slot_cap,
                                      const int *active_slots, const int *positions);

// <<<ceil(num_tokens / 256), 256>>>
// Marks a whole list of token ids in one slot's repetition-penalty mask row. Only a
// preempted sequence needs this: recomputation replays its prompt and its already-generated
// tokens through prefill, and the mask has to come back with it or the sequence would
// resume with a different penalty state than it was preempted with.
__global__ void markTokenListKernel(const int *tokens, int num_tokens, int slot,
                                    unsigned char *penalty_mask);

// ---- sampling ----
//
// Greedy sampling of one token per logits row, entirely on the device. The host used to copy
// the whole (rows, VOCAB_SIZE) logits block back and scan it, which is 7.58 MiB over PCIe
// plus a 4M-element single-threaded scan every decode step; now only the sampled ids come
// back. See the GPU argmax section of config.h for the two-pass shape.
//
// The repetition penalty moved along with it, because it has to be applied before the
// comparison. Instead of uploading each sequence's token history every step, the engine
// keeps a [BATCH_SIZE, VOCAB_SIZE] byte mask on the device, with a 1 for every token the
// sequence has already emitted; markSampledTokensKernel maintains it. That is the same
// "penalise each distinct previous token once" rule the host-side std::unordered_set had.

// <<<dim3(ARGMAX_CHUNKS_PER_ROW, num_rows), ARGMAX_BLOCK_THREADS>>>
// First pass: block (c, row) reduces tokens [c * ARGMAX_CHUNK_TOKENS, +ARGMAX_CHUNK_TOKENS)
// of its row to one (value, index) pair at partials[row * ARGMAX_CHUNKS_PER_ROW + c].
// penalty_mask may be null, which skips the repetition penalty; when it is given,
// row_slot_ids[row] names the slot whose mask row applies to logits row `row`.
__global__ void argmaxPartialKernel(const __nv_bfloat16 *logits, int num_rows,
                                    const unsigned char *penalty_mask, const int *row_slot_ids,
                                    float *partial_values, int *partial_indices);

// <<<num_rows, ARGMAX_CHUNKS_PER_ROW>>>
// Second pass: reduces one row's partials and writes the winning token id to sampled_tokens.
__global__ void argmaxFinalizeKernel(const float *partial_values, const int *partial_indices,
                                     int num_rows, int *sampled_tokens);

// <<<1, BATCH_SIZE>>>, one thread per row
// Records this step's sampled token in the owning slot's penalty mask row.
__global__ void markSampledTokensKernel(const int *sampled_tokens, const int *row_slot_ids,
                                        int num_rows, unsigned char *penalty_mask);
