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
// Writes the whole batch's K/V of one layer into the paged cache in a single launch.
// Replaces the old per-page cudaMemcpy pair: a 512-token prompt used to cost
// 2 * 32 memcpys per layer, i.e. ~16k cudaMemcpy calls for a 16-prompt batch.
// Pages must already be reserved in block_table_gpu by the host.
__global__ void scatterKVPackedKernel(int layer, int num_tokens, const __nv_bfloat16 *k_src,
                                      const __nv_bfloat16 *v_src, __nv_bfloat16 *kv_cache,
                                      const int *block_table_gpu, const int *token_slot_ids,
                                      const int *token_positions);

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
__global__ void pagedAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj,
                                     __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens,
                                     int *gpu_active_slots, __nv_bfloat16 *output);
