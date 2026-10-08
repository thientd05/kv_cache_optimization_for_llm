#include "kernels.cuh"
#include <cmath>
#include <vector>

float *d_cos_table = nullptr; // [MAX_SEQ_LEN, HEAD_DIM]
float *d_sin_table = nullptr; // [MAX_SEQ_LEN, HEAD_DIM]

// ---- shared by prefill and decode ----

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ float rms_vector[EMBED_HALF];
    int workIndex = threadIdx.x + blockIdx.x * EMBEDDING_LENGTH;
    if (workIndex < num_tokens * EMBEDDING_LENGTH)
    {
        rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + EMBED_HALF] * (float)input[workIndex + EMBED_HALF];
        __syncthreads();
        // tree reduction
        for (int i = 1; i < EMBED_HALF; i = i * 2)
        {
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            rms_vector[0] = sqrt(rms_vector[0] / (double)EMBEDDING_LENGTH + RMS_NORM_EPS);
        }
        __syncthreads();
        // <(^-^)>
        output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
        output[workIndex + EMBED_HALF] = (__nv_bfloat16)(((float)input[workIndex + EMBED_HALF] / rms_vector[0]) * (float)norm_weights[threadIdx.x + EMBED_HALF]);
    }
}

__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * EMBEDDING_LENGTH;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + EMBED_HALF] = input[workIndex + EMBED_HALF] + input_embeds[workIndex + EMBED_HALF];
}

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * HIDDEN_DIM;
    for (int i = 0; i < HIDDEN_DIM; i += MAX_THREADS_PER_BLOCK)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// ---- RoPE tables ----

void initRopeFrequencies()
{
    constexpr int half_dim = HEAD_DIM / 2;
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++)
    {
        inv_freq[i] = 1.0f / std::pow(ROPE_THETA, (2.0f * i) / HEAD_DIM);
    }
    float low_freq_wavelen = (float)ROPE_ORIGINAL_MAX_LEN / ROPE_LOW_FREQ_FACTOR;
    float high_freq_wavelen = (float)ROPE_ORIGINAL_MAX_LEN / ROPE_HIGH_FREQ_FACTOR;

    std::vector<float> inv_freq_llama = inv_freq;

    for (int i = 0; i < half_dim; i++)
    {
        float wavelen = 2.0f * M_PI / inv_freq[i];

        if (wavelen > low_freq_wavelen)
        {
            inv_freq_llama[i] = inv_freq[i] / ROPE_SCALING_FACTOR;
        }
        else if (wavelen >= high_freq_wavelen)
        {
            float smooth = ((float)ROPE_ORIGINAL_MAX_LEN / wavelen - ROPE_LOW_FREQ_FACTOR) / (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR);
            inv_freq_llama[i] = (1.0f - smooth) * (inv_freq[i] / ROPE_SCALING_FACTOR) + smooth * inv_freq[i];
        }
    }

    std::vector<float> cos_table(MAX_SEQ_LEN * HEAD_DIM);
    std::vector<float> sin_table(MAX_SEQ_LEN * HEAD_DIM);

    for (int pos = 0; pos < MAX_SEQ_LEN; pos++)
    {
        for (int i = 0; i < half_dim; i++)
        {
            float angle = pos * inv_freq_llama[i];
            float c = std::cos(angle);
            float s = std::sin(angle);
            cos_table[pos * HEAD_DIM + 2 * i] = c;
            cos_table[pos * HEAD_DIM + 2 * i + 1] = c;
            sin_table[pos * HEAD_DIM + 2 * i] = s;
            sin_table[pos * HEAD_DIM + 2 * i + 1] = s;
        }
    }

    constexpr size_t table_bytes = (size_t)MAX_SEQ_LEN * HEAD_DIM * sizeof(float);
    cudaMalloc(&d_cos_table, table_bytes);
    cudaMalloc(&d_sin_table, table_bytes);
    cudaMemcpy(d_cos_table, cos_table.data(), table_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin_table, sin_table.data(), table_bytes, cudaMemcpyHostToDevice);
}

// ---- decode ----

__global__ void embeddingGatherDecodeKernel(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    int input_token = gpu_last_tokens[blockIdx.x];
    int workIndex = blockIdx.x * EMBEDDING_LENGTH + threadIdx.x;
    if (workIndex < num_tokens * EMBEDDING_LENGTH)
    {
        output[workIndex] = embed_tokens[input_token * EMBEDDING_LENGTH + threadIdx.x];
        output[workIndex + EMBED_HALF] = embed_tokens[input_token * EMBEDDING_LENGTH + threadIdx.x + EMBED_HALF];
    }
}

__global__ void ropeDecodeKernel(__nv_bfloat16 *input, int position_in_sequence, int proj_dim,
                                 const float *cos_table, const float *sin_table)
{
    int tid = threadIdx.x;
    int half_proj = proj_dim / 2;
    constexpr int half_dim = HEAD_DIM / 2;

    if (tid >= half_proj)
        return;

    int head_idx = tid / half_dim;
    int pair_idx = tid % half_dim;

    int base = head_idx * HEAD_DIM;
    int idx1 = base + pair_idx;
    int idx2 = base + pair_idx + half_dim;

    float x1 = (float)input[idx1];
    float x2 = (float)input[idx2];

    int table_idx = position_in_sequence * HEAD_DIM + pair_idx * 2;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    input[idx1] = (__nv_bfloat16)(x1 * c - x2 * s);
    input[idx2] = (__nv_bfloat16)(x1 * s + x2 * c);
}

// inside a single particular thread that processes a single position of particular Q head for a particular sequence, for particular layer
__global__ void pagedAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj,
                                     __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens,
                                     int *gpu_active_slots, __nv_bfloat16 *output)
{
    __shared__ float dot_products[2];
    int active_slot = blockIdx.x; // active_slot == seq_id
    int slot = gpu_active_slots[active_slot];
    int q_head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;
    __nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];
    int seq_len = gpu_seq_lens[active_slot];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // for online softmax https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    float current_max = -INFINITY;
    float acc = 0.0f;
    float d = 0.0f; // denominator, same name as in paper above

    for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx)
    {
        int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
        int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
        for (int token = 0; token < tokens_in_block; ++token)
        {
            __nv_bfloat16 *k = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            __nv_bfloat16 *v = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + V_OFFSET + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            float qk = (float)q * (float)*k;
            // tree reduction within current warp, thread 0 gets sum of all 32 elements within warp
            // could be done with __syncthreads but accessing memory of other threads in warp is op
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 16);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 8);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 4);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 2);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 1);
            if (thread_id == 0)
            {
                dot_products[0] = qk;
            }
            if (thread_id == WARP_SIZE)
            {
                dot_products[1] = qk;
            }
            __syncthreads();
            if (thread_id == 0)
            {
                dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
            }
            __syncthreads();
            float dot_product = dot_products[0];
            // online softmax
            float new_max = current_max;
            if (dot_product > current_max)
            {
                new_max = dot_product;
            }
            float correction_factor = expf(current_max - new_max);
            current_max = new_max;
            float exp_score = expf(dot_product - current_max);
            d = d * correction_factor + exp_score;
            acc = acc * correction_factor + exp_score * (float)*v;
            // Warp 0 overwrites dot_products[0] for the next token as soon as it gets
            // there, so without this barrier warp 1 can still be reading the current
            // token's score when that write lands. Both loop bounds are block-uniform
            // (seq_len and logical_block_idx are), so every thread reaches this barrier.
            __syncthreads();
        }
    }
    output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;
}

__global__ void ropeDecodeBatchKernel(__nv_bfloat16 *input, int num_rows, int proj_dim,
                                      const int *positions, const float *cos_table,
                                      const float *sin_table)
{
    int row = blockIdx.x;
    if (row >= num_rows)
        return;

    int tid = threadIdx.x;
    int half_proj = proj_dim / 2;
    constexpr int half_dim = HEAD_DIM / 2;
    if (tid >= half_proj)
        return;

    int head_idx = tid / half_dim;
    int pair_idx = tid % half_dim;

    __nv_bfloat16 *row_ptr = input + (size_t)row * proj_dim;
    int base = head_idx * HEAD_DIM;
    int idx1 = base + pair_idx;
    int idx2 = base + pair_idx + half_dim;

    float x1 = (float)row_ptr[idx1];
    float x2 = (float)row_ptr[idx2];

    int table_idx = positions[row] * HEAD_DIM + pair_idx * 2;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    row_ptr[idx1] = (__nv_bfloat16)(x1 * c - x2 * s);
    row_ptr[idx2] = (__nv_bfloat16)(x1 * s + x2 * c);
}

__global__ void markTokenListKernel(const int *tokens, int num_tokens, int slot,
                                    unsigned char *penalty_mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_tokens)
    {
        penalty_mask[(size_t)slot * VOCAB_SIZE + tokens[i]] = 1;
    }
}

__global__ void scatterKVDecodeKernel(int layer, int num_rows, const __nv_bfloat16 *k_src,
                                      const __nv_bfloat16 *v_src, __nv_bfloat16 *kv_cache,
                                      const int *block_table_gpu, const int *active_slots,
                                      const int *positions)
{
    int row = blockIdx.x;
    if (row >= num_rows)
        return;

    int slot = active_slots[row];
    int position = positions[row];
    int logical_block_idx = position / BLOCK_SIZE;
    int token_in_block_idx = position % BLOCK_SIZE;

    int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
    if (physical_block < 0)
        return; // host failed to map the page, nothing sane to write

    char *page = (char *)kv_cache + (size_t)physical_block * BLOCK_BYTES + (size_t)token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16);
    __nv_bfloat16 *k_dst = (__nv_bfloat16 *)page;
    __nv_bfloat16 *v_dst = (__nv_bfloat16 *)(page + V_OFFSET);

    size_t src = (size_t)row * KV_DIM + threadIdx.x;
    k_dst[threadIdx.x] = k_src[src];
    v_dst[threadIdx.x] = v_src[src];
}

// ---- batched ("packed") prefill ----

__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * EMBEDDING_LENGTH;
    if (workIndex < num_input_tokens * EMBEDDING_LENGTH)
    {
        gpu_input_embeds[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * EMBEDDING_LENGTH + threadIdx.x];
        gpu_input_embeds[workIndex + EMBED_HALF] = embed_tokens[gpu_input_tokens[blockIdx.x] * EMBEDDING_LENGTH + threadIdx.x + EMBED_HALF];
    }
}

__global__ void ropePackedKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim,
                                 const int *token_positions, const float *cos_table,
                                 const float *sin_table)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int half_proj = proj_dim / 2;
    constexpr int half_dim = HEAD_DIM / 2;

    if (token_idx >= num_tokens || tid >= half_proj)
        return;

    int head_idx = tid / half_dim;
    int pair_idx = tid % half_dim;

    size_t base = (size_t)token_idx * proj_dim + head_idx * HEAD_DIM;
    size_t idx1 = base + pair_idx;
    size_t idx2 = base + pair_idx + half_dim;

    float x1 = (float)input[idx1];
    float x2 = (float)input[idx2];

    // the only difference from ropeDecodeKernel: the angle comes from the token's
    // position inside its own prompt, not from its index in the packed buffer
    int table_idx = token_positions[token_idx] * HEAD_DIM + pair_idx * 2;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    input[idx1] = (__nv_bfloat16)(x1 * c - x2 * s);
    input[idx2] = (__nv_bfloat16)(x1 * s + x2 * c);
}

__global__ void scatterKVPackedKernel(int layer, int num_tokens, const __nv_bfloat16 *k_src,
                                      const __nv_bfloat16 *v_src, __nv_bfloat16 *kv_cache,
                                      const int *block_table_gpu, const int *token_slot_ids,
                                      const int *token_positions)
{
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    int slot = token_slot_ids[token_idx];
    int position = token_positions[token_idx];
    int logical_block_idx = position / BLOCK_SIZE;
    int token_in_block_idx = position % BLOCK_SIZE;

    int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
    if (physical_block < 0)
        return; // host failed to reserve the page, nothing sane to write

    char *page = (char *)kv_cache + (size_t)physical_block * BLOCK_BYTES + (size_t)token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16);
    __nv_bfloat16 *k_dst = (__nv_bfloat16 *)page;
    __nv_bfloat16 *v_dst = (__nv_bfloat16 *)(page + V_OFFSET);

    size_t src = (size_t)token_idx * KV_DIM + threadIdx.x;
    k_dst[threadIdx.x] = k_src[src];
    v_dst[threadIdx.x] = v_src[src];
}

// Flash-attention style prefill attention over the packed batch: one launch per layer for
// the whole batch, and nothing of size seq_len^2 is ever materialised. The old path built a
// (NUM_Q_HEADS, L, L) score buffer per prompt and walked it three times (mask, softmax,
// scores*V); for L=512 that is ~96 MiB of traffic per prompt per layer, and it needed 66
// kernel/GEMM launches per prompt per layer on top.
//
// grid  = (num_tiles, NUM_Q_HEADS): one block per (tile of query tokens, Q head)
// block = PREFILL_ATTN_QUERIES_PER_TILE warps of 32 lanes; warp w owns query token
//         tile_token_begin[tile] + w and keeps its own online softmax state end to end,
//         so there is no cross-warp reduction.
// Each lane owns head dim elements `lane` and `lane + 32` (HEAD_DIM is 64).
//
// The warps of a block step through their prompt's keys together, staging
// PREFILL_ATTN_KEYS_PER_TILE keys of K and V in shared memory at a time. That staging is
// the whole point: one query token per block meant every query re-read all of K and V from
// DRAM (~70 GB for a full 16-prompt batch, ~300 ms on this card's 192 GB/s), whereas a tile
// of 8 queries reads each K/V element once for all 8.
__global__ void prefillAttentionKernel(const __nv_bfloat16 *q_proj, const __nv_bfloat16 *k_packed,
                                       const __nv_bfloat16 *v_packed, const int *tile_token_begin,
                                       const int *tile_token_count, const int *token_seq_start,
                                       const int *token_positions, __nv_bfloat16 *output)
{
    // 2 * 32 * 64 * 2 B = 8 KiB, so 4 blocks of 256 threads still fit in the 64 KiB an SM has
    __shared__ __nv_bfloat16 k_tile[PREFILL_ATTN_KEYS_PER_TILE][HEAD_DIM];
    __shared__ __nv_bfloat16 v_tile[PREFILL_ATTN_KEYS_PER_TILE][HEAD_DIM];

    int tile = blockIdx.x;
    int q_head_id = blockIdx.y;
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;

    int warp = threadIdx.x / PREFILL_ATTN_LANES;
    int lane = threadIdx.x % PREFILL_ATTN_LANES;

    int token_begin = tile_token_begin[tile];
    int token_count = tile_token_count[tile];
    int seq_start = token_seq_start[token_begin];
    // how far the block as a whole has to walk: the last query in the tile sits deepest
    int tile_num_keys = token_positions[token_begin + token_count - 1] + 1;

    // the final tile of a prompt can be short, those warps idle but still have to reach
    // every __syncthreads() below
    bool active = warp < token_count;
    int token_idx = token_begin + warp;
    int num_keys = active ? token_positions[token_idx] + 1 : 0; // causal limit for this query

    // Reading and writing the same q_proj rows is safe: block (tile, head) is the only block
    // touching q_proj[token, head, :] for its tokens, and it loads them before writing anything.
    float q0 = 0.0f;
    float q1 = 0.0f;
    if (active)
    {
        const __nv_bfloat16 *q = q_proj + (size_t)token_idx * EMBEDDING_LENGTH + q_head_id * HEAD_DIM;
        q0 = (float)q[lane];
        q1 = (float)q[lane + PREFILL_ATTN_LANES];
    }

    // online softmax, same recurrence as pagedAttentionKernel
    float current_max = -INFINITY;
    float denom = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int base = 0; base < tile_num_keys; base += PREFILL_ATTN_KEYS_PER_TILE)
    {
        int keys_staged = min(PREFILL_ATTN_KEYS_PER_TILE, tile_num_keys - base);

        for (int element = threadIdx.x; element < keys_staged * HEAD_DIM; element += PREFILL_ATTN_THREADS)
        {
            int key = element / HEAD_DIM;
            int head_dim_idx = element % HEAD_DIM;
            size_t row = (size_t)(seq_start + base + key) * KV_DIM + kv_head_idx * HEAD_DIM + head_dim_idx;
            k_tile[key][head_dim_idx] = k_packed[row];
            v_tile[key][head_dim_idx] = v_packed[row];
        }
        __syncthreads();

        // keys_staged is block-uniform and num_keys is warp-uniform, so every lane of a warp
        // runs the same number of iterations and the shuffles below stay full-warp
        int keys_for_this_query = min(keys_staged, num_keys - base);
        for (int key = 0; key < keys_for_this_query; ++key)
        {
            float dot = q0 * (float)k_tile[key][lane] + q1 * (float)k_tile[key][lane + PREFILL_ATTN_LANES];
            // xor shuffle so every lane ends up with the full 64-element dot product
            for (int offset = PREFILL_ATTN_LANES / 2; offset > 0; offset >>= 1)
            {
                dot += __shfl_xor_sync(WARP_FULL_MASK, dot, offset);
            }
            dot /= SQRT_HEAD_DIM;

            // Same online softmax recurrence as pagedAttentionKernel, rearranged: the
            // running max only moves O(log n) times, so most keys skip the rescale and its
            // exp. dot is warp-uniform, so this branch never diverges.
            float exp_score;
            if (dot > current_max)
            {
                float correction_factor = (current_max == -INFINITY) ? 0.0f : __expf(current_max - dot);
                current_max = dot;
                denom *= correction_factor;
                acc0 *= correction_factor;
                acc1 *= correction_factor;
                exp_score = 1.0f; // exp(dot - current_max) with current_max just set to dot
            }
            else
            {
                exp_score = __expf(dot - current_max);
            }

            denom += exp_score;
            acc0 += exp_score * (float)v_tile[key][lane];
            acc1 += exp_score * (float)v_tile[key][lane + PREFILL_ATTN_LANES];
        }
        __syncthreads(); // nobody may overwrite the staged tile while a slower warp still reads it
    }

    if (active)
    {
        // num_keys >= 1 for an active warp and the key at the running max contributes
        // exp(0) = 1, so denom >= 1
        __nv_bfloat16 *out = output + (size_t)token_idx * EMBEDDING_LENGTH + q_head_id * HEAD_DIM;
        out[lane] = (__nv_bfloat16)(acc0 / denom);
        out[lane + PREFILL_ATTN_LANES] = (__nv_bfloat16)(acc1 / denom);
    }
}

// Picks out the rows named by row_indices, which lets the lm_head GEMM run on just the
// last token of every prompt instead of on all of them. That GEMM is
// (num_rows, 2048) x (2048, 128256): at 512 tokens it is 269 GFLOP per prompt and all but
// one row of the result was thrown away.
__global__ void gatherRowsKernel(const __nv_bfloat16 *src, __nv_bfloat16 *dst, const int *row_indices, int num_rows)
{
    if (blockIdx.x >= num_rows)
        return;
    size_t src_base = (size_t)row_indices[blockIdx.x] * EMBEDDING_LENGTH + threadIdx.x;
    size_t dst_base = (size_t)blockIdx.x * EMBEDDING_LENGTH + threadIdx.x;
    dst[dst_base] = src[src_base];
    dst[dst_base + EMBED_HALF] = src[src_base + EMBED_HALF];
}

// ---- sampling ----

// Keeps the better of two (value, index) candidates. The tie-break on the lower index is
// what the host scan did implicitly with its strict `>`, and it has to be kept: ties on
// bf16 logits are not rare at all (bf16 has 8 mantissa bits, so a 128k-entry row has plenty
// of exactly equal values), and without it the sampled token would depend on how the
// vocabulary happened to be split across blocks.
__device__ __forceinline__ void argmaxMerge(float &best_value, int &best_index, float value, int index)
{
    if (value > best_value || (value == best_value && index < best_index))
    {
        best_value = value;
        best_index = index;
    }
}

__global__ void argmaxPartialKernel(const __nv_bfloat16 *logits, int num_rows,
                                    const unsigned char *penalty_mask, const int *row_slot_ids,
                                    float *partial_values, int *partial_indices)
{
    const int row = blockIdx.y;
    if (row >= num_rows)
    {
        return;
    }

    const int chunk_begin = blockIdx.x * ARGMAX_CHUNK_TOKENS;
    const int chunk_end = min(chunk_begin + ARGMAX_CHUNK_TOKENS, VOCAB_SIZE);

    const __nv_bfloat16 *row_logits = logits + (size_t)row * VOCAB_SIZE;
    // the mask is indexed by slot, the logits by their position in this pass
    const unsigned char *row_mask = penalty_mask == nullptr
                                        ? nullptr
                                        : penalty_mask + (size_t)row_slot_ids[row] * VOCAB_SIZE;

    float best_value = -INFINITY;
    int best_index = VOCAB_SIZE; // out of range, so a chunk that saw nothing never wins a tie

    // consecutive threads read consecutive tokens, so every pass over the row is coalesced
    for (int token = chunk_begin + threadIdx.x; token < chunk_end; token += ARGMAX_BLOCK_THREADS)
    {
        float logit = (float)row_logits[token];
        if (row_mask != nullptr && row_mask[token] != 0)
        {
            // same two-sided penalty as before: divide a positive logit, multiply a negative
            // one, so either way the token is pushed down
            logit = logit > 0.0f ? logit / REPETITION_PENALTY : logit * REPETITION_PENALTY;
        }
        argmaxMerge(best_value, best_index, logit, token);
    }

    // reduce inside the warp first; a lane whose partner does not exist shuffles its own
    // value back, and merging a candidate with itself changes nothing
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        float other_value = __shfl_down_sync(WARP_FULL_MASK, best_value, offset);
        int other_index = __shfl_down_sync(WARP_FULL_MASK, best_index, offset);
        argmaxMerge(best_value, best_index, other_value, other_index);
    }

    constexpr int warps_per_block = ARGMAX_BLOCK_THREADS / WARP_SIZE;
    __shared__ float warp_values[warps_per_block];
    __shared__ int warp_indices[warps_per_block];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0)
    {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int other = 1; other < warps_per_block; ++other)
        {
            argmaxMerge(warp_values[0], warp_indices[0], warp_values[other], warp_indices[other]);
        }
        const int partial = row * ARGMAX_CHUNKS_PER_ROW + blockIdx.x;
        partial_values[partial] = warp_values[0];
        partial_indices[partial] = warp_indices[0];
    }
}

__global__ void argmaxFinalizeKernel(const float *partial_values, const int *partial_indices,
                                     int num_rows, int *sampled_tokens)
{
    const int row = blockIdx.x;
    if (row >= num_rows)
    {
        return;
    }

    __shared__ float values[ARGMAX_CHUNKS_PER_ROW];
    __shared__ int indices[ARGMAX_CHUNKS_PER_ROW];

    const int chunk = threadIdx.x;
    values[chunk] = partial_values[row * ARGMAX_CHUNKS_PER_ROW + chunk];
    indices[chunk] = partial_indices[row * ARGMAX_CHUNKS_PER_ROW + chunk];
    __syncthreads();

    // ARGMAX_CHUNKS_PER_ROW is a power of two, so the halving never leaves an odd element out
    for (int stride = ARGMAX_CHUNKS_PER_ROW / 2; stride > 0; stride >>= 1)
    {
        if (chunk < stride)
        {
            argmaxMerge(values[chunk], indices[chunk], values[chunk + stride], indices[chunk + stride]);
        }
        __syncthreads();
    }

    if (chunk == 0)
    {
        sampled_tokens[row] = indices[0];
    }
}

__global__ void markSampledTokensKernel(const int *sampled_tokens, const int *row_slot_ids,
                                        int num_rows, unsigned char *penalty_mask)
{
    const int row = threadIdx.x;
    if (row >= num_rows)
    {
        return;
    }
    penalty_mask[(size_t)row_slot_ids[row] * VOCAB_SIZE + sampled_tokens[row]] = 1;
}
