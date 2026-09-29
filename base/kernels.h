#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void embedding_kernel(__half* out, const __half* embed, const int* tokens, int num_tokens, int hidden_size);
__global__ void rmsnorm_kernel(__half* out, const __half* in, const __half* weight, int size, int num_tokens);
__device__ float silu(float x);
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int total_elements);
__global__ void rope_kernel(__half* q, __half* k, int start_pos, int num_tokens);
__global__ void save_kv_cache_kernel(const __half* k_buf, const __half* v_buf, __half* k_cache, __half* v_cache, int layer, int start_pos, int num_tokens);
__global__ void add_residual_kernel(__half* x, const __half* res, int total_elements);
__global__ void gqa_attention_kernel(__half* out, const __half* q, const __half* k_cache, const __half* v_cache, int start_pos, int num_tokens, int layer);
__global__ void argmax_kernel(const __half* logits, int* out_idx, int vocab_size);