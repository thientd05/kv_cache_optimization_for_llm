#include "kernels.h"
#include "config.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


// 1. Nhúng Token hàng loạt
__global__ void embedding_kernel(__half* out, const __half* embed, const int* tokens, int num_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * hidden_size) {
        int t = idx / hidden_size;
        int d = idx % hidden_size;
        out[idx] = embed[tokens[t] * hidden_size + d];
    }
}

// 2. RMSNorm 2D
__global__ void rmsnorm_kernel(__half* out, const __half* in, const __half* weight, int size, int num_tokens) {
    int token_idx = blockIdx.y; 
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    const __half* in_row = in + token_idx * size;
    __half* out_row = out + token_idx * size;
    
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        sum_sq += val * val;
    }
    
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    atomicAdd(&s_sum, sum_sq);
    __syncthreads();
    
    float inv_rms = rsqrtf(s_sum / size + RMS_EPS);
    for (int i = tid; i < size; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(val * inv_rms * w);
    }
}

// 4. RoPE 2D (Chạy song song cho toàn bộ chuỗi prompt)
__global__ void rope_kernel(__half* q, __half* k, int start_pos, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;
    
    int pos = start_pos + token_idx;
    int half_dim = HEAD_DIM / 2;
    
    if (idx < (NUM_Q_HEADS + NUM_KV_HEADS) * half_dim) {
        bool is_q = idx < NUM_Q_HEADS * half_dim;
        int head_idx = is_q ? (idx / half_dim) : ((idx - NUM_Q_HEADS * half_dim) / half_dim);
        int dim_idx = idx % half_dim; 
        
        float freq = 1.0f / powf(ROPE_THETA, (float)(dim_idx * 2) / HEAD_DIM);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        __half* vec = is_q ? 
            &q[token_idx * NUM_Q_HEADS * HEAD_DIM + head_idx * HEAD_DIM] : 
            &k[token_idx * NUM_KV_HEADS * HEAD_DIM + head_idx * HEAD_DIM];
        
        float v0 = __half2float(vec[dim_idx]);
        float v1 = __half2float(vec[dim_idx + half_dim]);
        
        vec[dim_idx]            = __float2half(v0 * fcr - v1 * fci);
        vec[dim_idx + half_dim] = __float2half(v0 * fci + v1 * fcr);
    }
}

// 5. Lưu KV Cache song song
__global__ void save_kv_cache_kernel(const __half* k_buf, const __half* v_buf, __half* k_cache, __half* v_cache, int layer, int start_pos, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;
    
    int total_kv_dim = NUM_KV_HEADS * HEAD_DIM;
    if (idx < total_kv_dim) {
        int actual_pos = start_pos + token_idx;
        int cache_offset = layer * MAX_SEQ_LEN * total_kv_dim + actual_pos * total_kv_dim + idx;
        int buf_offset = token_idx * total_kv_dim + idx;
        
        k_cache[cache_offset] = k_buf[buf_offset];
        v_cache[cache_offset] = v_buf[buf_offset];
    }
}

// 6. Causal Attention 2D (Token i chỉ nhìn được từ 0 đến i)
__global__ void gqa_attention_kernel(__half* out, const __half* q, const __half* k_cache, const __half* v_cache, int start_pos, int num_tokens, int layer) {
    int head_q = blockIdx.x; 
    int token_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;
    
    int actual_pos = start_pos + token_idx;
    int head_kv = head_q / (NUM_Q_HEADS / NUM_KV_HEADS);
    int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    float* att_scores = s_mem; 

    const __half* my_q = q + token_idx * (NUM_Q_HEADS * HEAD_DIM) + head_q * HEAD_DIM;
    
    // Tự động mask Causal bằng cách chỉ cho vòng for chạy đến <= actual_pos
    for (int t = tid; t <= actual_pos; t += blockDim.x) {
        const __half* my_k = k_cache + layer * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM + t * NUM_KV_HEADS * HEAD_DIM + head_kv * HEAD_DIM;
        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM; ++i) {
            score += __half2float(my_q[i]) * __half2float(my_k[i]);
        }
        att_scores[t] = score / sqrtf((float)HEAD_DIM);
    }
    __syncthreads();

    float max_val = -1e20f;
    for (int t = 0; t <= actual_pos; ++t) {
        if (att_scores[t] > max_val) max_val = att_scores[t];
    }
    
    __shared__ float s_sum_exp;
    if (tid == 0) s_sum_exp = 0.0f;
    __syncthreads();

    for (int t = tid; t <= actual_pos; t += blockDim.x) {
        float e = expf(att_scores[t] - max_val);
        att_scores[t] = e;
        atomicAdd(&s_sum_exp, e); 
    }
    __syncthreads();
    
    for (int t = tid; t <= actual_pos; t += blockDim.x) {
        att_scores[t] /= s_sum_exp;
    }
    __syncthreads();

    if (tid < HEAD_DIM) {
        float out_val = 0.0f;
        for (int t = 0; t <= actual_pos; ++t) {
            const __half* my_v = v_cache + layer * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM + t * NUM_KV_HEADS * HEAD_DIM + head_kv * HEAD_DIM;
            out_val += att_scores[t] * __half2float(my_v[tid]);
        }
        out[token_idx * (NUM_Q_HEADS * HEAD_DIM) + head_q * HEAD_DIM + tid] = __float2half(out_val);
    }
}

// 7. Add Residual (Tuyến tính hóa)
__global__ void add_residual_kernel(__half* x, const __half* res, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        x[i] = __float2half(__half2float(x[i]) + __half2float(res[i]));
    }
}


__device__ float silu(float x) { return x / (1.0f + expf(-x)); }

// 3. SwiGLU (Chuyển sang xử lý mảng dài tuyến tính)
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        float val1 = __half2float(hb[i]);
        float val2 = __half2float(hb2[i]);
        hb[i] = __float2half(silu(val1) * val2);
    }
}

// Hàm hỗ trợ: Tìm Max cục bộ trong 1 Warp (32 luồng) bằng Warp Shuffle
__inline__ __device__ void warp_reduce_argmax(float& max_val, int& max_idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // Đẩy giá trị của luồng khác sang luồng hiện tại
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx   = __shfl_down_sync(0xffffffff, max_idx, offset);
        
        // Cập nhật nếu giá trị của luồng kia lớn hơn
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }
}

// Kernel chính
__global__ void argmax_kernel(const __half* logits, int* out_idx, int vocab_size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // 1. Thread-Local Reduction: Mỗi thread tự quét các phần tử của riêng nó
    float local_max = -1e20f;
    int local_best_idx = 0;

    for (int i = tid; i < vocab_size; i += step) {
        float val = __half2float(logits[i]);
        if (val > local_max) {
            local_max = val;
            local_best_idx = i;
        }
    }

    // 2. Warp-Level Reduction: Tìm max trong từng cụm 32 threads
    warp_reduce_argmax(local_max, local_best_idx);

    // 3. Đưa kết quả của mỗi Warp vào Shared Memory
    __shared__ float s_max_val[32];
    __shared__ int s_max_idx[32];

    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // Luồng số 0 của mỗi warp sẽ giữ giá trị lớn nhất của warp đó
    if (lane_id == 0) {
        s_max_val[warp_id] = local_max;
        s_max_idx[warp_id] = local_best_idx;
    }
    __syncthreads(); // Chờ tất cả các warp ghi xong vào shared memory

    // 4. Block-Level Reduction: Dùng warp đầu tiên (warp 0) để gom kết quả của các warp khác
    if (warp_id == 0) {
        // Chỉ đọc các giá trị hợp lệ (trường hợp blockDim.x < 1024)
        local_max = (tid < (blockDim.x / 32)) ? s_max_val[tid] : -1e20f;
        local_best_idx = (tid < (blockDim.x / 32)) ? s_max_idx[tid] : 0;

        // Tiến hành gom lần cuối
        warp_reduce_argmax(local_max, local_best_idx);

        // Luồng 0 cuối cùng sẽ nắm giữ Max toàn cục
        if (tid == 0) {
            *out_idx = local_best_idx;
        }
    }
}