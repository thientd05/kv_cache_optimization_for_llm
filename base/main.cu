#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "config.h"
#include "kernels.h"
#include <chrono>

// --- HOST LOGIC ---

void matmul(cublasHandle_t handle, __half* out, const __half* in, const __half* weight, int m, int k, int n) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    // cuBLAS mặc định là Column-Major, trọng số PyTorch là Row-Major
    // Thực hiện in * weight^T
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                n, m, k, &alpha, weight, k, in, k, &beta, out, n);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: ./llama_engine <prompt_bin> <output_bin> <max_tokens>\n");
        return 1;
    }

    int max_tokens = atoi(argv[3]);

    printf("[DEBUG] Bắt đầu khởi chạy. Đang mở file weights.bin...\n"); fflush(stdout);
    
    // 1. Cấp phát và load weights
    FILE* f_weight = fopen("weights.bin", "rb");

    if (!f_weight) {
        printf("[ERROR] Không tìm thấy file weights.bin!\n");
        return 1;
    }

    printf("[DEBUG] Đang cấp phát VRAM và load trọng số (bước này tốn vài giây)...\n"); fflush(stdout);
    
    __half *d_embed, *d_norm_final;
    __half *d_in_norm[NUM_LAYERS], *d_out_norm[NUM_LAYERS];
    __half *d_q[NUM_LAYERS], *d_k[NUM_LAYERS], *d_v[NUM_LAYERS], *d_o[NUM_LAYERS];
    __half *d_gate[NUM_LAYERS], *d_up[NUM_LAYERS], *d_down[NUM_LAYERS];

    // Đọc Host buffer và copy (Lược giản code check error để tối giản)
    void* host_buf = malloc(VOCAB_SIZE * HIDDEN_SIZE * sizeof(__half)); // Max buffer needed

    cudaMalloc(&d_embed, VOCAB_SIZE * HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_norm_final, HIDDEN_SIZE * sizeof(__half));
    
    fread(host_buf, sizeof(__half), VOCAB_SIZE * HIDDEN_SIZE, f_weight);
    cudaMemcpy(d_embed, host_buf, VOCAB_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < NUM_LAYERS; ++i) {
        cudaMalloc(&d_in_norm[i], HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_out_norm[i], HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_q[i], NUM_Q_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_k[i], NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_v[i], NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_o[i], HIDDEN_SIZE * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_gate[i], INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_up[i], INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half));
        cudaMalloc(&d_down[i], HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half));
        
        fread(host_buf, sizeof(__half), HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_in_norm[i], host_buf, HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_out_norm[i], host_buf, HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        
        fread(host_buf, sizeof(__half), NUM_Q_HEADS * HEAD_DIM * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_q[i], host_buf, NUM_Q_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_k[i], host_buf, NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_v[i], host_buf, NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), HIDDEN_SIZE * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_o[i], host_buf, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        
        fread(host_buf, sizeof(__half), INTERMEDIATE_SIZE * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_gate[i], host_buf, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), INTERMEDIATE_SIZE * HIDDEN_SIZE, f_weight); 
        cudaMemcpy(d_up[i], host_buf, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
        fread(host_buf, sizeof(__half), HIDDEN_SIZE * INTERMEDIATE_SIZE, f_weight); 
        cudaMemcpy(d_down[i], host_buf, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    }
    fread(host_buf, sizeof(__half), HIDDEN_SIZE, f_weight); 
    cudaMemcpy(d_norm_final, host_buf, HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    fclose(f_weight);
    
    // 2. Cấp phát RunState mở rộng cho Batched Prefill
    __half *d_x, *d_xb, *d_h;
    __half *d_q_buf, *d_k_buf, *d_v_buf, *d_att_out;
    __half *d_hb, *d_hb2;
    __half *d_logits;
    int *d_next_token, *d_prompt_tokens;
    
    int max_chunk = MAX_SEQ_LEN; // Sẵn sàng chứa nguyên 1 chunk dài nhất
    
    cudaMalloc(&d_x, max_chunk * HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_xb, max_chunk * HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_h, max_chunk * HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_q_buf, max_chunk * NUM_Q_HEADS * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_k_buf, max_chunk * NUM_KV_HEADS * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_v_buf, max_chunk * NUM_KV_HEADS * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_att_out, max_chunk * HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_hb, max_chunk * INTERMEDIATE_SIZE * sizeof(__half));
    cudaMalloc(&d_hb2, max_chunk * INTERMEDIATE_SIZE * sizeof(__half));
    cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(__half));
    cudaMalloc(&d_next_token, sizeof(int));
    cudaMalloc(&d_prompt_tokens, max_chunk * sizeof(int));
    
    __half *d_k_cache, *d_v_cache;
    cudaMalloc(&d_k_cache, NUM_LAYERS * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_v_cache, NUM_LAYERS * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(__half));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // 3. Đọc Prompt
    FILE* f_prompt = fopen(argv[1], "rb");
    fseek(f_prompt, 0, SEEK_END);
    int prompt_len = ftell(f_prompt) / sizeof(int);
    fseek(f_prompt, 0, SEEK_SET);
    int* prompt_tokens = (int*)malloc(prompt_len * sizeof(int));
    fread(prompt_tokens, sizeof(int), prompt_len, f_prompt);
    fclose(f_prompt);

    int* output_tokens = (int*)malloc(max_tokens * sizeof(int));
    int seq_len = 0;

    // 4. GENERATION LOOP
    // 4. GENERATION LOOP
    printf("Bắt đầu Inference. Chiều dài prompt: %d tokens\n", prompt_len);

    cudaDeviceSynchronize(); // Đồng bộ GPU trước khi đo
    auto start_time = std::chrono::high_resolution_clock::now();
    auto first_token_time = start_time;
    bool is_first_token = true;

    int pos = 0;
    int token = 0; 
    
    // Đẩy toàn bộ prompt lên GPU 1 lần duy nhất trước khi bắt đầu
    cudaMemcpy(d_prompt_tokens, prompt_tokens, prompt_len * sizeof(int), cudaMemcpyHostToDevice);

    while (pos < prompt_len + max_tokens - 1) {
        int num_tokens = (pos == 0) ? prompt_len : 1;
        int block_emb = (num_tokens * HIDDEN_SIZE + 255) / 256;
        
        if (pos == 0) {
            // Lần 1: Xử lý toàn bộ prompt
            embedding_kernel<<<block_emb, 256>>>(d_x, d_embed, d_prompt_tokens, num_tokens, HIDDEN_SIZE);
        } else {
            // Các lần sau: Sử dụng LUÔN d_next_token đang nằm trên GPU (Không cần copy từ Host xuống)
            embedding_kernel<<<block_emb, 256>>>(d_x, d_embed, d_next_token, num_tokens, HIDDEN_SIZE);
        }
        
        for (int l = 0; l < NUM_LAYERS; ++l) {
            // [ĐÃ BỎ] Không dùng cudaMemcpy d_res nữa!
            
            dim3 grid_norm(1, num_tokens);
            rmsnorm_kernel<<<grid_norm, 1024>>>(d_xb, d_x, d_in_norm[l], HIDDEN_SIZE, num_tokens);
            
            matmul(handle, d_q_buf, d_xb, d_q[l], num_tokens, HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM);
            matmul(handle, d_k_buf, d_xb, d_k[l], num_tokens, HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM);
            matmul(handle, d_v_buf, d_xb, d_v[l], num_tokens, HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM);
            
            dim3 grid_rope(((NUM_Q_HEADS + NUM_KV_HEADS) * (HEAD_DIM / 2) + 255) / 256, num_tokens);
            rope_kernel<<<grid_rope, 256>>>(d_q_buf, d_k_buf, pos, num_tokens);
            
            dim3 grid_kv((NUM_KV_HEADS * HEAD_DIM + 255) / 256, num_tokens);
            save_kv_cache_kernel<<<grid_kv, 256>>>(d_k_buf, d_v_buf, d_k_cache, d_v_cache, l, pos, num_tokens);
            
            int shared_mem_size = MAX_SEQ_LEN * sizeof(float);
            dim3 grid_att(NUM_Q_HEADS, num_tokens);
            gqa_attention_kernel<<<grid_att, 256, shared_mem_size>>>(d_att_out, d_q_buf, d_k_cache, d_v_cache, pos, num_tokens, l);
            
            // --- TỐI ƯU RESIDUAL 1 ---
            // Ghi kết quả vào d_h (thay vì ghi đè d_x như cũ)
            matmul(handle, d_h, d_att_out, d_o[l], num_tokens, HIDDEN_SIZE, HIDDEN_SIZE);
            // Cộng thẳng d_h vào d_x tại chỗ
            int block_sz = (num_tokens * HIDDEN_SIZE + 255) / 256;
            add_residual_kernel<<<block_sz, 256>>>(d_x, d_h, num_tokens * HIDDEN_SIZE);
            
            rmsnorm_kernel<<<grid_norm, 1024>>>(d_xb, d_x, d_out_norm[l], HIDDEN_SIZE, num_tokens);
            
            matmul(handle, d_hb, d_xb, d_gate[l], num_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
            matmul(handle, d_hb2, d_xb, d_up[l], num_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
            
            swiglu_kernel<<<(num_tokens * INTERMEDIATE_SIZE + 255) / 256, 256>>>(d_hb, d_hb2, num_tokens * INTERMEDIATE_SIZE);
            
            // --- TỐI ƯU RESIDUAL 2 ---
            // Ghi kết quả vào d_h
            matmul(handle, d_h, d_hb, d_down[l], num_tokens, INTERMEDIATE_SIZE, HIDDEN_SIZE);
            // Cộng thẳng d_h vào d_x
            add_residual_kernel<<<block_sz, 256>>>(d_x, d_h, num_tokens * HIDDEN_SIZE);
        }
        
        dim3 grid_norm_final(1, num_tokens);
        rmsnorm_kernel<<<grid_norm_final, 1024>>>(d_xb, d_x, d_norm_final, HIDDEN_SIZE, num_tokens);
        
        __half* last_token_ptr = d_xb + (num_tokens - 1) * HIDDEN_SIZE;
        matmul(handle, d_logits, last_token_ptr, d_embed, 1, HIDDEN_SIZE, VOCAB_SIZE);
        argmax_kernel<<<1, 1024>>>(d_logits, d_next_token, VOCAB_SIZE);
        
        // Chỉ copy về CPU để kiểm tra điều kiện kết thúc.
        // GPU không bị nghẽn ở vòng lặp tiếp theo vì ta dùng trực tiếp d_next_token.
        cudaMemcpy(&token, d_next_token, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (pos == 0) {
            printf("=> Prefill xong! Sinh token đầu tiên: %d\n", token);
        } else {
            printf("=> Gen token: %d\n", token);
        }

        output_tokens[seq_len++] = token;
        if (token == 128001 || token == 128009) {
            printf("-> Gặp token EOS, dừng sinh!\n");
            break;
        }
        
        pos += num_tokens; 
    }
    printf("Hoàn thành vòng lặp. Đang ghi file output...\n");


    // Kết thúc bấm giờ toàn bộ vòng lặp
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Tính toán thời gian (tính bằng giây)
    std::chrono::duration<double> prefill_elapsed = first_token_time - start_time;
    std::chrono::duration<double> decode_elapsed = end_time - first_token_time;
    std::chrono::duration<double> total_elapsed = end_time - start_time;

    double prefill_time = prefill_elapsed.count();
    double decode_time = decode_elapsed.count();
    
    // Đảm bảo không chia cho 0 nếu gen lỗi
    if (decode_time <= 0.0) decode_time = 0.0001; 
    
    printf("\n=== THỐNG KÊ HIỆU NĂNG (CUDA ENGINE) ===\n");
    printf("- Số token prompt          : %d\n", prompt_len);
    printf("- Số token gen ra          : %d\n", seq_len);
    printf("- Thời gian xử lý Prompt   : %.3f s (Time to First Token)\n", prefill_time);
    printf("- Thời gian sinh Token     : %.3f s (Decoding phase)\n", decode_time);
    printf("- Tổng thời gian tính toán : %.3f s\n", total_elapsed.count());
    printf("\n=> TỐC ĐỘ GENERATION     : %.2f tokens/giây\n", seq_len / decode_time);
    printf("=========================================\n");

    // Xuất kết quả
    FILE* f_out = fopen(argv[2], "wb");
    fwrite(output_tokens, sizeof(int), seq_len, f_out);
    fclose(f_out);

    return 0;
}