import os
import struct
import subprocess
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
import time

MODEL_PATH = "../model/Llama-3.2-1B-Instruct"
WEIGHTS_BIN = "weights.bin"
PROMPT_BIN = "prompt.bin"
OUTPUT_BIN = "output.bin"
EXE_PATH = "./llama_engine"

def export_weights_if_needed():
    if os.path.exists(WEIGHTS_BIN):
        return
    print(f"Đang xuất trọng số từ {MODEL_PATH}/model.safetensors sang {WEIGHTS_BIN} (chỉ chạy lần đầu)...")
    with safe_open(f"{MODEL_PATH}/model.safetensors", framework="pt", device="cpu") as f:
        with open(WEIGHTS_BIN, "wb") as out_f:
            # 1. Embeddings
            out_f.write(f.get_tensor("model.embed_tokens.weight").to(torch.float16).numpy().tobytes())
            # 2. 16 Layers
            for i in range(16):
                prefix = f"model.layers.{i}."
                out_f.write(f.get_tensor(prefix + "input_layernorm.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "post_attention_layernorm.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "self_attn.q_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "self_attn.k_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "self_attn.v_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "self_attn.o_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "mlp.gate_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "mlp.up_proj.weight").to(torch.float16).numpy().tobytes())
                out_f.write(f.get_tensor(prefix + "mlp.down_proj.weight").to(torch.float16).numpy().tobytes())
            # 3. Final Norm & LM Head (tie_word_embeddings = true nên dùng lại embed_tokens)
            out_f.write(f.get_tensor("model.norm.weight").to(torch.float16).numpy().tobytes())
    print("Xuất trọng số hoàn tất!")

def main():
    export_weights_if_needed()
    
    print("Đang tải Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=False)
    
    print("="*50)
    print("🚀 LLAMA 3.2 1B - C++/CUDA INFERENCE ENGINE")
    print("Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("="*50)

    while True:
        try:
            # Nhận prompt từ người dùng
            user_input = input("\n👤 Bạn: ")
            
            # Điều kiện thoát
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Tạm biệt!")
                break
            if not user_input.strip():
                continue

            # Đóng gói prompt vào template của Llama
            messages = [{"role": "user", "content": user_input}]
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt", 
                return_dict=True
            )
            input_ids = inputs["input_ids"][0].tolist()
            
            # Ghi prompt ra file nhị phân cho C++
            with open(PROMPT_BIN, "wb") as f:
                f.write(struct.pack(f"{len(input_ids)}i", *input_ids))
            
            # Gọi C++ Engine (tăng lên 250 tokens)
            # capture_output=True sẽ giấu đi các dòng printf debug của C++
            # Bắt đầu tính tổng thời gian người dùng chờ
            start_t = time.time()
            subprocess.run([EXE_PATH, PROMPT_BIN, OUTPUT_BIN, "1000"], capture_output=False)
            end_t = time.time()
            
            # Đọc file kết quả do C++ sinh ra
            with open(OUTPUT_BIN, "rb") as f:
                data = f.read()
                # Cấu trúc lại thành mảng tuple các số nguyên
                output_ids = struct.unpack(f"{len(data)//4}i", data)
            
            # Giải mã và in ra màn hình
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"🤖 Llama:\n{response}")

            # In ra độ trễ tổng thể
            print(f"\n[Python] Tổng độ trễ End-to-End: {end_t - start_t:.2f} giây (Bao gồm overhead Load Tạ)")
            
        except KeyboardInterrupt:
            # Xử lý khi bấm Ctrl+C
            print("\nĐã hủy. Tạm biệt!")
            break

if __name__ == "__main__":
    main()