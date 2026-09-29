import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

model_path = "./model/Llama-3.2-1B-Instruct"

print("LOADING TOKENIZER...")
tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)

print("LOADING MODEL...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# --- 1. WARM-UP (KHỞI ĐỘNG LẠNH) ---
print("Đang khởi động CUDA (Warm-up)...")
dummy_inputs = tokenizer("hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    model.generate(**dummy_inputs, max_new_tokens=2)
torch.cuda.synchronize() # Đảm bảo GPU đã sẵn sàng

# --- 2. CUSTOM STREAMER ĐỂ ĐO TIME-TO-FIRST-TOKEN ---
class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.is_first_token = True
        self.time_first_token = 0

    def put(self, value):
        if self.is_first_token:
            torch.cuda.synchronize() # Chờ GPU thực sự đẩy tensor ra
            self.time_first_token = time.perf_counter()
            self.is_first_token = False

    def end(self):
        pass

# --- 3. ĐO ĐẠC THỰC TẾ ---
messages = [{"role": "user", "content": "hello"}]
prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_templated, return_tensors="pt").to("cuda")
prompt_len = inputs["input_ids"].shape[1]

streamer = TimingStreamer()

print("GENERATE...\n")
torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150, # Đặt cao để model tự sinh đến token EOS
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

torch.cuda.synchronize()
end_time = time.perf_counter()

# --- 4. TÍNH TOÁN (Giống hệt logic của file main.cu) ---
num_tokens_generated = outputs.shape[1] - prompt_len

prefill_time = streamer.time_first_token - start_time
decode_time = end_time - streamer.time_first_token
total_time = end_time - start_time

print("\n=== THỐNG KÊ HIỆU NĂNG (HUGGING FACE) ===")
print(f"- Số token prompt          : {prompt_len}")
print(f"- Số token gen ra          : {num_tokens_generated}")
print(f"- Thời gian xử lý Prompt   : {prefill_time:.3f} s (Time to First Token)")
print(f"- Thời gian sinh Token     : {decode_time:.3f} s (Decoding phase)")
print(f"- Tổng thời gian tính toán : {total_time:.3f} s")
print(f"\n=> TỐC ĐỘ GENERATION       : {num_tokens_generated / decode_time:.2f} tokens/giây")
print("===========================================")