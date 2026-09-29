import os
from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
save_dir = "./model/Llama-3.2-1B-Instruct"

print("start")

snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pt", "*.bin", "*.pth"]
)

print("done")