import os
from transformers import AutoTokenizer

# The HF model snapshot lives in model/ at the repo root (two levels up from this file).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(REPO_ROOT, "model", "Llama-3.2-1B-Instruct")
)

t = AutoTokenizer.from_pretrained(MODEL_DIR)

prompts = {
    "What is 2+2?": [791, 4320, 374, 220, 19],
    "Name a color.": [10544],
    "Say hello.": [9906, 0, 2650, 649, 358, 1520, 499, 449, 4205, 499, 1205, 30],
    "Capital of France?": [791, 6864, 315, 9822, 374, 12366, 13],
}

for prompt, tokens in prompts.items():
    print(f"{prompt} → {t.decode(tokens)}")