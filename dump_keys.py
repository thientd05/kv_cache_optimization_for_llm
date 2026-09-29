from safetensors import safe_open

tensors_path = "./model/Llama-3.2-1B-Instruct/model.safetensors"

with safe_open(tensors_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")

    print("-" * 50)
    for key in keys[:15]:
        print(f"{key}: {f.get_slice(key).get_shape()}")