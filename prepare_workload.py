"""Build the benchmark workloads from the datasets the PagedAttention paper uses.

Paper §6.1: "We synthesize workloads based on ShareGPT and Alpaca datasets, which contain
input and output texts of real LLM services... We tokenize the datasets and use their input
and output lengths to synthesize client requests."

The point of using the real datasets rather than a fitted distribution is that prompt length
and output length are *correlated* in real traffic, and it is their sum that decides how much
KV cache a sequence occupies. Drawing them independently would mis-model exactly the variable
under test.

Two datasets on purpose, because the paper's conclusion changes with length: ShareGPT has
8.4x longer prompts and 5.8x longer outputs than Alpaca.

Caching: each dataset is downloaded once, tokenized once, and written to
workloads/<name>.json as a list of {prompt, prompt_len, output_len}. bench.py reads that, so
no run pays for tokenisation. Re-running this script is a no-op unless --force is given.

usage: python prepare_workload.py [--force] [--max-samples N]
"""

import argparse
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct"))
WORKLOAD_DIR = os.path.join(REPO_ROOT, "workloads")
CACHE_DIR = os.path.join(REPO_ROOT, "workloads", ".cache")

# Must match src/config.h (shared block). A sample longer than either cap is dropped rather than
# truncated: truncating would squash the right tail of the length distribution, and the right
# tail is precisely what punishes a reservation-based cache.
MAX_PROMPT_LEN = 1024
MAX_NEW_TOKENS_GENERATED = 1024

# The dataset the paper calls "Alpaca". This is the original release from the Stanford repo,
# plain JSON, which avoids needing parquet support just to read 52k short instructions.
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# The ShareGPT snapshot vLLM's own serving benchmark uses.
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def fetch_alpaca():
    local = os.path.join(CACHE_DIR, "alpaca_data.json")
    if not os.path.exists(local):
        print(f"  downloading {ALPACA_URL}")
        urllib.request.urlretrieve(ALPACA_URL, local)
    with open(local, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_sharegpt():
    from huggingface_hub import hf_hub_download

    print(f"  downloading {SHAREGPT_REPO}/{SHAREGPT_FILE} (~670 MB, cached after the first time)")
    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset",
                           cache_dir=os.path.join(CACHE_DIR, "hf"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def alpaca_pairs(raw):
    """(prompt text, completion text) per sample. input is appended when present, which is
    how the instruction is actually presented to a model."""
    for s in raw:
        instruction = s.get("instruction", "").strip()
        extra = s.get("input", "").strip()
        out = s.get("output", "").strip()
        if not instruction or not out:
            continue
        prompt = f"{instruction}\n\n{extra}" if extra else instruction
        yield prompt, out


def sharegpt_pairs(raw):
    """First human turn as the prompt, first model turn as the completion - the same sampling
    vLLM's benchmark does. Conversations with fewer than two turns carry no output length."""
    for conv in raw:
        turns = conv.get("conversations") or []
        if len(turns) < 2:
            continue
        prompt = (turns[0].get("value") or "").strip()
        out = (turns[1].get("value") or "").strip()
        if not prompt or not out:
            continue
        yield prompt, out


def build(name, pairs, tokenizer, max_samples):
    """Tokenise with the chat template the engine will actually see, so prompt_len is the
    number of tokens the KV cache really has to hold."""
    samples = []
    dropped_prompt = dropped_output = 0
    for prompt, completion in pairs:
        messages = [
            {"role": "system", "content": "You are a helpful and detailed AI assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(prompt_ids) > MAX_PROMPT_LEN:
            dropped_prompt += 1
            continue
        output_len = len(tokenizer.encode(completion, add_special_tokens=False))
        if output_len < 1 or output_len > MAX_NEW_TOKENS_GENERATED:
            dropped_output += 1
            continue
        samples.append({"prompt": prompt_ids, "prompt_len": len(prompt_ids), "output_len": output_len})
        if max_samples and len(samples) >= max_samples:
            break

    def stats(key):
        xs = sorted(s[key] for s in samples)
        n = len(xs)
        return {
            "mean": sum(xs) / n,
            "p50": xs[n // 2],
            "p90": xs[min(n - 1, int(0.90 * n))],
            "p99": xs[min(n - 1, int(0.99 * n))],
            "max": xs[-1],
        }

    meta = {
        "name": name,
        "num_samples": len(samples),
        "dropped_prompt_too_long": dropped_prompt,
        "dropped_output_too_long": dropped_output,
        "max_prompt_len": MAX_PROMPT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS_GENERATED,
        "prompt_len": stats("prompt_len"),
        "output_len": stats("output_len"),
    }
    return {"meta": meta, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="cap per dataset; a sweep never needs more than a few thousand")
    args = ap.parse_args()

    os.makedirs(WORKLOAD_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    targets = {"alpaca": (fetch_alpaca, alpaca_pairs), "sharegpt": (fetch_sharegpt, sharegpt_pairs)}
    todo = {k: v for k, v in targets.items()
            if args.force or not os.path.exists(os.path.join(WORKLOAD_DIR, f"{k}.json"))}
    if not todo:
        print("Both workloads already built; pass --force to rebuild.")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        for name, (fetch, pairs_fn) in todo.items():
            print(f"[{name}]")
            data = build(name, pairs_fn(fetch()), tokenizer, args.max_samples)
            out = os.path.join(WORKLOAD_DIR, f"{name}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"  {data['meta']['num_samples']} samples -> {out}")

    # The paper's own characterisation of the two datasets, which is the thing to check the
    # filtering against: "the ShareGPT dataset has 8.4x longer input prompts and 5.8x longer
    # outputs on average than the Alpaca dataset".
    metas = {}
    for name in targets:
        p = os.path.join(WORKLOAD_DIR, f"{name}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                metas[name] = json.load(f)["meta"]
    print("\n%-10s %8s %10s %8s %10s %8s" % ("dataset", "samples", "prompt_avg", "p99", "out_avg", "p99"))
    for name, m in metas.items():
        print("%-10s %8d %10.1f %8d %8.1f %10d" % (
            name, m["num_samples"], m["prompt_len"]["mean"], m["prompt_len"]["p99"],
            m["output_len"]["mean"], m["output_len"]["p99"]))
    if {"alpaca", "sharegpt"} <= set(metas):
        a, s = metas["alpaca"], metas["sharegpt"]
        print(f"\nShareGPT / Alpaca ratio: prompt {s['prompt_len']['mean'] / a['prompt_len']['mean']:.2f}x "
              f"(paper: 8.4x), output {s['output_len']['mean'] / a['output_len']['mean']:.2f}x (paper: 5.8x)")
        print("A ratio well below the paper's is a filtering artefact of our 1024-token caps and\n"
              "must be reported alongside the results, not quietly ignored.")


if __name__ == "__main__":
    main()
