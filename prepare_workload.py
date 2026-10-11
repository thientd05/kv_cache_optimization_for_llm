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
import random
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct"))
WORKLOAD_DIR = os.path.join(REPO_ROOT, "workloads")
CACHE_DIR = os.path.join(REPO_ROOT, "workloads", ".cache")

# Must match src/config.h (shared block).
#
# What happens to a sample that does not fit is a per-workload decision, not a global one:
#
#   drop (§6.2 workloads)  truncating would squash the right tail of the length distribution,
#                          and the right tail is precisely what punishes a reservation-based
#                          cache. Worse, a truncated sample piles up exactly at the cap, which
#                          flatters Orca (Max) - its waste is the gap between 2048 and the real
#                          length, and piling samples at the cap closes that gap artificially.
#   truncate_tail (§6.5)   the paper's own rule for the chatbot workload: "we cut the prompt to
#                          the last 1024 tokens". Here the pile-up at the cap is not an artefact,
#                          it IS the regime under test - see sharegpt_chat_pairs().
#
# An output longer than the cap is always dropped: that is the engine's real generation ceiling,
# not a modelling choice.
MAX_PROMPT_LEN = 1024
MAX_NEW_TOKENS_GENERATED = 1024
DROP, TRUNCATE_TAIL = "drop", "truncate_tail"

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


_sharegpt_cache = None


def fetch_sharegpt():
    """Memoised: two workloads are built from this one file (§6.2 basic sampling and §6.5
    chatbot), and re-parsing 670 MB of JSON for the second is a minute of nothing."""
    global _sharegpt_cache
    if _sharegpt_cache is not None:
        return _sharegpt_cache
    from huggingface_hub import hf_hub_download

    print(f"  downloading {SHAREGPT_REPO}/{SHAREGPT_FILE} (~670 MB, cached after the first time)")
    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset",
                           cache_dir=os.path.join(CACHE_DIR, "hf"))
    with open(path, "r", encoding="utf-8") as f:
        _sharegpt_cache = json.load(f)
    return _sharegpt_cache


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


# §6.4's translation workload: many requests that all begin with the same few-shot prefix.
#
# Substitution, and it has a direction. The paper uses WMT16 de-en news; on the Hub that is
# §6.4's own corpus. The paper uses "the translation workload of WMT16 German-to-English", and
# this is that corpus: wmt/wmt16, config de-en, test split - the same 2999 sentence pairs.
#
# It arrives as parquet, which is why an earlier version of this file reached for Helsinki-NLP's
# Tatoeba TSV instead and documented the substitution. That repo turned out to be gated (403 even
# with a token, access has to be granted per-account), so the choice became "take a parquet reader
# or take no §6.4 at all". pyarrow it is - and the consolation is that the deviation is gone: this
# is the paper's dataset, not a stand-in, so no §6.4 number needs an asterisk about sentence
# length any more.
WMT16_REPO = "wmt/wmt16"
WMT16_FILE = "de-en/test-00000-of-00001.parquet"
# News sentences run long; the band keeps the ones that make a request whose body is comparable
# to the shared prefix, so prefix sharing is neither flattered nor hidden.
TRANSLATE_MIN_CHARS, TRANSLATE_MAX_CHARS = 60, 220

# The prefix every request shares. Fixed text, not drawn per request - that is the whole point
# of the experiment, and drawing it would quietly turn this back into an ordinary workload.
TRANSLATE_HEADER = "Translate the following English sentences into German.\n\n"

_wmt16_cache = None


def fetch_wmt16():
    global _wmt16_cache
    if _wmt16_cache is not None:
        return _wmt16_cache
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print(f"  downloading {WMT16_REPO}/{WMT16_FILE}")
    path = hf_hub_download(WMT16_REPO, WMT16_FILE, repo_type="dataset",
                           cache_dir=os.path.join(CACHE_DIR, "hf"))
    # one column, "translation", holding {"de": ..., "en": ...} per row
    table = pq.read_table(path)
    rows = []
    for item in table.column("translation").to_pylist():
        de = (item.get("de") or "").strip()
        en = (item.get("en") or "").strip()
        if not de or not en:
            continue
        if not (TRANSLATE_MIN_CHARS <= len(de) <= TRANSLATE_MAX_CHARS):
            continue
        rows.append((de, en))
    _wmt16_cache = rows
    return rows


def translation_pairs(raw, shots):
    """(prompt, completion) where every prompt starts with the same `shots` worked examples.

    The examples are taken from the head of the corpus and then excluded from the requests, so
    no request is ever asked to translate a sentence it was just shown the answer to.
    """
    if len(raw) <= shots:
        return
    examples = raw[:shots]
    # English -> German, the paper's direction ("the WMT16 English-to-German translation
    # dataset"). The corpus rows are (de, en), so the English side is the source here.
    prefix = TRANSLATE_HEADER + "".join(
        f"English: {en}\nGerman: {de}\n\n" for de, en in examples)
    for de, en in raw[shots:]:
        yield prefix + f"English: {en}\nGerman:", de


def sharegpt_chat_pairs(raw, rng=None):
    """§6.5's chatbot workload, which is a different thing from §6.2's basic sampling even
    though it comes from the same dataset.

    The paper builds each request by "concatenating the chatting history and the last user query
    into a prompt", then cuts it to the last 1024 tokens. That is what makes *most* prompts land
    exactly at the cap - and that regime is the whole point, because there:

        "Due to the buddy allocation algorithm, the Orca baselines reserve the space for 1024
        tokens for the request outputs, regardless of how they predict the output lengths. For
        this reason, the three Orca baselines behave similarly."

    i.e. once prompts are long enough, predicting the output length well stops helping at all,
    which is the strongest form of the paper's argument. sharegpt_pairs() takes only the first
    exchange and so can never produce it.

    Round 0 is excluded on purpose: it carries no history and would just reproduce
    sharegpt_pairs(). The round is drawn with a fixed seed so the workload is reproducible.
    """
    rng = rng or random.Random(20251009)
    model_roles = ("gpt", "chatgpt", "bard", "bing", "assistant")
    for conv in raw:
        turns = conv.get("conversations") or []
        rounds = [i for i in range(1, len(turns) - 1)
                  if turns[i].get("from") == "human"
                  and turns[i + 1].get("from") in model_roles]
        if not rounds:
            continue
        k = rng.choice(rounds)
        history = []
        for t in turns[:k + 1]:
            value = (t.get("value") or "").strip()
            if value:
                speaker = "Human" if t.get("from") == "human" else "Assistant"
                history.append(f"{speaker}: {value}")
        out = (turns[k + 1].get("value") or "").strip()
        if len(history) < 3 or not out:
            continue  # at least one full previous round plus the new query
        yield "\n\n".join(history), out


def build(name, pairs, tokenizer, max_samples, on_prompt_too_long=DROP):
    """Tokenise with the chat template the engine will actually see, so prompt_len is the
    number of tokens the KV cache really has to hold."""
    samples = []
    dropped_prompt = dropped_output = truncated_prompt = 0
    bos = getattr(tokenizer, "bos_token_id", None)
    for prompt, completion in pairs:
        messages = [
            {"role": "system", "content": "You are a helpful and detailed AI assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(prompt_ids) > MAX_PROMPT_LEN:
            if on_prompt_too_long == DROP:
                dropped_prompt += 1
                continue
            # Keep the tail, which is what the paper does. This cuts into the chat template's
            # opening header - the kept span starts mid-conversation with no system block. That
            # is a real consequence of the rule and is left visible rather than papered over;
            # the one concession is restoring BOS, because a sequence that does not start with
            # it is outside anything the model was trained on.
            prompt_ids = prompt_ids[-MAX_PROMPT_LEN:]
            if bos is not None and prompt_ids[0] != bos:
                prompt_ids[0] = bos
            truncated_prompt += 1
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

    # §6.4 needs to know how long the prefix every request shares actually is, in tokens, and
    # the honest way to get that is to measure it rather than to assert it: the longest common
    # token prefix over the whole set. For an ordinary workload this comes out at the handful of
    # tokens the chat template opens with; for the translation workloads it is the instruction
    # plus the worked examples, which is the thing being shared.
    shared_prefix = 0
    if samples:
        first = samples[0]["prompt"]
        shared_prefix = len(first)
        for sample in samples[1:]:
            ids = sample["prompt"]
            limit = min(shared_prefix, len(ids))
            i = 0
            while i < limit and ids[i] == first[i]:
                i += 1
            shared_prefix = i
            if shared_prefix == 0:
                break

    meta = {
        "name": name,
        "shared_prefix_tokens": shared_prefix,
        "num_samples": len(samples),
        "prompt_policy": on_prompt_too_long,
        "dropped_prompt_too_long": dropped_prompt,
        "truncated_prompt_to_cap": truncated_prompt,
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

    targets = {
        "alpaca": (fetch_alpaca, alpaca_pairs, DROP),
        "sharegpt": (fetch_sharegpt, sharegpt_pairs, DROP),
        # §6.5. Same dataset as "sharegpt", different workload: history-concatenated prompts
        # cut to the cap rather than dropped. See sharegpt_chat_pairs().
        "sharegpt_chat": (fetch_sharegpt, sharegpt_chat_pairs, TRUNCATE_TAIL),
        # §6.4. Two prefix lengths, matching the paper's 1-example and 5-example settings.
        "translate_1shot": (fetch_wmt16, lambda r: translation_pairs(r, 1), DROP),
        "translate_5shot": (fetch_wmt16, lambda r: translation_pairs(r, 5), DROP),
    }
    todo = {k: v for k, v in targets.items()
            if args.force or not os.path.exists(os.path.join(WORKLOAD_DIR, f"{k}.json"))}
    if not todo:
        print("Every workload is already built; pass --force to rebuild.")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        for name, (fetch, pairs_fn, policy) in todo.items():
            print(f"[{name}]")
            data = build(name, pairs_fn(fetch()), tokenizer, args.max_samples, policy)
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
    # p50 is here because it is the gate for the chatbot workload: §6.5 only reproduces if most
    # prompts sit at the cap. A median near 150 means the history concatenation silently did
    # nothing and the workload is just "sharegpt" under another name.
    print("\n%-14s %8s %10s %7s %7s %9s %7s %7s  %s" % (
        "dataset", "samples", "prompt_avg", "p50", "p99", "out_avg", "p50", "p99", "cap policy"))
    for name, m in metas.items():
        policy = m.get("prompt_policy", "drop")
        extra = (f"truncated {m.get('truncated_prompt_to_cap', 0)}" if policy != "drop"
                 else f"dropped {m.get('dropped_prompt_too_long', 0)}")
        print("%-14s %8d %10.1f %7d %7d %9.1f %7d %7d  %s" % (
            name, m["num_samples"], m["prompt_len"]["mean"], m["prompt_len"]["p50"],
            m["prompt_len"]["p99"], m["output_len"]["mean"], m["output_len"]["p50"],
            m["output_len"]["p99"], f"{policy}, {extra}"))
    chat = metas.get("sharegpt_chat")
    if chat:
        p50 = chat["prompt_len"]["p50"]
        ok = p50 >= 0.9 * MAX_PROMPT_LEN
        print(f"\n[§6.5 gate] sharegpt_chat prompt p50 = {p50} "
              f"({'OK' if ok else 'FAIL'}; needs >= {int(0.9 * MAX_PROMPT_LEN)})")
        if not ok:
            print("  Most prompts are NOT at the cap, so the regime where all three Orca\n"
                  "  policies collapse onto each other is absent. Check the history\n"
                  "  concatenation before running the sweep on this workload.")
    if {"alpaca", "sharegpt"} <= set(metas):
        a, s = metas["alpaca"], metas["sharegpt"]
        print(f"\nShareGPT / Alpaca ratio: prompt {s['prompt_len']['mean'] / a['prompt_len']['mean']:.2f}x "
              f"(paper: 8.4x), output {s['output_len']['mean'] / a['output_len']['mean']:.2f}x (paper: 5.8x)")
        print("A ratio well below the paper's is a filtering artefact of our 1024-token caps and\n"
              "must be reported alongside the results, not quietly ignored.")


if __name__ == "__main__":
    main()
