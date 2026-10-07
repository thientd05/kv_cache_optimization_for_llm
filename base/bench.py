"""Serving benchmark for the KV-cache comparison, shaped after the one in the
PagedAttention paper (Kwon et al., SOSP '23, section 6.2).

The question that paper asks is not "how fast is one decode step" - it is "how many
requests per second can this server absorb before latency falls apart". That framing is
the whole point: a KV cache manager does not make arithmetic faster, it decides how many
sequences fit in memory at once, and concurrency is what turns into throughput on a
memory-bound decoder. So:

  * requests arrive as a Poisson process at a given rate, they are not dumped in a batch;
  * the headline metric is normalised latency - end-to-end request latency divided by the
    number of output tokens, averaged over requests - plotted against that rate;
  * output lengths vary per request and the server is not told them in advance, which is
    what makes reservation expensive and on-demand allocation cheap;
  * the engine's own telemetry (how many requests are batched, how much of the cache is
    holding real tokens) is recorded alongside, because it is the mechanism behind
    whatever the latency curve does.

The workload is deterministic given --seed, so every variant sees the identical sequence
of (prompt, output length, arrival time) triples.

usage: bench.py --rate 0.5 --num-requests 60 [--seed 0] [--out results.json]
"""

import argparse
import atexit
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct"))
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")

# Output-length distribution. ShareGPT, the trace the paper uses, has output lengths that
# are short in the median and heavily right-skewed, which is exactly the shape that
# punishes reservation: reserve for the tail and you waste most of the cache on the median.
# A lognormal clipped to the engine's budget reproduces that shape without shipping a
# dataset. MEDIAN/SIGMA are chosen so the mean lands near 150 tokens with a tail that
# reaches the 512-token cap.
OUTPUT_LEN_MEDIAN = 110
OUTPUT_LEN_SIGMA = 0.85
OUTPUT_LEN_MIN = 16
OUTPUT_LEN_MAX = 512

STALL_TIMEOUT_S = float(os.environ.get("STALL_TIMEOUT_S", "300"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True,
                    help="Poisson arrival rate in requests/second; 0 means all at once")
    ap.add_argument("--num-requests", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--output-len", type=int, default=0,
                    help="give every request this fixed output budget instead of sampling "
                         "one; the sensitivity case where nothing about length is unknown")
    return ap.parse_args()


def build_workload(prompts, n, seed, fixed_len=0):
    """The (prompt, output length) pairs, identical for every variant."""
    rng = random.Random(seed)
    work = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        if fixed_len:
            out_len = fixed_len
        else:
            out_len = int(round(OUTPUT_LEN_MEDIAN * pow(2.718281828, rng.gauss(0.0, OUTPUT_LEN_SIGMA))))
            out_len = max(OUTPUT_LEN_MIN, min(OUTPUT_LEN_MAX, out_len))
        work.append((prompt, out_len))
    return work


def main():
    args = parse_args()

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        sys.exit(f"No prompts in {PROMPTS_FILE}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    workload = build_workload(prompts, args.num_requests, args.seed, args.output_len)
    # Tokenise before the clock starts: the arrival process must not inherit the
    # tokeniser's jitter, and every variant must send byte-identical token lists.
    encoded = []
    for prompt, out_len in workload:
        messages = [
            {"role": "system", "content": "You are a helpful and detailed AI assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded.append((tokenizer.encode(text, add_special_tokens=False), out_len))

    engine_path = "./build/tiny-vllm"
    if not os.path.exists(engine_path):
        sys.exit(f"Engine not found at {engine_path}; build it first.")

    env = dict(os.environ)
    env["IGNORE_EOS"] = "1"  # every request emits exactly its budget, in every variant
    engine = subprocess.Popen(
        [engine_path, MODEL_DIR], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
    )

    stopped = threading.Lock()
    done_flag = {"stopped": False}

    def shutdown():
        with stopped:
            if done_flag["stopped"]:
                return
            done_flag["stopped"] = True
        if engine.poll() is None:
            try:
                if engine.stdin and not engine.stdin.closed:
                    engine.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            engine.terminate()
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine.kill()

    atexit.register(shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (shutdown(), sys.exit(128 + s)))

    state_lock = threading.Lock()
    reqs = {}
    engine_config = {}
    steps = []
    preempt_events = []
    ready = threading.Event()
    finished = threading.Event()
    last_update = [time.monotonic()]

    def reader():
        for line in iter(engine.stdout.readline, ""):
            if not line:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            if kind == "engine_config":
                engine_config.update(data)
                ready.set()
                continue
            with state_lock:
                last_update[0] = time.monotonic()
                if kind == "step":
                    steps.append(data)
                    continue
                entry = reqs.get(data.get("id"))
                if entry is None:
                    continue
                if kind == "preempted":
                    preempt_events.append(data)
                    entry["preemptions"] += 1
                elif kind == "prefill_stats":
                    # a resumed sequence prefills again; the first one is the TTFT
                    entry.setdefault("prefill_ms", data.get("time_ms"))
                elif data.get("done"):
                    if entry["t_done"] is None:
                        entry["t_done"] = time.perf_counter()
                        entry["engine_tokens"] = data.get("tokens", 0)
                        entry["error"] = data.get("error")
                        if all(r["t_done"] is not None for r in reqs.values()) and submit_done.is_set():
                            finished.set()
                else:
                    entry["tokens"] += 1
                    # order-sensitive rolling hash of the generated ids: two builds that
                    # agree on this produced the same text, not just the same token count
                    entry["checksum"] = (entry["checksum"] * 1000003 + data.get("tok", 0)) & 0xFFFFFFFF
                    if entry["t_first"] is None:
                        entry["t_first"] = time.perf_counter()

    threading.Thread(target=reader, daemon=True).start()
    if not ready.wait(timeout=600):
        shutdown()
        sys.exit("engine never reported its configuration")

    submit_done = threading.Event()
    rng = random.Random(args.seed + 977)

    def submit():
        for i, (tokens, out_len) in enumerate(encoded):
            if args.rate > 0 and i > 0:
                time.sleep(rng.expovariate(args.rate))
            with state_lock:
                reqs[i] = {"id": i, "out_budget": out_len, "prompt_tokens": len(tokens),
                           "t_arrival": time.perf_counter(), "t_first": None, "t_done": None,
                           "tokens": 0, "engine_tokens": 0, "preemptions": 0, "error": None, "checksum": 0,
                           "prefill_ms": None}
            try:
                engine.stdin.write(f"{i} {out_len} " + " ".join(map(str, tokens)) + "\n")
                engine.stdin.flush()
            except (BrokenPipeError, OSError):
                break
        submit_done.set()

    t0 = time.perf_counter()
    threading.Thread(target=submit, daemon=True).start()

    abandoned = None
    while True:
        if finished.wait(timeout=1.0):
            break
        with state_lock:
            pending = sum(1 for r in reqs.values() if r["t_done"] is None)
            quiet = time.monotonic() - last_update[0]
            submitted = len(reqs)
        if submit_done.is_set() and pending == 0 and submitted == args.num_requests:
            break
        if engine.poll() is not None:
            abandoned = f"engine exited with code {engine.returncode}"
            break
        if quiet > STALL_TIMEOUT_S:
            abandoned = f"engine silent for {quiet:.0f}s with {pending} request(s) unfinished"
            break
        sys.stderr.write(f"\r  {submitted}/{args.num_requests} sent, {pending} in flight, "
                         f"{time.perf_counter() - t0:7.1f}s")
        sys.stderr.flush()
    wall = time.perf_counter() - t0
    sys.stderr.write("\n")
    shutdown()

    with state_lock:
        records = [r for r in reqs.values() if r["t_done"] is not None and not r["error"]]
        failures = [r for r in reqs.values() if r["t_done"] is None or r["error"]]

    def pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]

    latencies = [r["t_done"] - r["t_arrival"] for r in records]
    # The paper's y axis: seconds of end-to-end latency per output token. Dividing by the
    # output length is what makes requests of wildly different lengths comparable.
    norm = [(r["t_done"] - r["t_arrival"]) / max(1, r["tokens"]) for r in records]
    ttft = [r["t_first"] - r["t_arrival"] for r in records if r["t_first"] is not None]
    out_tokens = sum(r["tokens"] for r in records)

    result = {
        "label": args.label,
        "rate": args.rate,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "output_len_fixed": args.output_len,
        "engine_config": engine_config,
        "wall_s": wall,
        "completed": len(records),
        "failed": len(failures),
        "normalized_latency_s_per_token": sum(norm) / len(norm) if norm else None,
        "normalized_latency_p90": pct(norm, 90),
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "latency_p90_s": pct(latencies, 90),
        "ttft_mean_s": sum(ttft) / len(ttft) if ttft else None,
        "ttft_p90_s": pct(ttft, 90),
        "throughput_req_s": len(records) / wall if wall > 0 else 0.0,
        "throughput_tok_s": out_tokens / wall if wall > 0 else 0.0,
        "output_tokens": out_tokens,
        "preemptions": sum(r["preemptions"] for r in reqs.values()),
        "running_mean": (sum(s["running"] for s in steps) / len(steps)) if steps else 0.0,
        "running_max": max((s["running"] for s in steps), default=0),
        "waiting_mean": (sum(s["waiting"] for s in steps) / len(steps)) if steps else 0.0,
        "steps": steps,
        "requests": [{k: v for k, v in r.items() if k != "prompt"} for r in reqs.values()],
        "abandoned": abandoned,
    }

    print(json.dumps({k: v for k, v in result.items() if k not in ("steps", "requests")}, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f)
    if abandoned:
        sys.exit(f"benchmark abandoned: {abandoned}")


if __name__ == "__main__":
    main()
