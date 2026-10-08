"""Serving benchmark for the KV-cache comparison, shaped after the one in the PagedAttention
paper (Kwon et al., SOSP '23, §6.2).

The question that paper asks is not "how fast is one decode step" - it is "how many requests
per second can this server absorb before latency falls apart". That framing is the whole
point: a KV cache manager does not make arithmetic faster, it decides how many sequences fit
in memory at once, and concurrency is what turns into throughput on a memory-bound decoder.
So:

  * requests arrive as a Poisson process at a given rate, they are not dumped in a batch;
  * the headline metric is normalised latency - end-to-end request latency divided by the
    number of output tokens, averaged over requests - plotted against that rate;
  * prompt and output lengths come from the real ShareGPT / Alpaca traces, as *pairs*, and
    the server is never told the output length in advance;
  * the engine's own telemetry (how many requests are batched, how much of the cache holds
    real tokens) is recorded alongside, because it is the mechanism behind whatever the
    latency curve does.

The workload is deterministic given --seed, so every variant sees the identical sequence of
(prompt, output length, arrival time) triples. Run prepare_workload.py first.

usage: python bench.py --engine-dir base --workload sharegpt --rate 0.5 [--num-requests 200]
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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct"))
WORKLOAD_DIR = os.path.join(REPO_ROOT, "workloads")

STALL_TIMEOUT_S = float(os.environ.get("STALL_TIMEOUT_S", "600"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True, help="base or paged_attention")
    ap.add_argument("--workload", default="sharegpt", choices=["alpaca", "sharegpt"])
    ap.add_argument("--rate", type=float, required=True,
                    help="Poisson arrival rate in requests/second; 0 means all at once")
    ap.add_argument("--num-requests", type=int, default=200,
                    help="the paper runs 1-hour traces; see the convergence check below for "
                         "what we preserve instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--max-wall", type=float, default=0,
                    help="abort the run after this many seconds (0 = no limit)")
    return ap.parse_args()


def gpu_state():
    """SM clock, temperature and power. The paper ran on a datacenter A100; this is a 30 W
    laptop that throttles, so every run records the thermal state it ran under and sweep.py
    interleaves variants. Without that, whichever variant runs last carries the hot GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, temp, power = (x.strip() for x in out.split(","))
        return {"sm_clock_mhz": float(sm), "temp_c": float(temp), "power_w": float(power)}
    except Exception:
        return {}


def load_workload(name, num_requests, seed):
    path = os.path.join(WORKLOAD_DIR, f"{name}.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found; run `python prepare_workload.py` first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    # Sampled, not taken in order: the file is in dataset order and the head of either
    # dataset is not representative of it. The seed makes it identical across variants.
    rng = random.Random(seed)
    picked = [samples[rng.randrange(len(samples))] for _ in range(num_requests)]
    return picked, data["meta"]


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def main():
    args = parse_args()
    requests_spec, workload_meta = load_workload(args.workload, args.num_requests, args.seed)

    engine_dir = os.path.abspath(args.engine_dir)
    engine_path = os.path.join(engine_dir, "build", "tiny-vllm")
    if not os.path.exists(engine_path):
        sys.exit(f"Engine not found at {engine_path}; build it first.")

    env = dict(os.environ)
    env["IGNORE_EOS"] = "1"  # every request emits exactly its budget, in every variant
    engine = subprocess.Popen(
        [engine_path, MODEL_DIR], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env, cwd=engine_dir,
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
    ready = threading.Event()
    finished = threading.Event()
    submit_done = threading.Event()
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
                    entry["preemptions"] += 1
                elif kind == "prefill_stats":
                    # a resumed sequence prefills again; the first one is the TTFT
                    entry.setdefault("prefill_ms", data.get("time_ms"))
                elif data.get("done"):
                    if entry["t_done"] is None:
                        entry["t_done"] = time.perf_counter()
                        entry["engine_tokens"] = data.get("tokens", 0)
                        entry["error"] = data.get("error")
                        if submit_done.is_set() and all(r["t_done"] is not None for r in reqs.values()):
                            finished.set()
                else:
                    entry["tokens"] += 1
                    # order-sensitive rolling hash of the generated ids: two builds that
                    # agree on this produced the same text, not just the same token count
                    entry["checksum"] = (entry["checksum"] * 1000003 + data.get("tok", 0)) & 0xFFFFFFFF
                    if entry["t_first"] is None:
                        entry["t_first"] = time.perf_counter()

    threading.Thread(target=reader, daemon=True).start()
    if not ready.wait(timeout=900):
        shutdown()
        sys.exit("engine never reported its configuration")

    rng = random.Random(args.seed + 977)

    def submit():
        for i, spec in enumerate(requests_spec):
            if args.rate > 0 and i > 0:
                time.sleep(rng.expovariate(args.rate))
            if done_flag["stopped"]:
                break
            with state_lock:
                reqs[i] = {"id": i, "out_budget": spec["output_len"], "prompt_tokens": spec["prompt_len"],
                           "t_arrival": time.perf_counter(), "t_first": None, "t_done": None,
                           "tokens": 0, "engine_tokens": 0, "preemptions": 0, "error": None,
                           "checksum": 0, "prefill_ms": None}
            try:
                engine.stdin.write(f"{i} {spec['output_len']} " + " ".join(map(str, spec["prompt"])) + "\n")
                engine.stdin.flush()
            except (BrokenPipeError, OSError):
                break
        submit_done.set()

    gpu_before = gpu_state()
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
        elapsed = time.perf_counter() - t0
        if submit_done.is_set() and pending == 0 and submitted == args.num_requests:
            break
        if engine.poll() is not None:
            abandoned = f"engine exited with code {engine.returncode}"
            break
        if quiet > STALL_TIMEOUT_S:
            abandoned = f"engine silent for {quiet:.0f}s with {pending} request(s) unfinished"
            break
        if args.max_wall and elapsed > args.max_wall:
            abandoned = f"exceeded --max-wall {args.max_wall:.0f}s with {pending} unfinished"
            break
        sys.stderr.write(f"\r  {submitted}/{args.num_requests} sent, {pending} in flight, {elapsed:7.1f}s")
        sys.stderr.flush()
    wall = time.perf_counter() - t0
    gpu_after = gpu_state()
    sys.stderr.write("\n")
    shutdown()

    with state_lock:
        records = [r for r in reqs.values() if r["t_done"] is not None and not r["error"]]
        failures = [r for r in reqs.values() if r["t_done"] is None or r["error"]]

    # The paper's y axis: seconds of end-to-end latency per output token. Dividing by the
    # output length is what makes requests of wildly different lengths comparable.
    def norm_of(r):
        return (r["t_done"] - r["t_arrival"]) / max(1, r["tokens"])

    records.sort(key=lambda r: r["t_arrival"])
    norm = [norm_of(r) for r in records]
    latencies = [r["t_done"] - r["t_arrival"] for r in records]
    out_tokens = sum(r["tokens"] for r in records)

    # Steady state check, standing in for the paper's 1-hour traces. We cannot match the wall
    # clock (an A100 pushes ~100x more requests through an hour than this GPU), so instead of
    # assuming the mean has settled we measure whether it has: split the completed requests by
    # arrival order and compare the halves. Over 10% apart means the run was too short.
    half = len(norm) // 2
    first_half = sum(norm[:half]) / half if half else 0.0
    second_half = sum(norm[half:]) / (len(norm) - half) if len(norm) - half else 0.0
    drift = abs(second_half - first_half) / max(first_half, 1e-9) if half else 1.0

    # Token accounting. These must hold at any concurrency, unlike the checksum, which only
    # matches across builds when they batch the same number of sequences - see CLAUDE.md.
    budget_mismatch = [r["id"] for r in records if r["tokens"] != r["out_budget"]]
    engine_mismatch = [r["id"] for r in records if r["engine_tokens"] != r["tokens"]]

    result = {
        "label": args.label or f"{os.path.basename(engine_dir)}:{os.environ.get('ORCA_POLICY', '-')}",
        "engine_dir": os.path.basename(engine_dir),
        "orca_policy": os.environ.get("ORCA_POLICY", ""),
        "workload": args.workload,
        "workload_meta": workload_meta,
        "rate": args.rate,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "engine_config": engine_config,
        "wall_s": wall,
        "completed": len(records),
        "failed": len(failures),
        "normalized_latency_s_per_token": sum(norm) / len(norm) if norm else None,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "throughput_req_s": len(records) / wall if wall > 0 else 0.0,
        "output_tokens": out_tokens,
        "preemptions": sum(r["preemptions"] for r in reqs.values()),
        "running_mean": (sum(s["running"] for s in steps) / len(steps)) if steps else 0.0,
        "running_max": max((s["running"] for s in steps), default=0),
        "waiting_mean": (sum(s["waiting"] for s in steps) / len(steps)) if steps else 0.0,
        "convergence": {"first_half": first_half, "second_half": second_half, "drift": drift,
                        "converged": drift < 0.10 and len(norm) >= 20},
        "invariants": {"budget_mismatch": budget_mismatch, "engine_mismatch": engine_mismatch,
                       "ok": not budget_mismatch and not engine_mismatch and not failures},
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "steps": steps,
        "requests": records + failures,
        "abandoned": abandoned,
    }

    summary = {k: v for k, v in result.items()
               if k not in ("steps", "requests", "workload_meta", "engine_config")}
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f)
    if abandoned:
        sys.exit(f"benchmark abandoned: {abandoned}")


if __name__ == "__main__":
    main()
