"""Experiment driver for the PagedAttention replication. Resumable, stage by stage.

Every run writes its own file under results/runs/ and is never repeated: re-invoking this
script picks up exactly where it left off, which is what makes a multi-hour sweep survivable
on a laptop. Nothing is kept only in memory.

  python sweep.py --all            # everything, in order, skipping what is already done
  python sweep.py --stage e1       # just one stage
  python sweep.py --status         # what is done, what is left, how long the rest will take
  python sweep.py --all --force    # ignore existing results and redo

Stages:
  prepare  build the ShareGPT / Alpaca workloads (delegates to prepare_workload.py)
  build    build both engines
  e1       §6.2 serving curves: normalised latency vs request rate, 4 variants x 2 workloads
  e3       §7.1 attention kernel microbenchmark, paged vs contiguous addressing
  e4       §7.2 block size ablation, BLOCK_SIZE 1..256
  report   figures and the HTML report (delegates to report.py)

Two things this driver does that the paper did not have to: it interleaves variants rather
than running one to completion before the next, and it sleeps between runs. The paper used a
datacenter A100; this is a 30 W laptop that thermally throttles, so without interleaving the
variant that happens to run last carries a hot GPU and the comparison measures cooling.
"""

import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO_ROOT, "results")
RUNS = os.path.join(RESULTS, "runs")
PYTHON = sys.executable

# ---- E1: the serving sweep ----------------------------------------------------------------
# Four variants, all from two binaries; the three baseline ones differ only by an env var.
# base:max is the only one a real contiguous engine could implement, so it is the official
# comparison; pow2 and oracle are unreachable upper bounds and must always be labelled so.
VARIANTS = [
    ("paged", "paged_attention", {}),
    ("base:max", "base", {"ORCA_POLICY": "max"}),
    ("base:pow2", "base", {"ORCA_POLICY": "pow2"}),
    ("base:oracle", "base", {"ORCA_POLICY": "oracle"}),
]
WORKLOADS = ["alpaca", "sharegpt"]
# Throughput ceiling on this GPU is roughly 250 tok/s over ~190 output tokens per request,
# i.e. ~1.3 req/s, so this spans both sides of the knee.
RATES = [0.3, 0.6, 0.9, 1.2, 1.6, 2.0]
NUM_REQUESTS = 200
SEED = 0
# A single run must not be able to wedge the whole sweep. At the saturated rates a run is
# arrival-limited to ~11 min; past 25 the engine is not coming back.
MAX_WALL_S = 1500
COOLDOWN_S = 60

# ---- E4: block size ablation ---------------------------------------------------------------
# The block table scales as MAX_SEQUENCES * N_LAYERS * (MAX_TOKENS_PER_SEQUENCE / BLOCK_SIZE),
# which at BLOCK_SIZE=1 and 384 slots is 25 MiB and does not fit. The ablation therefore runs
# at a fixed, smaller slot count for *every* block size, so only one variable moves inside it.
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ABLATION_SEQUENCES = 64
# 1.2, not 0.9. Block size only affects how efficiently the KV cache is read, and at rate 0.9 the
# batch was 4.5 (Alpaca) / 10.1 (ShareGPT) - far from saturated, where each decode step is
# dominated by streaming all 2.30 GiB of weights and the KV path is a rounding error. 1.2 sits just
# below paged's capacity point: batch 26 on ShareGPT, so there is a lot of cache to read, while the
# queue stays near empty (0.13) so latency still reflects compute rather than waiting. A saturated
# rate would be just as useless in the other direction - queueing time would swamp the effect.
ABLATION_RATE = 1.2
# 200, not 60. A first pass at 60 requests was measurably underpowered: at rate 0.9 that is only
# ~67 s of arrivals, barely two request lifetimes, so the run never leaves its warm-up transient.
# Every one of those runs drifted 12-18% between its halves while the block-size effect being
# measured was 2-3% - the noise was larger than the signal. 200 matches E1 and gives ~6 lifetimes.
ABLATION_REQUESTS = 200


def run_key(stage, **kw):
    parts = [stage] + [f"{k}={v}" for k, v in sorted(kw.items())]
    return "_".join(parts).replace("/", "-").replace(":", "-")


def result_path(stage, **kw):
    return os.path.join(RUNS, run_key(stage, **kw) + ".json")


def done(stage, **kw):
    p = result_path(stage, **kw)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return not d.get("abandoned")
    except (json.JSONDecodeError, OSError):
        return False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, env=None, cwd=None, timeout=None):
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(cmd, env=full, cwd=cwd or REPO_ROOT, timeout=timeout)


# ============================ stages ============================

def stage_prepare(force):
    marker = os.path.join(REPO_ROOT, "workloads", "sharegpt.json")
    if os.path.exists(marker) and not force:
        log("prepare: workloads already built")
        return
    log("prepare: building workloads from ShareGPT + Alpaca")
    cmd = [PYTHON, "prepare_workload.py"] + (["--force"] if force else [])
    if sh(cmd).returncode != 0:
        sys.exit("prepare_workload.py failed")


def build_engine(directory, defines=None, label=""):
    """Configure and build one engine. defines are -D flags passed to cmake, used by the
    block-size ablation to vary BLOCK_SIZE and MAX_SEQUENCES without editing sources."""
    build_dir = os.path.join(REPO_ROOT, directory, "build")
    cmake = ["cmake", "..", "-G", "Ninja"] + [f"-D{d}" for d in (defines or [])]
    os.makedirs(build_dir, exist_ok=True)
    if sh(cmake, cwd=build_dir).returncode != 0:
        sys.exit(f"cmake failed for {directory} {label}")
    if sh(["ninja"], cwd=build_dir).returncode != 0:
        sys.exit(f"ninja failed for {directory} {label}")


def stage_build(force):
    for d in ("base", "paged_attention"):
        binary = os.path.join(REPO_ROOT, d, "build", "tiny-vllm")
        if os.path.exists(binary) and not force:
            log(f"build: {d} already built")
            continue
        log(f"build: {d}")
        build_engine(d)


def bench(out_path, engine_dir, workload, rate, num_requests, env, label, seed=SEED):
    cmd = [PYTHON, "bench.py", "--engine-dir", engine_dir, "--workload", workload,
           "--rate", str(rate), "--num-requests", str(num_requests), "--seed", str(seed),
           "--out", out_path, "--label", label, "--max-wall", str(MAX_WALL_S)]
    rc = sh(cmd, env=env, timeout=MAX_WALL_S + 900).returncode
    return rc == 0


def stage_e1(force):
    """Interleaved by rate, then by variant, so thermal drift is spread across variants
    instead of landing on whichever one went last."""
    todo = [(w, r, v) for w in WORKLOADS for r in RATES for v in VARIANTS
            if force or not done("e1", workload=w, rate=r, variant=v[0])]
    log(f"e1: {len(todo)} run(s) to go of {len(WORKLOADS) * len(RATES) * len(VARIANTS)}")
    for i, (workload, rate, (name, directory, env)) in enumerate(todo, 1):
        out = result_path("e1", workload=workload, rate=rate, variant=name)
        log(f"e1 [{i}/{len(todo)}] {workload} rate={rate} {name}")
        ok = bench(out, directory, workload, rate, NUM_REQUESTS, env, name)
        if not ok:
            log(f"  !! run did not complete cleanly; its file records why. Continuing.")
        if i < len(todo):
            time.sleep(COOLDOWN_S)  # let the laptop cool between runs


def stage_e3(force):
    """§7.1: the attention kernel on its own, paged block-table walk vs contiguous stride."""
    out = result_path("e3", kind="kernel")
    if done("e3", kind="kernel") and not force:
        log("e3: already done")
        return
    log("e3: building microbenchmarks")
    micro_build = os.path.join(REPO_ROOT, "micro", "build")
    os.makedirs(micro_build, exist_ok=True)
    if sh(["cmake", "..", "-G", "Ninja"], cwd=micro_build).returncode != 0:
        sys.exit("micro cmake failed")
    if sh(["ninja"], cwd=micro_build).returncode != 0:
        sys.exit("micro ninja failed")

    log("e3: measuring")
    rows = []
    for binary, mechanism in (("attn_bench_base", "contiguous"), ("attn_bench_paged", "paged")):
        proc = subprocess.run([os.path.join(micro_build, binary)], capture_output=True, text=True,
                              timeout=900)
        if proc.returncode != 0:
            sys.exit(f"{binary} failed:\n{proc.stderr}")
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f)
    log(f"e3: {len(rows)} measurements -> {out}")


def stage_e4(force):
    """§7.2: rebuild the paged engine per block size. Each build is immediately benchmarked
    and its result written before moving on, so an interrupted ablation resumes mid-sweep."""
    todo = [(b, w) for b in BLOCK_SIZES for w in WORKLOADS
            if force or not done("e4", block_size=b, workload=w)]
    log(f"e4: {len(todo)} run(s) to go of {len(BLOCK_SIZES) * len(WORKLOADS)}")
    current_block = None
    for i, (block, workload) in enumerate(todo, 1):
        if block != current_block:
            log(f"e4: building paged_attention with BLOCK_SIZE={block}, "
                f"MAX_SEQUENCES={ABLATION_SEQUENCES}")
            build_engine("paged_attention",
                         defines=[f"ABLATION_BLOCK_SIZE={block}",
                                  f"ABLATION_MAX_SEQUENCES={ABLATION_SEQUENCES}"],
                         label=f"block={block}")
            current_block = block
        out = result_path("e4", block_size=block, workload=workload)
        log(f"e4 [{i}/{len(todo)}] block={block} {workload}")
        bench(out, "paged_attention", workload, ABLATION_RATE, ABLATION_REQUESTS, {},
              f"paged:block{block}")
        time.sleep(COOLDOWN_S)
    if current_block is not None:
        log("e4: restoring the default paged_attention build")
        shutil.rmtree(os.path.join(REPO_ROOT, "paged_attention", "build"), ignore_errors=True)
        build_engine("paged_attention")


def stage_report(force):
    log("report: generating figures and HTML")
    if sh([PYTHON, "report.py"]).returncode != 0:
        sys.exit("report.py failed")


# ============================ status ============================

def stage_status(_force):
    total = pending = 0
    print(f"{'stage':8s} {'done':>6s} {'total':>6s}  detail")
    e1_all = [(w, r, v[0]) for w in WORKLOADS for r in RATES for v in VARIANTS]
    e1_done = [x for x in e1_all if done("e1", workload=x[0], rate=x[1], variant=x[2])]
    print(f"{'e1':8s} {len(e1_done):6d} {len(e1_all):6d}  serving curves")
    total += len(e1_all); pending += len(e1_all) - len(e1_done)

    e3_done = 1 if done("e3", kind="kernel") else 0
    print(f"{'e3':8s} {e3_done:6d} {1:6d}  attention kernel microbenchmark")
    total += 1; pending += 1 - e3_done

    e4_all = [(b, w) for b in BLOCK_SIZES for w in WORKLOADS]
    e4_done = [x for x in e4_all if done("e4", block_size=x[0], workload=x[1])]
    print(f"{'e4':8s} {len(e4_done):6d} {len(e4_all):6d}  block size ablation")
    total += len(e4_all); pending += len(e4_all) - len(e4_done)

    print(f"\n{total - pending}/{total} runs complete.")
    if pending:
        # ~6 min a run plus cooldown, measured on the gate runs
        print(f"Roughly {pending * 7 / 60:.1f} h of wall clock left. Re-run `python sweep.py --all` "
              f"to continue; finished runs are never redone.")
    # Quality, not just completion. A run whose latency drifted because the system was past
    # its capacity point is correct data - it is what makes the curve explode in the paper's
    # Fig 12 - so it must not be listed as a problem. See runqual.py.
    import runqual
    buckets = defaultdict(list)
    for name in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        try:
            with open(os.path.join(RUNS, name), "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "convergence" not in d:
            continue  # e3 and other non-serving results
        kind, why = runqual.classify(d)
        buckets[kind].append((name, why))
    if buckets:
        print("\nRun quality:")
        for kind in (runqual.STEADY, runqual.SATURATED, runqual.NOISY, runqual.BROKEN):
            if buckets[kind]:
                print(f"  {kind:10s} {len(buckets[kind]):3d}")
    attention = buckets[runqual.NOISY] + buckets[runqual.BROKEN]
    if attention:
        print(f"\n{len(attention)} run(s) need attention (saturated runs are NOT listed - "
              f"they are valid data):")
        for name, why in attention[:20]:
            print(f"  {name}\n    {why}")


STAGES = {"prepare": stage_prepare, "build": stage_build, "e1": stage_e1,
          "e3": stage_e3, "e4": stage_e4, "report": stage_report, "status": stage_status}
ORDER = ["prepare", "build", "e1", "e3", "e4", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", choices=list(STAGES),
                    help="run just this stage; repeatable")
    ap.add_argument("--all", action="store_true", help="run every stage in order")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--force", action="store_true", help="redo work that is already done")
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    if args.status:
        stage_status(False)
        return
    stages = ORDER if args.all else (args.stage or [])
    if not stages:
        ap.error("pass --all, --stage <name>, or --status")
    for name in stages:
        log(f"=== stage {name} ===")
        STAGES[name](args.force)
    log("done")


if __name__ == "__main__":
    main()
