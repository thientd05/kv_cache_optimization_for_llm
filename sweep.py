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
WORKLOADS = ["alpaca", "sharegpt", "sharegpt_chat"]
# One grid per workload, because the two saturate an order of magnitude apart. The original
# single grid was picked from an estimated ceiling of ~250 tok/s -> ~1.3 req/s; measured, this
# GPU peaks at 154 tok/s on ShareGPT and 121 tok/s on Alpaca, i.e. a real request ceiling near
# 0.8 req/s and 2.2 req/s respectively. One grid therefore cannot serve both: 2.0 is far past
# the knee for ShareGPT and still short of it for Alpaca.
#
# The points added over the original [0.3 .. 2.0]:
#   sharegpt 0.15  an idle anchor. At 0.3 base:max is already 15% above the other three and its
#                  running_max is pinned at the memory ceiling of 13, so the "unloaded" value the
#                  2x capacity threshold is built from is itself degraded - which pins base:max's
#                  capacity point to the bottom of the grid and makes the headline ratio a bound
#                  rather than a measurement.
#   sharegpt 0.45  resolution where the curve is steepest: base:max jumps 11x in normalised
#                  latency (0.164 -> 1.807) between 0.3 and 0.6, so the entire knee falls in a
#                  gap with no measured point in it.
#   alpaca 2.5,3.0 Alpaca never saturates inside the old grid (at 2.0 paged still sustains
#                  1.84 req/s of 2.0 offered). Without a saturated point there is no operating
#                  point with real memory pressure, which is why the block-size ablation cannot
#                  reproduce the paper's "large blocks hurt short sequences" result.
#   sharegpt_chat  §6.5. Prompts are history-concatenated and sit at the 1024-token cap, so a
#                  request is roughly twice the work of a plain ShareGPT one and the knee lands
#                  correspondingly lower. The grid is shifted down rather than stretched; the
#                  two lowest points dominate the stage's wall clock (0.1 req/s is 2000 s of
#                  arrivals alone) but dropping them would re-create the left-censoring this
#                  plan exists to remove.
RATES = {
    "alpaca":        [0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0],
    "sharegpt":      [0.15, 0.3, 0.45, 0.6, 0.9, 1.2, 1.6, 2.0],
    # 0.24 and 0.27 were added after the first pass, and for the reason plan 3 exists: all
    # four variants crossed the "2x idle" capacity threshold inside the 0.2 -> 0.3 gap, so
    # every capacity point snapped to 0.2 and the headline ratio would have come out 1.00x
    # while latency at a matched 0.3 req/s differed 3.4x. The measured throughput ceilings
    # sit at 0.234 (base:max) and 0.270 (paged) req/s, so these two rates land between them
    # and separate the ceilings instead of hiding them in one bin.
    "sharegpt_chat": [0.1, 0.2, 0.24, 0.27, 0.3, 0.45, 0.6, 0.9],
}
NUM_REQUESTS = 200
SEED = 0
COOLDOWN_S = 60


# Slowest req/s a run of each workload can sustain, measured on the BASELINE side - the deadline
# has to cover the slow variant or it truncates the losing one. sharegpt/alpaca are read off the
# 64 e1 runs already on disk; sharegpt_chat is derived from the paged probe (100 requests in 427 s
# = 0.23 req/s) divided by the concurrency ratio the pool forces on the baseline (13 slots of 2048
# reserved tokens, against paged's measured 34), hence ~0.09; the translate pair from the capacity
# probe. A missing entry just means no service floor, which is the old behaviour.
SLOWEST_SERVICE_RATE = {
    "alpaca": 0.6,
    "sharegpt": 0.25,
    "sharegpt_chat": 0.09,
    "translate_1shot": 1.0,
    "translate_5shot": 0.4,
}


def max_wall_for(rate, num_requests=NUM_REQUESTS, workload=None):
    """Deadline for one run. Generous on purpose; it exists to stop a wedged run, not to bound
    a slow one, and a genuinely hung engine is caught by STALL_TIMEOUT_S (600 s with no output).

    A correction worth keeping, because the wrong model is the intuitive one: wall clock at
    saturation is NOT "arrivals plus drain". Arrivals and service overlap, so it comes out at
    num_requests / achieved_throughput. Measured on sharegpt_chat with base:max, which is the
    slowest variant on the heaviest workload:

        rate 0.10 -> wall 1990 s   (arrival-limited: 200/0.1 = 2000)
        rate 0.20 -> wall 1013 s   (arrival-limited)
        rate 0.30 -> wall  855 s   (throughput-limited: 200/0.234)
        rate 0.45 -> wall  849 s   (throughput-limited)

    Past the capacity point the wall clock FALLS as the rate rises, because the arrival floor
    stops binding. The old flat 1500 s floor was therefore already sufficient everywhere. The
    service term below is kept only as slack - it costs nothing when a run finishes early - but
    it was added on a wrong premise and nothing depends on it."""
    arrivals = num_requests / rate if rate > 0 else 0.0
    slowest = SLOWEST_SERVICE_RATE.get(workload or "", 0.0)
    service = num_requests / slowest if slowest > 0 else 0.0
    return max(1500.0, arrivals * 1.3 + 600.0, arrivals + service * 1.3 + 600.0)


# ---- E4: block size ablation ---------------------------------------------------------------
# The block table scales as MAX_SEQUENCES * N_LAYERS * (MAX_TOKENS_PER_SEQUENCE / BLOCK_SIZE),
# which at BLOCK_SIZE=1 and 384 slots is 25 MiB and does not fit. The ablation therefore runs
# at a fixed, smaller slot count for *every* block size, so only one variable moves inside it.
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ABLATION_SEQUENCES = 64
# ShareGPT 1.2, not 0.9. Block size only affects how efficiently the KV cache is read, and at rate
# 0.9 the batch was 4.5 (Alpaca) / 10.1 (ShareGPT) - far from saturated, where each decode step is
# dominated by streaming all 2.30 GiB of weights and the KV path is a rounding error. 1.2 sits just
# below paged's capacity point: batch 26 on ShareGPT, so there is a lot of cache to read, while the
# queue stays near empty (0.13) so latency still reflects compute rather than waiting. A saturated
# rate would be just as useless in the other direction - queueing time would swamp the effect.
#
# Alpaca is listed twice on purpose. 1.2 is the original point and is kept, but at that rate
# Alpaca batches only 8.1 sequences for ~2 000 tokens of a 31 358-token pool - 7% - and with the
# pool that empty, the internal fragmentation of a large page never turns into latency. That is
# why the ablation cannot see the paper's "larger block sizes significantly degrade the
# performance" on Alpaca. The second point is a rate with real memory pressure; the two curves
# side by side are the direct evidence that the effect needs a full pool to appear.
# 2.5, chosen from the extended e1 grid: paged batches 20.1 sequences there with waiting_mean
# 0.30, which is the most load that still leaves latency reflecting compute rather than queueing
# (3.0 reaches 23.5 but waiting climbs to 0.38, past runqual's saturation line).
#
# It is worth being blunt about what this does and does not buy. Pool usage goes from 6.6% at
# 1.2 req/s to 16.4% at 2.5 - 2.5x more pressure, and enough to say something. But it is still
# nowhere near a full pool, and no arrival rate can fix that, because the ceiling is not the
# rate:
#
#     64 slots x ~256 tokens of Alpaca context = 16 384 tokens, of a 31 358-token pool = 52%,
#     and that is with every slot occupied. Filling the pool would need ~122 concurrent
#     sequences; ABLATION_SEQUENCES is 64.
#
# So the paper's "larger block sizes significantly degrade the performance" on Alpaca is out of
# reach of this ablation *by construction*, not by an unlucky choice of rate. Running 2.5
# anyway is still worth it: a second curve at 2.5x the pressure turns "the pool was too empty"
# from a hand-wave into a measurement.
ALPACA_PRESSURE_RATE = 2.5
ABLATION_RATES = {
    "sharegpt": [1.2],
    "alpaca": [1.2] + ([ALPACA_PRESSURE_RATE] if ALPACA_PRESSURE_RATE else []),
}
# 200, not 60. A first pass at 60 requests was measurably underpowered: at rate 0.9 that is only
# ~67 s of arrivals, barely two request lifetimes, so the run never leaves its warm-up transient.
# Every one of those runs drifted 12-18% between its halves while the block-size effect being
# measured was 2-3% - the noise was larger than the signal. 200 matches E1 and gives ~6 lifetimes.
ABLATION_REQUESTS = 200

# ---- E5: recovery from running out of pages (§7.3, Fig 19) ----------------------------------
# The paged build already recovers by *recomputing* a preempted sequence, and has done so in
# every sweep so far, but the cost of doing that has never been measured. Fig 19 compares it
# against swapping pages out to host RAM, across block sizes, and finds recomputation flat
# (it never touches a page) while swapping degrades at small block sizes (many tiny PCIe
# transfers).
#
# Two things gate this stage, and neither may be guessed:
#
#   RECOVERY_RATE   must be a load where preemption actually happens often enough to measure.
#                   At 2.0 req/s on ShareGPT the whole run preempts 7 times out of 200 requests
#                   - too thin to average. The gate is >= 30 preemptions in a run; pick the rate
#                   from the extended e1 grid, raising it beyond 2.0 if nothing there qualifies.
#   RECOVERY_POLICIES
#                   "swap" only exists once plan 5 part B lands. Until then this stage measures
#                   the recompute branch alone, which is still the half of Fig 19 we can get.
# Both are now set from measurement, and the measurement moved the choice twice.
#
# ShareGPT cannot carry this stage. Probed at 3.0 req/s (far past its 0.8 capacity point), 100
# requests: peak pool use 73%, running_max 90, and **zero** preemptions. Its sequences are too
# short to fill the pool at a concurrency this GPU can reach.
#
# sharegpt_chat can: ~1086 tokens of context per sequence against a 27 390-token pool, so the
# pool pegs. Probed at 1.2 req/s, 100 requests: peak 99.2% and 6 preemptions.
#
# 6 per 100 is low, and it is low for a reason worth stating rather than engineering around.
# §4.5 says "once it preempts a sequence and evicts its blocks, vLLM stops accepting new requests
# until all preempted sequences are completed", and this engine implements that faithfully - so
# every preemption releases the pressure that caused it. A build that preempted 30+ times per 200
# requests would be one that *ignored* §4.5. The original gate of ">= 30 per run" was written for
# a run-average measurement; Fig 19 is a per-event cost, and the sweep has 9 block sizes x 2
# policies, so ~12 events per run is ~100 events per policy for the fit. The gate is therefore
# per-run >= 10 with the total reported, not per-run >= 30.
RECOVERY_RATE = 1.2
RECOVERY_WORKLOAD = "sharegpt_chat"
RECOVERY_POLICIES = ["recompute", "swap"]
MIN_PREEMPTIONS_FOR_E5 = 10


# ---- E6: several answers to one prompt (§6.3, Fig 14 and 15) --------------------------------
# The memory saving of sharing a prompt's pages between the n answers to it. Fig 15 is the cheap
# half - one rate, a few values of n - and it comes first on purpose: if sharing does not show up
# there, the curves of Fig 14 would be five hours of GPU spent measuring a bug.
#
# Sampling has to be ON. At temperature 0 the n answers are identical, nothing ever diverges, and
# the sharing measured would be the best case rather than the real one. The paper's §6.3 is
# parallel *sampling*, so TEMPERATURE is set and the answers separate after the prompt - which is
# exactly when copy-on-write starts paying for itself.
#
# MAX_SEQUENCES goes to 768 for this stage and for *both* builds (plan 6, part F). At n = 1 the
# paged build already reached 91 concurrent sequences on ShareGPT; at n = 6 the ceiling of 384
# would be hit by the side that is winning, and the run would measure the slot limit instead of
# the mechanism. The cost is ~146 MiB of KV pool, symmetric, so it compresses the dynamic range
# without favouring either side - and it means an e6 run is NOT comparable point-for-point with
# an e1 run. They are different experiments.
PARALLEL_N = [2, 4, 6]
PARALLEL_RATE = 0.6              # inside ShareGPT's capacity point at n = 1, so n is the variable
PARALLEL_WORKLOAD = "sharegpt"
PARALLEL_SEQUENCES = 768
PARALLEL_TEMPERATURE = 0.8
PARALLEL_REQUESTS = 120          # 120 requests x 6 answers is 720 sequences of work
# Fig 14: the serving curve at one n, against all four variants.
PARALLEL_CURVE_N = 6
PARALLEL_CURVE_RATES = [0.15, 0.3, 0.45, 0.6]

# ---- E7: prompts that share a prefix (§6.4, Fig 16) -----------------------------------------
# "the LLM service provider can store the KV cache of the prefix in advance to reduce the
# redundant computation spent on the prefix. In vLLM, this can be conveniently achieved by
# reserving a set of physical blocks for a set of predefined shared prefixes".
#
# The prefix length is not declared here - it is read from the workload file, where
# prepare_workload measured it as the longest common token prefix over the whole set (82 tokens
# 1-shot, 303 tokens 5-shot, against the paper's 80 and 341). Declaring it would make the
# experiment depend on a number nobody checked.
PREFIX_WORKLOADS = ["translate_1shot", "translate_5shot"]
# Measured, not guessed. Probed with the paged build at a deliberately over-capacity 6.0 req/s,
# 150 requests, which makes achieved throughput the ceiling: 4.13 req/s on 1-shot and 1.87 req/s
# on 5-shot. The 5-shot prompt is 3x longer and prefill is GEMM-bound, which is the whole of the
# difference. Each grid therefore has an anchor well under its own ceiling and two points past it.
#
# The probe also confirmed the mechanism works: 149 of 150 requests mapped the registry instead
# of paying for the prefix (the first one establishes it), with 80 pinned pages on 1-shot
# (82/16 = 5 blocks x 16 layers) and 288 on 5-shot (303/16 = 18 x 16).
PREFIX_RATES = {
    "translate_1shot": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "translate_5shot": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
}
# Outputs here are ~35 tokens, a tenth of ShareGPT's, so 150 requests give the same number of
# measured tokens as 200 do there while cutting the arrival floor at the low end of the grid.
PREFIX_REQUESTS = 150


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
    # Every workload the sweep will ask for, not just one: adding sharegpt_chat to WORKLOADS
    # without this check would leave the stage a no-op and the new runs would die on a missing
    # file, one at a time, hours in.
    missing = [w for w in WORKLOADS
               if not os.path.exists(os.path.join(REPO_ROOT, "workloads", f"{w}.json"))]
    if not missing and not force:
        log("prepare: workloads already built")
        return
    log(f"prepare: building workloads ({', '.join(missing) if missing else 'all, forced'})")
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


def bench(out_path, engine_dir, workload, rate, num_requests, env, label, seed=SEED,
          num_seqs=1, shared_prefix_len=0, policy=None):
    max_wall = max_wall_for(rate, num_requests, workload)
    cmd = [PYTHON, "bench.py", "--engine-dir", engine_dir, "--workload", workload,
           "--rate", str(rate), "--num-requests", str(num_requests), "--seed", str(seed),
           "--out", out_path, "--label", label, "--max-wall", str(max_wall),
           "--num-seqs", str(num_seqs), "--shared-prefix-len", str(shared_prefix_len)]
    if policy:
        cmd += ["--preemption-policy", policy]
    rc = sh(cmd, env=env, timeout=max_wall + 900).returncode
    return rc == 0


def e1_grid():
    return [(w, r, v) for w in WORKLOADS for r in RATES[w] for v in VARIANTS]


def stage_e1(force):
    """Interleaved by rate, then by variant, so thermal drift is spread across variants
    instead of landing on whichever one went last."""
    todo = [(w, r, v) for (w, r, v) in e1_grid()
            if force or not done("e1", workload=w, rate=r, variant=v[0])]
    log(f"e1: {len(todo)} run(s) to go of {len(e1_grid())}")
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


def e4_grid():
    # .get, not [], because the ablation deliberately covers only the two workloads the paper's
    # §7.2 uses. A workload with no entry simply does not take part.
    return [(b, w, r) for b in BLOCK_SIZES for w in WORKLOADS
            for r in ABLATION_RATES.get(w, [])]


def stage_e4(force):
    """§7.2: rebuild the paged engine per block size. Each build is immediately benchmarked
    and its result written before moving on, so an interrupted ablation resumes mid-sweep.
    Ordered block-major so one build serves every workload and rate at that block size."""
    todo = [(b, w, r) for (b, w, r) in e4_grid()
            if force or not done("e4", block_size=b, rate=r, workload=w)]
    log(f"e4: {len(todo)} run(s) to go of {len(e4_grid())}")
    current_block = None
    for i, (block, workload, rate) in enumerate(todo, 1):
        if block != current_block:
            log(f"e4: building paged_attention with BLOCK_SIZE={block}, "
                f"MAX_SEQUENCES={ABLATION_SEQUENCES}")
            build_engine("paged_attention",
                         defines=[f"ABLATION_BLOCK_SIZE={block}",
                                  f"ABLATION_MAX_SEQUENCES={ABLATION_SEQUENCES}"],
                         label=f"block={block}")
            current_block = block
        out = result_path("e4", block_size=block, rate=rate, workload=workload)
        log(f"e4 [{i}/{len(todo)}] block={block} {workload} rate={rate}")
        bench(out, "paged_attention", workload, rate, ABLATION_REQUESTS, {},
              f"paged:block{block}")
        time.sleep(COOLDOWN_S)
    if current_block is not None:
        log("e4: restoring the default paged_attention build")
        shutil.rmtree(os.path.join(REPO_ROOT, "paged_attention", "build"), ignore_errors=True)
        build_engine("paged_attention")


def e5_grid():
    if not RECOVERY_RATE:
        return []
    return [(b, p) for b in BLOCK_SIZES for p in RECOVERY_POLICIES]


def stage_e5(force):
    """§7.3: what recovering from an exhausted page pool costs, across block sizes.

    Same rebuild-per-block-size shape as e4, with two differences that are the whole point:
    it runs *past* the capacity point so preemption actually happens, and it runs once per
    recovery policy."""
    if not RECOVERY_RATE:
        log("e5: RECOVERY_RATE is unset - pick it from the e1 data first (see the comment "
            "next to it). Skipping.")
        return
    todo = [(b, p) for (b, p) in e5_grid()
            if force or not done("e5", block_size=b, policy=p, rate=RECOVERY_RATE,
                                 workload=RECOVERY_WORKLOAD)]
    log(f"e5: {len(todo)} run(s) to go of {len(e5_grid())}")
    current_block = None
    for i, (block, policy) in enumerate(todo, 1):
        if block != current_block:
            log(f"e5: building paged_attention with BLOCK_SIZE={block}, "
                f"MAX_SEQUENCES={ABLATION_SEQUENCES}")
            build_engine("paged_attention",
                         defines=[f"ABLATION_BLOCK_SIZE={block}",
                                  f"ABLATION_MAX_SEQUENCES={ABLATION_SEQUENCES}"],
                         label=f"block={block}")
            current_block = block
        out = result_path("e5", block_size=block, policy=policy, rate=RECOVERY_RATE,
                          workload=RECOVERY_WORKLOAD)
        log(f"e5 [{i}/{len(todo)}] block={block} policy={policy}")
        bench(out, "paged_attention", RECOVERY_WORKLOAD, RECOVERY_RATE, ABLATION_REQUESTS,
              {}, f"paged:block{block}:{policy}", policy=policy)
        time.sleep(COOLDOWN_S)
    if current_block is not None:
        log("e5: restoring the default paged_attention build")
        shutil.rmtree(os.path.join(REPO_ROOT, "paged_attention", "build"), ignore_errors=True)
        build_engine("paged_attention")
    # The measurement is only worth anything if preemption happened often enough to average.
    thin = []
    for block, policy in e5_grid():
        p = result_path("e5", block_size=block, policy=policy, rate=RECOVERY_RATE,
                        workload=RECOVERY_WORKLOAD)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                n = json.load(f).get("preemptions", 0)
        except (json.JSONDecodeError, OSError):
            continue
        if n < MIN_PREEMPTIONS_FOR_E5:
            thin.append((block, policy, n))
    if thin:
        log(f"e5: WARNING - {len(thin)} run(s) preempted fewer than {MIN_PREEMPTIONS_FOR_E5} "
            f"times, so their recovery cost is an average over too few events. Raise "
            f"RECOVERY_RATE and re-run. Worst: {min(thin, key=lambda x: x[2])}")


def e6_grid():
    """(kind, n, rate, variant). "mem" is Fig 15 (paged vs Orca (Max), one rate, several n);
    "curve" is Fig 14 (one n, all four variants, a rate grid)."""
    mem_variants = [v for v in VARIANTS if v[0] in ("paged", "base:max")]
    grid = [("mem", n, PARALLEL_RATE, v) for n in PARALLEL_N for v in mem_variants]
    grid += [("curve", PARALLEL_CURVE_N, r, v) for r in PARALLEL_CURVE_RATES for v in VARIANTS]
    return grid


def stage_e6(force):
    """§6.3. Both builds are rebuilt with MAX_SEQUENCES=PARALLEL_SEQUENCES first and the default
    build is restored at the end, the same shape as e4/e5 - with one addition: the slot ceiling
    is checked afterwards, because a run that touched it is not usable data."""
    todo = [g for g in e6_grid()
            if force or not done("e6", kind=g[0], n=g[1], rate=g[2], variant=g[3][0])]
    log(f"e6: {len(todo)} run(s) to go of {len(e6_grid())}")
    if not todo:
        return
    for directory in ("base", "paged_attention"):
        log(f"e6: building {directory} with MAX_SEQUENCES={PARALLEL_SEQUENCES}")
        shutil.rmtree(os.path.join(REPO_ROOT, directory, "build"), ignore_errors=True)
        build_engine(directory, defines=[f"ABLATION_MAX_SEQUENCES={PARALLEL_SEQUENCES}"],
                     label=f"seqs={PARALLEL_SEQUENCES}")
    for i, (kind, n, rate, (name, directory, env)) in enumerate(todo, 1):
        out = result_path("e6", kind=kind, n=n, rate=rate, variant=name)
        log(f"e6 [{i}/{len(todo)}] {kind} n={n} rate={rate} {name}")
        run_env = dict(env)
        run_env["TEMPERATURE"] = str(PARALLEL_TEMPERATURE)
        bench(out, directory, PARALLEL_WORKLOAD, rate, PARALLEL_REQUESTS, run_env,
              f"{name}:n{n}", num_seqs=n)
        time.sleep(COOLDOWN_S)
    log("e6: restoring the default builds")
    for directory in ("base", "paged_attention"):
        shutil.rmtree(os.path.join(REPO_ROOT, directory, "build"), ignore_errors=True)
        build_engine(directory)
    # Plan 6 part F: a run that reached the slot ceiling measured the ceiling, not the mechanism.
    capped = []
    for kind, n, rate, (name, _d, _e) in e6_grid():
        p = result_path("e6", kind=kind, n=n, rate=rate, variant=name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("running_max", 0) >= PARALLEL_SEQUENCES:
            capped.append((kind, n, rate, name, d["running_max"]))
    if capped:
        log(f"e6: WARNING - {len(capped)} run(s) hit MAX_SEQUENCES={PARALLEL_SEQUENCES}. Those "
            f"numbers are the slot limit, not the mechanism; raise it (hard ceiling 1024, see "
            f"markSampledTokensKernel) and re-run. First: {capped[0]}")


def e7_grid():
    return [(w, r, v) for w in PREFIX_WORKLOADS for r in PREFIX_RATES.get(w, [])
            for v in VARIANTS]


def stage_e7(force):
    """§6.4. The prefix length comes from the workload file (-1 tells bench.py to read
    meta.shared_prefix_tokens), so this stage never hardcodes it."""
    if not PREFIX_RATES:
        log("e7: PREFIX_RATES is unset - probe each translate workload's capacity first and "
            "write the grid next to it, the way RATES was set. Skipping.")
        return
    todo = [(w, r, v) for (w, r, v) in e7_grid()
            if force or not done("e7", workload=w, rate=r, variant=v[0])]
    log(f"e7: {len(todo)} run(s) to go of {len(e7_grid())}")
    for i, (workload, rate, (name, directory, env)) in enumerate(todo, 1):
        out = result_path("e7", workload=workload, rate=rate, variant=name)
        log(f"e7 [{i}/{len(todo)}] {workload} rate={rate} {name}")
        bench(out, directory, workload, rate, PREFIX_REQUESTS, env, name,
              shared_prefix_len=-1)
        if i < len(todo):
            time.sleep(COOLDOWN_S)


def stage_report(force):
    log("report: generating figures and HTML")
    if sh([PYTHON, "report.py"]).returncode != 0:
        sys.exit("report.py failed")


# ============================ status ============================

def stage_status(_force):
    total = pending = 0
    print(f"{'stage':8s} {'done':>6s} {'total':>6s}  detail")
    # Seconds a pending run will take. Arrivals alone set the floor (200 requests at 0.15 req/s
    # is 22 minutes before anything else happens), so a flat per-run estimate is wrong by 3x at
    # the ends of the grid. 1.4 is the measured ratio of wall clock to arrival floor across the
    # 67 runs already on disk; the 6-minute floor covers the saturated end where drain dominates.
    def eta_s(rate, num_requests=NUM_REQUESTS, workload=None):
        floor = max(360.0, (num_requests / rate) * 1.4 if rate > 0 else 360.0)
        slowest = SLOWEST_SERVICE_RATE.get(workload or "", 0.0)
        if slowest > 0:
            floor = max(floor, num_requests / slowest * 0.7)
        return floor + COOLDOWN_S

    remaining = []
    e1_all = [(w, r, v[0]) for (w, r, v) in e1_grid()]
    e1_done = [x for x in e1_all if done("e1", workload=x[0], rate=x[1], variant=x[2])]
    remaining += [eta_s(r, NUM_REQUESTS, w) for (w, r, v) in e1_all
                  if (w, r, v) not in e1_done]
    print(f"{'e1':8s} {len(e1_done):6d} {len(e1_all):6d}  serving curves")
    total += len(e1_all); pending += len(e1_all) - len(e1_done)

    e3_done = 1 if done("e3", kind="kernel") else 0
    print(f"{'e3':8s} {e3_done:6d} {1:6d}  attention kernel microbenchmark")
    total += 1; pending += 1 - e3_done
    remaining += [] if e3_done else [120.0]

    e4_all = e4_grid()
    e4_done = [x for x in e4_all
               if done("e4", block_size=x[0], rate=x[2], workload=x[1])]
    # a block size that is not done yet also costs a rebuild
    rebuilds = len({b for (b, w, r) in e4_all if (b, w, r) not in e4_done})
    remaining += [eta_s(r, ABLATION_REQUESTS) for (b, w, r) in e4_all if (b, w, r) not in e4_done]
    remaining += [120.0] * rebuilds
    print(f"{'e4':8s} {len(e4_done):6d} {len(e4_all):6d}  block size ablation")
    total += len(e4_all); pending += len(e4_all) - len(e4_done)

    e6_all = e6_grid()
    e6_done = [x for x in e6_all
               if done("e6", kind=x[0], n=x[1], rate=x[2], variant=x[3][0])]
    remaining += [eta_s(r, PARALLEL_REQUESTS) for (k, n, r, v) in e6_all
                  if (k, n, r, v) not in e6_done]
    remaining += [120.0] * (4 if len(e6_done) < len(e6_all) else 0)   # two rebuilds each way
    print(f"{'e6':8s} {len(e6_done):6d} {len(e6_all):6d}  parallel sampling (n answers per prompt)")
    total += len(e6_all); pending += len(e6_all) - len(e6_done)

    e7_all = e7_grid()
    if e7_all:
        e7_done = [x for x in e7_all if done("e7", workload=x[0], rate=x[1], variant=x[2][0])]
        remaining += [eta_s(r, PREFIX_REQUESTS) for (w, r, v) in e7_all
                      if (w, r, v) not in e7_done]
        print(f"{'e7':8s} {len(e7_done):6d} {len(e7_all):6d}  shared prefix")
        total += len(e7_all); pending += len(e7_all) - len(e7_done)
    else:
        print(f"{'e7':8s} {'-':>6s} {'-':>6s}  shared prefix (PREFIX_RATES unset, see sweep.py)")

    e5_all = e5_grid()
    if e5_all:
        e5_done = [x for x in e5_all
                   if done("e5", block_size=x[0], policy=x[1], rate=RECOVERY_RATE,
                           workload=RECOVERY_WORKLOAD)]
        rebuilds = len({b for (b, p) in e5_all if (b, p) not in e5_done})
        remaining += [eta_s(RECOVERY_RATE, ABLATION_REQUESTS)
                      for x in e5_all if x not in e5_done]
        remaining += [120.0] * rebuilds
        print(f"{'e5':8s} {len(e5_done):6d} {len(e5_all):6d}  recovery cost vs block size")
        total += len(e5_all); pending += len(e5_all) - len(e5_done)
    else:
        print(f"{'e5':8s} {'-':>6s} {'-':>6s}  recovery cost (RECOVERY_RATE unset, see sweep.py)")

    print(f"\n{total - pending}/{total} runs complete.")
    if pending:
        print(f"Roughly {sum(remaining) / 3600:.1f} h of wall clock left. Re-run "
              f"`python sweep.py --all` to continue; finished runs are never redone.")
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
          "e6": stage_e6, "e7": stage_e7,
          "e3": stage_e3, "e4": stage_e4, "e5": stage_e5, "report": stage_report,
          "status": stage_status}
ORDER = ["prepare", "build", "e1", "e3", "e4", "e5", "e6", "e7", "report"]


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
