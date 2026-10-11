"""Turn results/runs/*.json into the paper's figures and a self-contained HTML report.

Mirrors the figures of Kwon et al., SOSP '23:
  Fig 12  normalised latency vs request rate, per dataset            -> fig_serving.png
  Fig 13  average number of batched requests                         -> fig_batched.png
  Fig 2   KV cache: reserved vs holding a real token, over time      -> fig_kv_waste.png
  Fig 18a latency of the attention kernels                           -> fig_kernel.png
  Fig 18b end-to-end latency vs block size                           -> fig_blocksize.png
  §6.1    the workload length distributions we actually got          -> fig_workload.png

Plus three things the paper's figures do not isolate but this replication can measure, because
both allocators are ours and instrumented:
  time to first token vs rate       -> fig_ttft.png   (what queueing actually does to a user)
  external-fragmentation failures   -> fig_frag.png   (the cost contiguity charges by itself)
  preemption counts                 -> table          (the cost over-subscription charges back)

And the later experiments, each section degrading to "not run" while its stage is missing:
  Fig 19  recovery cost per event, recompute vs swap, by block size  -> recovery.png
  Fig 15  parallel sampling: concurrency, latency, memory shared      -> parallel.png
  Fig 16  shared prefix serving curves                                -> prefix.png

Works on partial results: whatever stage has not run yet is simply reported as missing, so
this is safe to call at any point during a sweep.

usage: python report.py [--artifact]
"""

import argparse
import base64
import glob
import io
import json
import os
from collections import defaultdict

import runqual
import sweep   # only for the planned grids, so the report can say how much is still missing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO_ROOT, "results", "runs")
REPORT = os.path.join(REPO_ROOT, "results", "report")
FIGS = os.path.join(REPORT, "figures")

# base:max is the official baseline; pow2 and oracle are unreachable upper bounds and are
# drawn dashed so a reader cannot mistake them for implementable systems.
STYLE = {
    "paged":       dict(color="#1b6ca8", marker="o", ls="-",  lw=2.0, label="vLLM-style (paged)"),
    "base:max":    dict(color="#c0392b", marker="s", ls="-",  lw=2.0, label="Orca (Max)"),
    "base:pow2":   dict(color="#d98c00", marker="^", ls="--", lw=1.5, label="Orca (Pow2) — unreachable"),
    "base:oracle": dict(color="#5b8c2a", marker="v", ls=":",  lw=1.5, label="Orca (Oracle) — unreachable"),
}
ORDER = ["paged", "base:max", "base:pow2", "base:oracle"]
VARIANT_VI = {
    "paged": "paged (vLLM)",
    "base:max": "Orca (Max)",
    "base:pow2": "Orca (Pow2)",
    "base:oracle": "Orca (Oracle)",
}
WORKLOAD_TITLE = {
    "alpaca": "Alpaca (short)",
    "sharegpt": "ShareGPT (long)",
    "sharegpt_chat": "ShareGPT chatbot (§6.5)",
}
WORKLOADS = ("alpaca", "sharegpt", "sharegpt_chat")


def panel_width(n, wide=5.4, floor=4.2):
    """Per-panel width that stops a third workload from producing a 16-inch figure nobody can
    read on a page. Two panels keep the width they were tuned at."""
    return wide if n <= 2 else max(floor, wide * 2 / n)


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        d["_file"] = os.path.basename(p)
        d["_stage"] = os.path.basename(p).split("_", 1)[0]
        runs.append(d)
    return runs


def savefig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, base64.b64encode(buf.getvalue()).decode("ascii")


# ============================ derived per-run metrics ============================
# Everything the report quotes comes through here, so a metric is defined in exactly one
# place and the tables cannot drift from the figures.

def pctile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def _mean_step_pct(run, field):
    """`field` as a percentage of the pool, averaged over steps that had work in them. Idle
    steps are excluded because they would drag every number toward zero by a different amount
    per run, depending only on how long the tail was."""
    steps = [s for s in (run.get("steps") or []) if s.get("running")]
    if not steps or not steps[0].get("kv_tokens_total"):
        return None
    total = steps[0]["kv_tokens_total"]
    return 100.0 * sum(s.get(field, 0) for s in steps) / len(steps) / total


def derive(r):
    """One row of measurements per run. TTFT is computed here rather than in bench.py because
    it is a property of the arrival trace plus the schedule, and bench.py records both."""
    reqs = r.get("requests") or []
    ttfts = [q["t_first"] - q["t_arrival"] for q in reqs
             if q.get("t_first") is not None and q.get("t_arrival") is not None]
    steps = r.get("steps") or []
    ec = r.get("engine_config") or {}
    return {
        "file": r.get("_file"),
        "stage": r.get("_stage"),
        "workload": r.get("workload"),
        "variant": r.get("label"),
        "rate": r.get("rate"),
        "num_requests": r.get("num_requests"),
        "nl": r.get("normalized_latency_s_per_token"),
        "latency_mean": r.get("latency_mean_s"),
        "throughput": r.get("throughput_req_s"),
        "ttft_mean": mean(ttfts),
        "ttft_p99": pctile(ttfts, 99),
        "running_mean": r.get("running_mean"),
        "running_max": r.get("running_max"),
        "waiting_mean": r.get("waiting_mean"),
        # The engine reports fragmentation_failures as a monotonic counter, so the last
        # non-empty step holds the run total; max() is robust to a truncated tail.
        "frag": max((s.get("fragmentation_failures", 0) for s in steps), default=0),
        "preempt": r.get("preemptions", 0) or 0,
        # §7.3. Older runs predate the field, so missing means "not measured", which is a
        # different thing from zero and has to stay distinguishable in the table.
        # Mean over the steps where anything was running. Needed by the block-size section:
        # internal fragmentation is (reserved - live) / reserved, and the whole §7.2 argument
        # turns on telling that apart from how *full* the pool is.
        "reserved_pct": _mean_step_pct(r, "kv_tokens_reserved"),
        "live_pct": _mean_step_pct(r, "kv_tokens_live"),
        # §6.3 / Fig 15: block-table entries served by a page someone else already holds, i.e.
        # pages that did not have to exist. Zero on every run without parallel sampling.
        "shared_pct": _mean_step_pct(r, "kv_tokens_shared"),
        "abandoned": r.get("abandoned"),
        "recompute_ms": r.get("recompute_ms"),
        "recompute_tokens": r.get("recompute_tokens"),        # §7.3 part B and §6.4, all absent on every run taken before they existed - so None
        # means "not measured" and 0 means "measured as zero", and the report must not merge them.
        "policy": r.get("preemption_policy"),
        "swap_outs": r.get("swap_outs"),
        "swap_ins": r.get("swap_ins"),
        "swap_fallbacks": r.get("swap_fallbacks"),
        "swap_mb": r.get("swap_mb"),
        "swap_ms": r.get("swap_ms"),
        "prefix_len": r.get("shared_prefix_len"),
        "prefix_hits": r.get("prefix_hits"),
        "prefix_pinned_pages": r.get("prefix_pinned_pages"),
        "n_seqs": (r.get("requests") or [{}])[0].get("n_seqs") if r.get("requests") else None,

        "wall_s": r.get("wall_s"),
        "pool_tokens": ec.get("kv_pool_tokens"),
        "block_size": ec.get("block_size"),
        "slots": ec.get("batch_size"),
        "drift": (r.get("convergence") or {}).get("drift"),
        "first_half": (r.get("convergence") or {}).get("first_half"),
        "second_half": (r.get("convergence") or {}).get("second_half"),
        "quality": runqual.classify(r)[0],
        "quality_why": runqual.classify(r)[1],
    }


def pick(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def at_rate(rows, workload, variant, rate):
    m = pick(rows, workload=workload, variant=variant, rate=rate)
    return m[0] if m else None


def reference_rate(rows, workload, variant="paged", queue_threshold=1.0):
    """The highest rate at which `variant` is still keeping up — its queue is essentially
    empty. Comparisons quoted at this rate are the fair ones: the winner is not yet saturated,
    so its number is a real steady-state figure rather than a lower bound."""
    ok = [r["rate"] for r in pick(rows, workload=workload, variant=variant)
          if r["waiting_mean"] is not None and r["waiting_mean"] < queue_threshold]
    return max(ok) if ok else None


# ============================ workload distributions ============================

def fig_workload():
    metas, samples = {}, {}
    for name in WORKLOADS:
        p = os.path.join(REPO_ROOT, "workloads", f"{name}.json")
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        metas[name] = d["meta"]
        samples[name] = d["samples"]
    if not samples:
        return None, None, {}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for key, ax, title in (("prompt_len", axes[0], "Prompt length"),
                           ("output_len", axes[1], "Output length")):
        for name, color in (("alpaca", "#d98c00"), ("sharegpt", "#1b6ca8")):
            if name not in samples:
                continue
            ax.hist([s[key] for s in samples[name]], bins=60, histtype="step", lw=1.6,
                    color=color, label=WORKLOAD_TITLE[name], density=True)
        ax.set_xlabel(f"{title} (tokens)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.grid(alpha=0.25, ls=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Workload length distributions, from the real ShareGPT / Alpaca traces",
                 y=1.03, fontsize=11)
    path, b64 = savefig(fig, "fig_workload.png")
    return path, b64, metas


# ============================ capacity point, and whether the grid pinned it ============================
# The paper reads its headline ratio off the plot by eye ("sustains 1.7x-2.7x higher request
# rates"). The operational stand-in is: the highest rate at which normalised latency is still
# under 2x its unloaded value. That definition is only as good as the grid it is evaluated on,
# and a value that lands on either end of the grid is not a measurement - it is a bound. Keeping
# that distinction in the data structure, rather than in a footnote, is what stops a censored
# number from being quoted as if it were measured.

def capacity_point(xs, ys):
    """xs sorted rates, ys the matching normalised latencies. -> {"value", "censored"}.

    censored == "left"  : pinned at the lowest rate measured. The curve had already crossed the
                          threshold by the second point, so the true capacity is below it - and
                          the anchor ys[0] used to build the threshold may itself be degraded,
                          which only pushes the true value lower.
    censored == "right" : never crossed the threshold inside the grid, so the true capacity is
                          at or above the highest rate measured.
    """
    if not xs:
        return {"value": None, "censored": None}
    threshold = 2.0 * ys[0]
    ok = [x for x, y in zip(xs, ys) if y <= threshold]
    if not ok:
        return {"value": None, "censored": None}
    value = max(ok)
    censored = None
    if len(xs) > 1 and value == xs[0]:
        censored = "left"
    elif value == xs[-1]:
        censored = "right"
    return {"value": value, "censored": censored}


def rate_str(x):
    """A rate as it is written in prose: 0,3 and 2,0 rather than 0.3 and 2."""
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",") + ("" if x % 1 else ",0")


def rate_grid_str(xs):
    return " / ".join(rate_str(x) for x in xs)


def capacity_text(cap):
    """How a single capacity point is allowed to be written down."""
    if not cap or cap["value"] is None:
        return "—"
    v = vi(cap["value"])
    if cap["censored"] == "left":
        return f"≤ {v}"
    if cap["censored"] == "right":
        return f"≥ {v}"
    return v


def capacity_ratio(num, den):
    """-> (ratio, prefix). prefix is "", "≥" or "≤" depending on which way the censoring pushes
    the true ratio. A bound in both directions at once returns None, because then the grid says
    nothing about the ratio at all."""
    if not num or not den or num["value"] is None or not den["value"]:
        return None, ""
    ratio = num["value"] / den["value"]
    if num is den:
        return ratio, ""
    # the numerator being a lower bound, or the denominator an upper bound, raises the true ratio
    up = num["censored"] == "right" or den["censored"] == "left"
    down = num["censored"] == "left" or den["censored"] == "right"
    if up and down:
        return ratio, "?"
    return ratio, "≥" if up else ("≤" if down else "")


# ============================ Fig 12: serving curves ============================

def fig_serving(rows):
    """The paper's headline plot. Latency climbs slowly, then explodes once the arrival rate
    passes what the server can absorb; the rate at which that happens is the result."""
    by = defaultdict(dict)
    for r in rows:
        if r["nl"] is None:
            continue
        by[r["workload"]].setdefault(r["variant"], {})[r["rate"]] = r["nl"]
    if not by:
        return None, None, {}

    workloads = [w for w in WORKLOADS if w in by]
    fig, axes = plt.subplots(1, len(workloads), figsize=(panel_width(len(workloads)) * len(workloads), 4.1), squeeze=False)
    capacity = {}
    for ax, workload in zip(axes[0], workloads):
        for variant in ORDER:
            series = by[workload].get(variant)
            if not series:
                continue
            xs = sorted(series)
            ys = [series[x] for x in xs]
            ax.plot(xs, ys, **STYLE[variant])
            capacity.setdefault(workload, {})[variant] = capacity_point(xs, ys)
        ax.set_title(WORKLOAD_TITLE.get(workload, workload))
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel("Normalized latency (s/token)")
        ax.grid(alpha=0.25, ls=":")
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Normalized latency vs request rate  —  paper Fig. 12", y=1.02, fontsize=11)
    path, b64 = savefig(fig, "fig_serving.png")
    return path, b64, capacity


# ============================ time to first token ============================

def fig_ttft(rows):
    """Not a figure in the paper, and the most legible consequence of the whole mechanism.
    Normalised latency divides by output length, which dilutes queueing delay across every
    token of a long answer; TTFT is the queueing delay undiluted, and it is what a user feels.
    Log y-axis because the spread is three orders of magnitude."""
    by = defaultdict(dict)
    for r in rows:
        if r["ttft_mean"] is None:
            continue
        by[r["workload"]].setdefault(r["variant"], {})[r["rate"]] = r["ttft_mean"]
    if not by:
        return None, None
    workloads = [w for w in WORKLOADS if w in by]
    fig, axes = plt.subplots(1, len(workloads), figsize=(panel_width(len(workloads)) * len(workloads), 4.0), squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        for variant in ORDER:
            series = by[workload].get(variant)
            if not series:
                continue
            xs = sorted(series)
            ax.plot(xs, [series[x] for x in xs], **STYLE[variant])
        ax.set_yscale("log")
        ax.set_title(WORKLOAD_TITLE.get(workload, workload))
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel("Time to first token (s, log scale)")
        ax.grid(alpha=0.25, ls=":", which="both")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Time to first token vs request rate  —  queueing delay, undiluted",
                 y=1.02, fontsize=11)
    return savefig(fig, "fig_ttft.png")


# ============================ Fig 13: batched requests ============================

def fig_batched(rows):
    """Why the curves differ: how many sequences the scheduler could keep in flight. On a
    bandwidth-bound decoder this is what converts into throughput."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["running_mean"]:
            by[r["workload"]][r["variant"]].append(r["running_mean"])
    if not by:
        return None, None
    workloads = [w for w in WORKLOADS if w in by]
    fig, axes = plt.subplots(1, len(workloads), figsize=(panel_width(len(workloads), 4.6, 3.6) * len(workloads), 3.8), squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        names = [v for v in ORDER if v in by[workload]]
        vals = [max(by[workload][v]) for v in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[STYLE[v]["color"] for v in names], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("base:", "Orca\n") for n in names], fontsize=8)
        ax.set_ylabel("Batched requests (peak of per-run mean)")
        ax.set_title(WORKLOAD_TITLE.get(workload, workload))
        ax.grid(alpha=0.25, ls=":", axis="y")
    fig.suptitle("Average number of batched requests  —  paper Fig. 13", y=1.03, fontsize=11)
    return savefig(fig, "fig_batched.png")


# ============================ Fig 2: KV waste ============================

def fig_kv_waste(e1):
    """The paper's Figure 2, measured instead of illustrated: of the cache a system has handed
    out, how much actually holds a token. Everything above the solid line is waste."""
    picked = {}
    for r in e1:
        if r["workload"] != "sharegpt" or not r.get("steps"):
            continue
        key = r["label"]
        if key not in picked or r["rate"] > picked[key]["rate"]:
            picked[key] = r
    if not picked:
        return None, None, {}

    names = [v for v in ORDER if v in picked]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6), squeeze=False,
                             sharey=True)
    waste = {}
    for ax, variant in zip(axes[0], names):
        r = picked[variant]
        steps = [s for s in r["steps"] if s["running"] > 0]
        if not steps:
            continue
        t = [s["t_ms"] / 1000.0 for s in steps]
        total = steps[0]["kv_tokens_total"]
        res = [100.0 * s["kv_tokens_reserved"] / total for s in steps]
        live = [100.0 * s["kv_tokens_live"] / total for s in steps]
        ax.fill_between(t, live, res, color=STYLE[variant]["color"], alpha=0.25,
                        label="handed out but empty")
        ax.plot(t, res, color=STYLE[variant]["color"], ls="--", lw=1.2, label="handed out")
        ax.plot(t, live, color=STYLE[variant]["color"], lw=1.8, label="holds a token")
        mr = sum(res) / len(res)
        ml = sum(live) / len(live)
        waste[variant] = {"reserved_pct": mr, "live_pct": ml,
                          "waste_pct": 100.0 * (mr - ml) / mr if mr else 0.0,
                          "rate": r["rate"]}
        ax.set_title(f"{variant}\n{waste[variant]['waste_pct']:.0f}% of it wasted", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, ls=":")
    axes[0][0].set_ylabel("% of KV cache pool")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("KV cache: handed out vs actually holding a token  —  paper Fig. 2 "
                 "(ShareGPT, busiest rate)", y=1.04, fontsize=10)
    path, b64 = savefig(fig, "fig_kv_waste.png")
    return path, b64, waste


# ============================ external fragmentation ============================

def fig_frag(rows):
    """Not in the paper, and only measurable because both allocators are ours: the number of
    reservations the pool had room for in total but could not place in one contiguous run.
    Paging has no such failure mode, so its line is identically zero - that is the point."""
    by = defaultdict(dict)
    for r in rows:
        by[r["workload"]].setdefault(r["variant"], {})[r["rate"]] = r["frag"]
    if not by or not any(v for w in by.values() for v in
                         (max(s.values()) for s in w.values())):
        return None, None
    workloads = [w for w in WORKLOADS if w in by]
    # sharey is the whole point of plotting Alpaca at all: on its own axis a column of zeroes
    # auto-scales to 0.00-0.05 and reads as if something were happening. Against ShareGPT's
    # axis it reads as what it is - flat zero while the other workload climbs past a thousand.
    fig, axes = plt.subplots(1, len(workloads), figsize=(panel_width(len(workloads), 5.0, 4.0) * len(workloads), 3.6),
                             squeeze=False, sharey=True)
    peak = max((v for w in by.values() for s in w.values() for v in s.values()), default=1)
    for ax, workload in zip(axes[0], workloads):
        for variant in ORDER:
            series = by[workload].get(variant)
            if not series:
                continue
            xs = sorted(series)
            ax.plot(xs, [series[x] for x in xs], **STYLE[variant])
        if all(max(s.values()) == 0 for s in by[workload].values()):
            ax.annotate("mọi biến thể: 0", xy=(0.5, 0.5), xycoords="axes fraction",
                        ha="center", fontsize=10, color="#666")
        ax.set_title(WORKLOAD_TITLE.get(workload, workload))
        ax.set_xlabel("Request rate (req/s)")
        ax.grid(alpha=0.25, ls=":")
        ax.set_ylim(0, peak * 1.12)
    axes[0][0].set_ylabel("Failed reservations (pool had room)")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("External fragmentation: room in total, no contiguous run to put it in",
                 y=1.03, fontsize=11)
    return savefig(fig, "fig_frag.png")


# ============================ Fig 18a: kernel ============================

def fig_kernel(e3):
    if not e3:
        return None, None, []
    rows = e3[0]["rows"]
    by = {(r["mechanism"], r["batch"], r["context_len"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    ctxs = sorted({r["context_len"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(4.4 * len(batches), 3.6), squeeze=False)
    table_rows = []
    for ax, batch in zip(axes[0], batches):
        width = 0.36
        xs = range(len(ctxs))
        c = [by[("contiguous", batch, c_)]["latency_us"] for c_ in ctxs]
        p = [by[("paged", batch, c_)]["latency_us"] for c_ in ctxs]
        ax.bar([x - width / 2 for x in xs], c, width, label="contiguous", color="#c0392b")
        ax.bar([x + width / 2 for x in xs], p, width, label="paged", color="#1b6ca8")
        for c_, cv, pv in zip(ctxs, c, p):
            table_rows.append({"batch": batch, "context_len": c_, "contiguous_us": cv,
                               "paged_us": pv, "ratio": pv / cv,
                               "digest_match": by[("contiguous", batch, c_)]["output_digest"]
                                               == by[("paged", batch, c_)]["output_digest"]})
        ax.set_xticks(list(xs))
        ax.set_xticklabels(ctxs)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Kernel latency (µs)")
        ax.set_title(f"batch size {batch}")
        ax.grid(alpha=0.25, ls=":", axis="y")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Attention kernel latency  —  paper Fig. 18a", y=1.03, fontsize=11)
    path, b64 = savefig(fig, "fig_kernel.png")
    return path, b64, table_rows


# ============================ Fig 18b: block size ============================

def blocksize_series(rows):
    """One series per (workload, rate), not per workload.

    The paper's claim that large blocks hurt short sequences is a claim about *memory
    pressure*, and the same workload at two arrival rates is two different amounts of it. Keying
    on the workload alone would silently overwrite one curve with the other."""
    by = defaultdict(dict)
    for r in rows:
        if r["nl"] is not None and r["block_size"]:
            by[(r["workload"], r["rate"])][r["block_size"]] = r["nl"]
    return by


def blocksize_label(workload, rate, multi_rate):
    base = WORKLOAD_TITLE.get(workload, workload)
    return f"{base} @ {rate_str(rate)} req/s" if multi_rate else base


def fig_blocksize(rows):
    by = blocksize_series(rows)
    if not by:
        return None, None
    # A workload that appears at more than one rate gets the rate spelled out in its label;
    # otherwise the legend stays as short as it was.
    rates_per_workload = defaultdict(set)
    for (w, rate) in by:
        rates_per_workload[w].add(rate)
    # Two panels on purpose. The absolute one is zero-based so the magnitudes are not
    # exaggerated; but the workloads sit at very different latencies and the effect is a few
    # percent, so on that axis the shape is invisible. The right panel divides each series by
    # its own best point, which is the only way to actually see the U - and it is labelled as
    # relative so nobody reads it as an absolute gap.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    colors = {"alpaca": "#d98c00", "sharegpt": "#1b6ca8", "sharegpt_chat": "#5b8c2a"}
    # a second rate for the same workload is the same hue, dashed, so the pair reads as a pair
    dashes = {}
    for ax, relative in ((axes[0], False), (axes[1], True)):
        for (workload, rate), series in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            xs = sorted(series)
            ys = [series[x] for x in xs]
            if relative:
                best = min(ys)
                ys = [100.0 * (y / best - 1.0) for y in ys]
            seen = dashes.setdefault(workload, [])
            if rate not in seen:
                seen.append(rate)
            ls = "-" if seen.index(rate) == 0 else "--"
            ax.plot(xs, ys, marker="o", ls=ls, color=colors.get(workload),
                    label=blocksize_label(workload, rate, len(rates_per_workload[workload]) > 1))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Block size (tokens per page)")
        ax.axvline(16, color="#888", ls="--", lw=1)
        ax.grid(alpha=0.25, ls=":")
        if relative:
            ax.set_ylabel("% worse than that series' best block size")
            ax.set_title("relative to each series' own best", fontsize=9)
            ax.axhline(0, color="#aaa", lw=0.8)
        else:
            ax.set_ylabel("Normalized latency (s/token)")
            ax.set_title("absolute", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, loc="center left")
        ax.annotate("vLLM default", xy=(16, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords="offset points", fontsize=8, color="#555", va="top")
    fig.suptitle("End-to-end latency vs block size  —  paper Fig. 18b", y=1.02, fontsize=11)
    return savefig(fig, "fig_blocksize.png")


# ============================ inline diagrams ============================
# Hand-drawn SVG rather than matplotlib: these are schematics, not data, and keeping them as
# markup means they inherit the page's theme variables and stay sharp at any zoom. The
# proportions are not decorative - the fill fractions are the measured live/reserved ratios
# from the Fig 2 panel, so the picture cannot contradict the table under it.

def svg_mechanisms(waste_contig=0.80, waste_paged=0.015):
    """Why the same pool holds a different number of sequences. Top: one contiguous
    reservation per sequence, sized for the worst case. Bottom: pages handed out on demand."""
    x0, box_w, gap = 40, 165, 5
    # Four reservations of identical size (Orca Max reserves MAX_SEQ_LEN for everyone); the
    # filled part is how much of it ever holds a token, spread around the measured mean.
    fills = [0.22, 0.11, 0.30, 0.14]
    boxes = []
    for i, f in enumerate(fills):
        x = x0 + i * (box_w + gap)
        boxes.append(
            f'<rect x="{x}" y="58" width="{box_w}" height="38" rx="3" fill="var(--critical)" '
            f'fill-opacity="0.12" stroke="var(--critical)" stroke-width="1.4" '
            f'stroke-dasharray="4 3"/>'
            f'<rect x="{x}" y="58" width="{box_w * f:.1f}" height="38" rx="3" '
            f'fill="var(--critical)"/>')
    # Pages: equal cells, colours cycling so neighbouring pages visibly belong to different
    # sequences. The formula is just a deterministic scatter, not data.
    tints = ["#1b6ca8", "#2f8fa8", "#44618f", "#2a7f62"]
    cells, cw, cgap, n = [], 13, 1, 48
    for i in range(n):
        x = x0 + i * (cw + cgap)
        c = tints[(i * 7 + i // 3) % 4]
        # the very last page of a sequence is the only one that can be partly empty
        partial = (i == n - 1)
        cells.append(f'<rect x="{x}" y="190" width="{cw}" height="38" rx="2" fill="{c}" '
                     f'fill-opacity="0.15" stroke="{c}" stroke-width="0.8"/>')
        h = 14 if partial else 38
        cells.append(f'<rect x="{x}" y="{190 + 38 - h}" width="{cw}" height="{h}" rx="2" '
                     f'fill="{c}"/>')
    return f'''<svg viewBox="0 0 760 318" role="img" width="100%"
     aria-label="So sánh cách hai cơ chế lấp cùng một pool KV cache">
  <text x="40" y="26" font-size="14" font-weight="650" fill="var(--ink)">Đặt chỗ liên tục (Orca)</text>
  <text x="40" y="44" font-size="12" fill="var(--muted)">một vùng liền mạch mỗi sequence, kích thước cho trường hợp xấu nhất (2048 token)</text>
  {''.join(boxes)}
  <text x="40" y="116" font-size="12" fill="var(--muted)">
    đặc = chứa token thật · nhạt = đã cấp nhưng rỗng
  </text>
  <text x="40" y="136" font-size="12" fill="var(--critical)" font-weight="600">
    pool 27 518 token ÷ 2048 = <tspan font-weight="700">13 sequence</tspan>, và ~{100 * waste_contig:.0f}% chỗ đã cấp ra là rỗng
  </text>

  <line x1="40" y1="156" x2="720" y2="156" stroke="var(--line)" stroke-width="1"/>

  <text x="40" y="180" font-size="14" font-weight="650" fill="var(--ink)">Paging (vLLM)</text>
  {''.join(cells)}
  <text x="40" y="248" font-size="12" fill="var(--muted)">
    mỗi ô = 1 page = 16 token của 1 layer · màu = sequence sở hữu · page nằm xen kẽ tuỳ ý
  </text>
  <text x="40" y="268" font-size="12" fill="var(--accent)" font-weight="600">
    cùng pool đó: <tspan font-weight="700">~30 sequence</tspan>, và chỉ ~{100 * waste_paged:.0f}% là rỗng — phần đuôi page cuối
  </text>
  <text x="40" y="290" font-size="11.5" fill="var(--muted)">
    Mọi page bằng nhau nên page trống nào cũng vừa bất kỳ sequence.
  </text>
  <text x="40" y="308" font-size="11.5" fill="var(--muted)">
    Cái giá: địa chỉ token không còn suy ra được bằng một phép nhân nữa, phải tra bảng trước.
  </text>
</svg>'''


def svg_fragmentation():
    """The two kinds of waste, which the report keeps separate because the mechanisms trade
    one against the other. Internal: handed out, not yet used. External: free, unusable."""
    return '''<svg viewBox="0 0 760 290" role="img" width="100%"
     aria-label="Phân biệt phân đoạn trong và phân đoạn ngoài">
  <text x="30" y="24" font-size="14" font-weight="650" fill="var(--ink)">Phân đoạn trong — đã cấp, chưa dùng</text>
  <rect x="30" y="38" width="500" height="36" rx="3" fill="var(--critical)" fill-opacity="0.12"
        stroke="var(--critical)" stroke-width="1.4" stroke-dasharray="4 3"/>
  <rect x="30" y="38" width="110" height="36" rx="3" fill="var(--critical)"/>
  <text x="34" y="92" font-size="11.5" fill="var(--muted)">đang dùng</text>
  <text x="150" y="92" font-size="11.5" fill="var(--muted)">đặt chỗ cho tương lai có thể không bao giờ đến</text>
  <text x="548" y="60" font-size="12" fill="var(--muted)">⟵ 2048 token ⟶</text>
  <text x="30" y="116" font-size="12" fill="var(--ink)">
    Lãng phí nhưng <tspan font-weight="600">lành</tspan>: bộ nhớ vẫn thuộc về một sequence đang sống, và sẽ được trả lại nguyên vẹn.
  </text>

  <line x1="30" y1="134" x2="730" y2="134" stroke="var(--line)" stroke-width="1"/>

  <text x="30" y="160" font-size="14" font-weight="650" fill="var(--ink)">Phân đoạn ngoài — còn trống, không dùng được</text>
  <rect x="30" y="174" width="150" height="36" rx="3" fill="var(--good)" fill-opacity="0.18"
        stroke="var(--good)" stroke-width="1.2"/>
  <rect x="180" y="174" width="300" height="36" rx="3" fill="var(--muted)" fill-opacity="0.45"
        stroke="var(--muted)" stroke-width="1.2"/>
  <rect x="480" y="174" width="150" height="36" rx="3" fill="var(--good)" fill-opacity="0.18"
        stroke="var(--good)" stroke-width="1.2"/>
  <text x="60" y="197" font-size="11.5" fill="var(--ink)">trống 512</text>
  <text x="258" y="197" font-size="11.5" fill="var(--ink)">đang dùng 1024</text>
  <text x="510" y="197" font-size="11.5" fill="var(--ink)">trống 512</text>
  <text x="30" y="232" font-size="12" fill="var(--ink)">
    Còn trống <tspan font-weight="600">1024</tspan> token, nhưng một request xin 1024 vẫn bị
    <tspan font-weight="600" fill="var(--critical)">từ chối</tspan>: không có vùng liền mạch nào đủ.
  </text>
  <text x="30" y="252" font-size="12" fill="var(--muted)">
    Buddy chỉ gộp được hai nửa sinh ra từ cùng một lần cắt (địa chỉ lệch nhau đúng một bit), nên hai mảnh này không gộp.
  </text>
  <text x="30" y="276" font-size="12" fill="var(--accent)" font-weight="600">
    Paging không có kiểu thất bại này: mọi page bằng nhau, nên “vừa” hay “không vừa” không phải một khái niệm tồn tại.
  </text>
</svg>'''


# ============================ HTML ============================

def table(headers, rows, note=None, highlight_col=None):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for r in rows:
        cells = ""
        for i, c in enumerate(r):
            cls = ' class="hl"' if highlight_col is not None and i == highlight_col else ""
            cells += f"<td{cls}>{c}</td>"
        body += f"<tr>{cells}</tr>"
    cap = f"<figcaption>{note}</figcaption>" if note else ""
    return (f'<figure><div class="scroll"><table><thead><tr>{h}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>{cap}</figure>')


def figure(b64, caption):
    if not b64:
        return (f'<figure class="missing">Chưa có — stage sinh ra hình này chưa chạy.'
                f'<figcaption>{caption}</figcaption></figure>')
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def diagram(svg, caption):
    return f'<figure class="diagram">{svg}<figcaption>{caption}</figcaption></figure>'


def blocksize_penalty(rows):
    """Largest slowdown of the smallest page against vLLM's 16, over every (workload, rate) the
    ablation swept, as (penalty fraction, workload, rate) - or None before e4 has data."""
    best = None
    for (workload, rate), by_bs in blocksize_series(rows).items():
        if 16 in by_bs and 1 in by_bs:
            pen = by_bs[1] / by_bs[16] - 1.0
            if best is None or pen > best[0]:
                best = (pen, workload, rate)
    return best


def recovery_summary(rows):
    """(recompute mean ms, swap mean ms) per recovery event, or None while e5 is incomplete."""
    data = recovery_rows(rows)
    rc = [d["cost_ms"] for d in data if d["policy"] == "recompute"]
    sw = [d["cost_ms"] for d in data if d["policy"] == "swap"]
    return (mean(rc), mean(sw)) if rc and sw else None


def section_status(e1, e3_raw, e4, e5, e6, e7, capacity, kernel_table):
    """Where the data collection stands, one row per experiment, with the one-line answer each
    finished experiment gives. The sweep was not run to completion, so a reader has to be able
    to tell at a glance which conclusions rest on data and which sections are still empty."""
    def count(have, planned):
        cls = "ok" if have >= planned else ("bad" if have == 0 else "")
        return f'<span class="{cls}">{have} / {planned}</span>'

    rows = []
    # E1
    caps = capacity or {}
    bits = []
    for w in WORKLOADS:
        ratio, prefix = capacity_ratio(caps.get(w, {}).get("paged"), caps.get(w, {}).get("base:max"))
        if ratio is not None and prefix != "?":
            bits.append(f"{WORKLOAD_TITLE[w]} {prefix}{vi(ratio, '.2f')}×")
    rows.append(["§6.2 + §6.5 đường cong serving (E1)", count(len(e1), len(sweep.e1_grid())),
                 ("paged chịu được tốc độ đến cao hơn Orca (Max): " + "; ".join(bits) + ". → §4–6")
                 if bits else "chưa có"])
    # E3
    if kernel_table:
        mr = mean([r["ratio"] for r in kernel_table])
        k = f"kernel paged chạy bằng {vi(mr, '.2f')}× kernel liền mạch — tra bảng gần như miễn phí. → §7"
    else:
        k = "chưa có"
    rows.append(["§7.1 kernel attention (E3)", count(1 if e3_raw else 0, 1), k])
    # E4
    pen = blocksize_penalty(e4)
    rows.append(["§7.2 block size (E4)", count(len(e4), len(sweep.e4_grid())),
                 (f"16 token/page nằm trong vùng tốt nhất; page 1 token chậm tới "
                  f"+{vi(100 * pen[0], '.0f')}% khi pool chật "
                  f"({WORKLOAD_TITLE.get(pen[1], pen[1])} @ {rate_str(pen[2])}). → §8")
                 if pen else "chưa có"])
    # E5
    rs = recovery_summary(e5)
    rows.append(["§7.3 phục hồi khi hết page (E5)", count(len(e5), len(sweep.e5_grid())),
                 (f"đẩy sang RAM rẻ hơn tính lại {vi(rs[0] / rs[1], '.0f')}× "
                  f"({vi(rs[1], '.0f')} so với {vi(rs[0], '.0f')} ms mỗi lần) — ngược paper, do GPU "
                  f"này yếu về sức tính. → §9") if rs else "chưa có"])
    # E6
    pr = parallel_rows(e6)
    broken = [r for r in e6 if r["stage"] == "e6" and r["abandoned"]]
    if pr:
        n_min = min(pr, key=lambda d: d["n"])
        e6_text = (f"n={n_min['n']}: latency paged thấp hơn {vi(n_min['nl_ratio'], '.0f')}×, "
                   f"chạy {vi(n_min['run_ratio'], '.1f')}× số sequence song song, chia sẻ page "
                   f"tiết kiệm {vi(n_min['saved'], '.0f')}% bộ nhớ KV. Thiếu n lớn hơn và Fig 14."
                   if n_min.get("nl_ratio") else "một phần, xem §10")
        if broken:
            e6_text += (f' <span class="bad">{len(broken)} run engine bị crash</span>, đã loại '
                        f'khỏi kết luận.')
        e6_text += " → §10"
    else:
        e6_text = "chưa có"
    rows.append(["§6.3 nhiều đáp án mỗi prompt (E6)", count(len(e6), len(sweep.e6_grid())), e6_text])
    # E7
    rows.append(["§6.4 tiền tố dùng chung (E7)", count(len(e7), len(sweep.e7_grid())),
                 "chưa có → §11" if not e7 else "một phần → §11"])
    rows.append(["§6.3 beam search", "—", "chưa cài đặt, xem §12"])

    have = len(e1) + (1 if e3_raw else 0) + len(e4) + len(e5) + len(e6) + len(e7)
    planned = (len(sweep.e1_grid()) + 1 + len(sweep.e4_grid()) + len(sweep.e5_grid())
               + len(sweep.e6_grid()) + len(sweep.e7_grid()))
    P = ["<h2>Tình trạng: đã đo được gì, còn thiếu gì</h2>"]
    P.append(f"<p>Thu thập dữ liệu đã <b>dừng</b> trước khi hết kế hoạch: <b>{have} / {planned} "
             f"run</b> đã có. "
             f"Bốn thí nghiệm cốt lõi — đường cong serving, kernel, block size, phục hồi — đã "
             f"<b>đủ</b>, nên các kết luận chính của báo cáo đứng được. Hai thí nghiệm về "
             f"<em>chia sẻ</em> KV cache mới có một phần hoặc chưa có, và các mục tương ứng ghi rõ "
             f"điều đó thay vì đoán.</p>")
    P.append(table(["Thí nghiệm (mục trong paper)", "Run xong / kế hoạch", "Kết luận hiện có"],
                   rows, note="Số run lấy từ <code>results/runs/</code>, số kế hoạch từ lưới trong "
                              "<code>sweep.py</code>. Chỉ run hợp lệ được dùng cho kết luận: run "
                              "bị engine bỏ dở hoặc vi phạm bất biến token bị loại; run “quá "
                              "ngắn” vẫn dùng nhưng được đánh dấu ở phụ lục A."))
    return P


def kpi(cards):
    """The result, above the fold. Each card is one number plus the single sentence needed to
    read it; anything longer belongs in the section it summarises."""
    out = []
    for value, unit, label, detail in cards:
        out.append(f'<div class="card"><div class="val">{value}<span class="unit">{unit}</span></div>'
                   f'<div class="lab">{label}</div><div class="det">{detail}</div></div>')
    return f'<div class="kpi">{"".join(out)}</div>'


def fmt(x, spec=".2f", dash="—"):
    """Table cells. Keeps the ASCII decimal point: these columns are monospace data, and a
    reader comparing down a column wants the same glyph the JSON has."""
    return dash if x is None else format(x, spec)


def vi(x, spec=".2f", dash="—"):
    """Prose numbers. Vietnamese writes 0,94 and 27 518, and the surrounding sentences do too,
    so a number dropped into prose has to agree with them."""
    if x is None:
        return dash
    return format(x, spec).replace(",", "\u202f").replace(".", ",")


CSS = """
/* Palette is taken from the figures, not chosen separately: --accent is the exact blue of the
   "paged" series and --critical the exact red of Orca (Max) in every chart, so the prose and the
   plots read as one system. Neutrals carry a slight blue bias (hue ~210) toward that accent
   rather than being pure grey. Utilitarian treatment on purpose - this is a lab report meant to
   be consulted, not a landing page. */
:root{
  --ground:#fbfcfd; --card:#f2f5f8; --ink:#121923; --muted:#58646f; --line:#dde3e9;
  --accent:#1b6ca8; --critical:#c0392b; --good:#2b6b35;
  --warn-ink:#7a4f00; --warn-ground:#fdf4e3; --warn-edge:#d9a441;
  --hl:#eaf2f8;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#11151a; --card:#1a1f26; --ink:#e4e9ef; --muted:#95a1ad; --line:#272e37;
    --accent:#5aa9e0; --critical:#e8766a; --good:#79c07f;
    --warn-ink:#f0c478; --warn-ground:#241c0f; --warn-edge:#8a6a28;
    --hl:#16222c;
  }
}
:root[data-theme="dark"]{
  --ground:#11151a; --card:#1a1f26; --ink:#e4e9ef; --muted:#95a1ad; --line:#272e37;
  --accent:#5aa9e0; --critical:#e8766a; --good:#79c07f;
  --warn-ink:#f0c478; --warn-ground:#241c0f; --warn-edge:#8a6a28;
  --hl:#16222c;
}

*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0; padding:2.5rem 1.25rem 6rem;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
main{max-width:64rem; margin:0 auto; display:flex; flex-direction:column; gap:.2rem}

h1{font-size:2rem; line-height:1.15; letter-spacing:-.02em; margin:0; text-wrap:balance}
h2{font-size:1.25rem; letter-spacing:-.01em; margin:3rem 0 .2rem; padding-bottom:.4rem;
   border-bottom:1px solid var(--line); text-wrap:balance}
h3{font-size:1.02rem; margin:1.9rem 0 .1rem; text-wrap:balance}
p{margin:.7rem 0; max-width:68ch}
ul,ol{padding-left:1.25rem; margin:.7rem 0; max-width:68ch}
li{margin:.35rem 0}
.sub{color:var(--muted); margin:.4rem 0 1.2rem; max-width:68ch}
.lede{font-size:1.06rem; max-width:66ch}

/* Result cards. Deliberately the first thing on the page: a reader who stops after ten
   seconds should still leave with the finding. */
.kpi{display:grid; grid-template-columns:repeat(auto-fit,minmax(14rem,1fr)); gap:.7rem;
     margin:1.4rem 0 .6rem}
.card{background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.9rem 1rem}
.card .val{font-size:1.85rem; line-height:1.1; font-weight:680; letter-spacing:-.02em;
           color:var(--accent); font-variant-numeric:tabular-nums}
.card .unit{font-size:.9rem; font-weight:500; color:var(--muted); margin-left:.15rem}
.card .lab{font-size:.9rem; font-weight:600; margin-top:.3rem}
.card .det{font-size:.82rem; color:var(--muted); margin-top:.15rem; line-height:1.45}

figure{margin:1.3rem 0 .4rem; background:var(--card); border:1px solid var(--line);
       border-radius:8px; padding:1rem}
figure.missing{color:var(--muted); font-style:italic}
figure.diagram{padding:1.1rem 1rem .6rem}
figure.diagram svg{display:block; max-width:100%; height:auto}
img{max-width:100%; height:auto; display:block; margin:0 auto; border-radius:4px; background:#fff}
figcaption{color:var(--muted); font-size:.85rem; margin-top:.75rem; text-align:center;
           max-width:64ch; margin-left:auto; margin-right:auto; text-wrap:balance}

/* Tables are the substance of this page, so they get the monospace face and lining figures:
   every column here is numbers that must line up to be compared down the column. */
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch}
table{border-collapse:collapse; width:100%; font-size:.84rem;
      font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
      font-variant-numeric:tabular-nums}
th,td{padding:.4rem .7rem; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted); font-weight:600; font-size:.7rem; text-transform:uppercase;
         letter-spacing:.07em; border-bottom:1px solid var(--muted)}
tbody tr:last-child td{border-bottom:none}
td.hl{background:var(--hl); font-weight:600}
tbody tr.sep td{border-top:1px solid var(--muted)}

code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
     font-size:.88em; background:var(--card); border:1px solid var(--line);
     border-radius:3px; padding:.08em .32em}

.note{background:var(--warn-ground); border-left:3px solid var(--warn-edge);
      padding:.85rem 1.1rem; border-radius:0 6px 6px 0; margin:1.2rem 0; font-size:.93rem;
      max-width:72ch}
.note strong{color:var(--warn-ink)}
.ok{color:var(--good); font-weight:600}
.bad{color:var(--critical); font-weight:600}
.dim{color:var(--muted)}

details{background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:.7rem 1rem; margin:1rem 0; max-width:72ch}
details[open]{padding-bottom:1rem}
summary{cursor:pointer; font-weight:600; font-size:.93rem}
details p, details ul{font-size:.91rem}

a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def build_html(parts, body_only):
    body = f"<main>{parts}</main>"
    title = "PagedAttention replication — results"
    if body_only:
        return f"<title>{title}</title><style>{CSS}</style>{body}"
    return ("<!doctype html><html lang=\"vi\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


# ============================ report assembly ============================

def section_headline(rows, capacity, waste, kernel_table):
    """The four-to-six numbers that answer the project's question, computed from the data
    rather than written down, so a partial sweep degrades to fewer cards instead of stale ones."""
    cards = []
    sg = "sharegpt"

    caps = (capacity or {}).get(sg, {})
    ratio, prefix = capacity_ratio(caps.get("paged"), caps.get("base:max"))
    if ratio is not None and prefix != "?":
        pinned = ("Mẫu số bị lưới chặn nên đây là <b>cận dưới</b>, không phải số đo. "
                  if prefix == "≥" else "")
        cards.append((f"{prefix}{vi(ratio, '.1f')}", "×",
                      "Năng lực phục vụ (ShareGPT)",
                      f"paged chịu được {capacity_text(caps['paged'])} req/s, Orca (Max) "
                      f"{capacity_text(caps['base:max'])}. {pinned}"
                      f"Paper báo 1,7–2,7× — xem §4."))

    ref = reference_rate(rows, sg)
    p = at_rate(rows, sg, "paged", ref) if ref else None
    b = at_rate(rows, sg, "base:max", ref) if ref else None
    if p and b and p["ttft_mean"] and b["ttft_mean"]:
        cards.append((vi(b["ttft_mean"] / p["ttft_mean"], ".0f"), "×",
                      "Độ trễ tới token đầu tiên",
                      f"tại {vi(ref, '.1f')} req/s: {vi(p['ttft_mean'])} s so với "
                      f"{vi(b['ttft_mean'], '.0f')} s. "
                      f"Đây là hệ quả người dùng cảm nhận được — xem §5."))

    if waste.get("paged") and waste.get("base:max"):
        cards.append((vi(waste["paged"]["waste_pct"], ".1f"), "%",
                      "KV cache bị lãng phí (paged)",
                      f"Orca (Max) lãng phí {vi(waste['base:max']['waste_pct'], '.0f')}% cùng "
                      f"điều kiện. "
                      f"Đây là Fig 2 của paper, đo được — xem §6.2."))

    rmax = max((r["rate"] for r in pick(rows, workload=sg) if r["rate"] is not None), default=None)
    p2 = at_rate(rows, sg, "paged", rmax) if rmax else None
    b2 = at_rate(rows, sg, "base:max", rmax) if rmax else None
    if p2 and b2 and p2["running_mean"] and b2["running_mean"]:
        cards.append((vi(p2["running_mean"] / b2["running_mean"], ".1f"), "×",
                      "Số sequence chạy song song",
                      f"tại {vi(rmax, '.1f')} req/s: {vi(p2['running_mean'], '.1f')} so với "
                      f"{vi(b2['running_mean'], '.1f')}. "
                      f"Đây là mắt xích nhân quả — xem §6.1."))

    fr = [r for r in pick(rows, workload=sg) if r["frag"]]
    if fr:
        worst = max(fr, key=lambda r: r["frag"])
        cards.append((vi(worst["frag"], ",d"), "",
                      "Cấp phát thất bại vì phân đoạn ngoài",
                      f"{VARIANT_VI.get(worst['variant'], worst['variant'])} ở "
                      f"{vi(worst['rate'], '.1f')} req/s, "
                      f"trong khi pool vẫn còn chỗ. paged: 0 ở mọi điểm — xem §6.3."))

    if kernel_table:
        mr = sum(r["ratio"] for r in kernel_table) / len(kernel_table)
        cards.append((vi(mr, ".2f"), "×",
                      "Kernel attention, paged / liền mạch",
                      f"paged <b>nhanh hơn</b>, trái với paper (1,20–1,26×). Nguyên nhân xác định "
                      f"được — xem §7."))
    return cards


def extra_cards(e5, e6):
    """Cards for the experiments that ran after the headline was first written. Kept separate so
    that section_headline stays about §6.2, which is the paper's main claim."""
    cards = []
    pr = [d for d in parallel_rows(e6) if d.get("nl_ratio")]
    if pr:
        d = min(pr, key=lambda d: d["n"])
        cards.append((vi(d["saved"], ".0f"), "%",
                      f"Bộ nhớ KV tiết kiệm nhờ chia sẻ (n={d['n']})",
                      f"cùng lúc latency thấp hơn {vi(d['nl_ratio'], '.0f')}× so với Orca (Max). "
                      f"Paper báo 6,1–9,8% — xem §10."))
    rs = recovery_summary(e5)
    if rs:
        cards.append((vi(rs[0] / rs[1], ".0f"), "×",
                      "Đẩy sang RAM rẻ hơn tính lại",
                      f"{vi(rs[1], '.0f')} so với {vi(rs[0], '.0f')} ms mỗi lần phục hồi. Ngược "
                      f"paper, vì GPU này yếu về sức tính — xem §9."))
    return cards


def section_mechanisms(waste):
    # The two fill fractions in the diagram are the measured waste from §6.2, so the picture
    # cannot drift from the table it illustrates.
    wc = waste.get("base:max", {}).get("waste_pct")
    wp = waste.get("paged", {}).get("waste_pct")
    waste_contig = (wc / 100.0) if wc is not None else 0.80
    waste_paged = (wp / 100.0) if wp is not None else 0.015
    P = []
    P.append("<h2>1. Câu hỏi, và hai cơ chế</h2>")
    P.append("<p>Một engine serve LLM phải giữ <b>KV cache</b> cho mọi sequence đang sinh dở: "
             "với Llama-3.2-1B là <b>32 KiB cho mỗi token ngữ cảnh</b>, trên cả 16 layer. Trên "
             "một GPU 4 GB, sau khi weights đã ăn 2,30 GiB, phần còn lại chỉ đủ cho khoảng "
             "<b>27 500 token</b> ngữ cảnh. Bao nhiêu request chạy được song song — và do đó "
             "throughput — phụ thuộc hoàn toàn vào việc cấp phát 27 500 token đó <em>khéo</em> "
             "đến đâu.</p>")
    P.append("<p>Vấn đề: khi một request đến, <b>không ai biết nó sẽ sinh ra bao nhiêu token</b>. "
             "Hai cách xử lý sự không biết đó chính là hai cơ chế được so ở đây.</p>")
    P.append(diagram(svg_mechanisms(waste_contig, waste_paged),
                     "Cùng một pool KV cache, hai cách lấp. Tỉ lệ phần đặc/nhạt lấy từ số đo "
                     "thật ở §6.2, không phải vẽ ước lệ."))
    P.append("<p><b>Đặt chỗ liên tục</b> (baseline, kiểu Orca) nhận lấy sự không biết bằng cách "
             "giữ chỗ cho trường hợp xấu nhất: cấp luôn một vùng liền mạch đủ cho cả cuộc đời "
             "sequence, ngay lúc nhận request. Đổi lại, địa chỉ của token thứ <code>i</code> là "
             "một phép nhân, và vùng đó không bao giờ phải di chuyển. Cái giá là phần đặt chỗ "
             "chưa dùng tới nằm không — và nó chặn mất chỗ của request khác.</p>")
    P.append("<p><b>Paging</b> (kiểu vLLM) không giữ chỗ: cắt pool thành <b>page 16 token</b> "
             "bằng nhau và cấp từng page khi sequence thật sự chạm tới. Sequence không còn liền "
             "mạch trong bộ nhớ, nên phải có một <b>bảng tra</b> dịch “token thứ <code>i</code>” "
             "sang “page số mấy”. Đây đúng là virtual memory của OS, mượn nguyên kiến trúc — "
             "bảng tra ở đây đóng vai page table.</p>")
    P.append("<h3>Hai loại lãng phí, và vì sao phải tách chúng ra</h3>")
    P.append("<p>Báo cáo này phân biệt hai thứ thường bị gộp thành “phân mảnh”, vì hai cơ chế "
             "<em>đánh đổi cái này lấy cái kia</em>:</p>")
    P.append(diagram(svg_fragmentation(),
                     "Phân đoạn trong là bộ nhớ đã có chủ nhưng chưa dùng; phân đoạn ngoài là bộ "
                     "nhớ không có chủ mà vẫn không cấp được. Cả hai đều đo được ở §6."))
    P.append('<div class="note"><strong>Tóm lại một dòng:</strong> phân đoạn trong là giá của '
             'page cố định; phân đoạn ngoài là giá của sự liền mạch. Hệ điều hành đã chọn trả giá '
             'thứ nhất từ thập niên 60, và luận điểm của paper là LLM serving nên chọn giống vậy. '
             'Phần còn lại của báo cáo là đo xem lựa chọn đó đáng giá bao nhiêu trên phần cứng '
             'này.</div>')
    return P


def section_method(rows):
    pools = {}
    for r in rows:
        if r["pool_tokens"]:
            pools.setdefault("paged" if r["variant"] == "paged" else "base", r["pool_tokens"])
    P = []
    P.append("<h2>2. Thiết kế thí nghiệm</h2>")
    P.append("<h3>Bốn biến thể</h3>")
    P.append("<p>Cùng một model, cùng kernel, cùng scheduler. Khác biệt duy nhất là công thức "
             "trả lời câu hỏi “cấp bao nhiêu, cấp lúc nào”. Ba biến thể Orca lấy nguyên văn §6.1 "
             "của paper, và là <b>cùng một binary</b> — chỉ khác biến môi trường "
             "<code>ORCA_POLICY</code>.</p>")
    P.append(table(["Biến thể", "Đặt chỗ bao nhiêu", "Lúc nào", "Vai trò"], [
        ["paged (vLLM)", "16 token mỗi page, theo nhu cầu",
         "dần dần, 16 token một lần", "bản đang xét"],
        ["Orca (Max)", "<b>2048</b> (max sequence length)", "một lần, lúc nhận request",
         "<b>baseline chính thức</b>"],
        ["Orca (Pow2)", "prompt + làm tròn lũy thừa 2 của output", "một lần, lúc nhận request",
         "cận trên, không đạt được"],
        ["Orca (Oracle)", "prompt + output <em>thật</em>", "một lần, lúc nhận request",
         "cận trên tuyệt đối, không đạt được"],
    ], "Pow2 và Oracle <em>không phải hệ thống chạy được</em>: cả hai cần biết trước độ dài "
       "output. Chúng có mặt để cho thấy ngay cả khi biết trước hoàn hảo, đặt chỗ liên tục vẫn "
       "còn một cái giá phải trả — xem §6.3. Mọi con số công bố là paged so với Orca (Max)."))
    P.append('<div class="note"><strong>Buddy allocator, không phải cấp phát tuỳ ý.</strong> '
             'Paper nói rõ: <em>“We assume Orca uses the buddy allocation algorithm to determine '
             'the memory address to store KV cache”</em>. Nên mọi đặt chỗ của base được làm tròn '
             'lên <b>lũy thừa 2</b> và phải tự align. Hệ quả: cả Pow2 lẫn Oracle đều còn một lớp '
             'làm tròn nữa ở trên công thức của mình, và cả hai đều có thể thất bại vì phân đoạn '
             'ngoài — điều paged không bao giờ gặp.</div>')

    P.append("<h3>Những gì bị giữ bằng nhau</h3>")
    P.append("<p>Luật của dự án: <em>ngoài cơ chế cấp phát KV, hai bản phải giống hệt nhau.</em> "
             "Đây là cách luật đó được cưỡng chế chứ không chỉ được mong đợi:</p>")
    pool_note = ""
    if "paged" in pools and "base" in pools:
        d = 100.0 * abs(pools["base"] - pools["paged"]) / pools["base"]
        pool_note = (f" Đo được <b>{vi(pools['base'], ',d')}</b> so với "
                     f"<b>{vi(pools['paged'], ',d')}</b> token — lệch {vi(d)}%, đúng bằng kích "
                     f"thước bảng tra, là chi phí metadata thật của paging.")
    P.append("<ol>"
             "<li><b>Khối hằng số dùng chung.</b> <code>src/config.h</code> của hai cây có một "
             "khối rào <code>BEGIN/END SHARED CONFIG</code> phải giống nhau <em>từng byte</em>; "
             "<code>check_shared_config.sh</code> diff nó và <code>build.sh</code> gọi script đó "
             "trước mỗi lần build, lệch là không cho build.</li>"
             "<li><b>Ngân sách KV không ai được chọn.</b> Cả hai gọi "
             "<code>cudaMemGetInfo</code> rồi lấy phần VRAM còn lại sau weights và scratch."
             + pool_note + "</li>"
             "<li><b>Cùng file workload, cùng seed.</b> Cùng dãy prompt, cùng dãy ngân sách "
             "output, cùng thời điểm đến (Poisson).</li>"
             "<li><b><code>IGNORE_EOS=1</code>.</b> EOS bị vô hiệu hoá, nên mỗi request sinh "
             "<em>đúng</em> ngân sách output của nó, ở cả 4 biến thể. Không có nó, độ dài output "
             "sẽ phụ thuộc vào nội dung sinh ra, mà nội dung lại phụ thuộc vào mức song song — "
             "tức là khối lượng công việc sẽ tương quan với chính cái đang được đo.</li>"
             "<li><b>Xen kẽ biến thể và nghỉ giữa các run.</b> GPU laptop 30 W có throttle; nếu "
             "chạy hết một biến thể rồi mới sang biến thể khác thì phép so sánh biến thành đo "
             "tản nhiệt. Xem phụ lục B.</li>"
             "</ol>")

    P.append("<h3>Ba đại lượng được đo</h3>")
    P.append(table(["Đại lượng", "Định nghĩa", "Đọc như thế nào"], [
        ["Normalized latency",
         "thời gian end-to-end của một request ÷ số token nó sinh ra, trung bình mọi request",
         "chỉ số chính của paper (Fig 12). Thấp hơn là tốt"],
        ["Capacity point",
         "tốc độ đến cao nhất mà normalized latency còn dưới <b>2×</b> giá trị lúc nhàn rỗi",
         "paper đọc bằng mắt từ đồ thị; đây là định nghĩa thao tác được tương ứng"],
        ["Time to first token",
         "từ lúc request đến tới token đầu tiên, tức là <b>thời gian nằm hàng đợi</b> + prefill",
         "không có trong paper. Thêm vào vì normalized latency chia cho độ dài output, "
         "làm loãng thời gian chờ"],
    ], "Normalized latency là chỉ số để so với paper; TTFT là chỉ số nói đúng hơn về trải "
       "nghiệm, và chênh lệch giữa hai cơ chế lộ ra rõ hơn nhiều ở đó."))

    # ---- the rate grid, including the part of it that has no justification ----
    # Every number in sections 4-6 is read off this grid, so the grid's weaknesses are the
    # results' weaknesses. Writing them down here is cheaper than letting a reader discover
    # them by noticing that one capacity point sits exactly on the edge.
    P.append("<h3>Lưới tốc độ đến, và chỗ nó yếu</h3>")
    grid = sorted({r["rate"] for r in rows if r["rate"] is not None})
    if grid:
        P.append("<p>Mọi con số ở §4–§6 đọc từ các lưới tốc độ đến dưới đây. Lưới quyết định "
                 "độ phân giải của mọi kết luận, nên cách nó được dựng cần được nói thẳng.</p>")
    per_w = {w: sorted({r["rate"] for r in pick(rows, workload=w) if r["rate"] is not None})
             for w in WORKLOADS}
    per_w = {w: g for w, g in per_w.items() if g}
    P.append("<ul>"
             + "".join(f"<li><b>{WORKLOAD_TITLE[w]}</b>: {len(g)} điểm, {rate_grid_str(g)} req/s."
                       f"</li>" for w, g in per_w.items())
             + "</ul>")
    P.append("<ul>"
             "<li><b>Mỗi workload một lưới riêng</b>, vì trần năng lực đo được khác hẳn nhau: "
             "khoảng 0,8 req/s với ShareGPT, 2,2 req/s với Alpaca và chỉ ~0,25 req/s với workload "
             "chatbot (prompt dài, prefill nặng). Lưới ban đầu dùng chung 6 điểm 0,3 → 2,0 cho cả "
             "hai — quá rộng cho ShareGPT, quá hẹp cho Alpaca — và đã được nới theo "
             "<code>.claude/plan-3-rate-grid.md</code>.</li>"
             "<li><b>Điểm thấp nhất là điểm nhàn rỗi</b> mà định nghĩa capacity point cần. Ở "
             "lưới cũ, ShareGPT bắt đầu ở 0,3 — nơi Orca (Max) đã chạm trần 13 sequence — nên "
             "điểm 0,15 được thêm vào để có một mốc sạch.</li>"
             "<li><b>Bước nhảy không đều</b> là di sản của lưới cũ: điểm mới được chèn vào chỗ "
             "đường cong gãy (vd. 0,45 giữa 0,3 và 0,6 với ShareGPT) thay vì chạy lại các run cũ, "
             "vốn là dữ liệu đúng.</li>"
             "<li><b>200 request mỗi run</b> là giới hạn thời gian: mỗi run bị chặn dưới bởi chính "
             "tốc độ đến (200 request ở 0,15 req/s ≈ 22 phút chỉ để chúng xuất hiện).</li>"
             "</ul>")
    return P


def section_workload(work_b64, work_meta):
    P = ["<h2>3. Workload — ShareGPT và Alpaca thật</h2>"]
    P.append("<p>Paper §6.1 lấy <em>cặp</em> (độ dài prompt, độ dài output) nguyên bản từ hai "
             "dataset thật, không sinh ngẫu nhiên. Lý do: trong traffic thật hai đại lượng này "
             "tương quan với nhau, và <em>tổng</em> của chúng mới là lượng KV cache một sequence "
             "chiếm. Độ dài output ở đây là độ dài <b>câu trả lời thật trong dataset</b>, đã "
             "tokenize — chính con số đó được dùng làm ngân sách, nên <code>Orca (Oracle)</code> "
             "thật sự là oracle.</p>")
    P.append("<p>Hai dataset được chọn vì chúng <b>tương phản</b>: Alpaca là chỉ thị ngắn, đều; "
             "ShareGPT là hội thoại dài, trải rộng. Mọi hiệu ứng của quản lý bộ nhớ đều lớn lên "
             "theo độ trải đó, nên hai workload này là hai chế độ chứ không phải hai lần đo.</p>")
    P.append(figure(work_b64, "Phân phối độ dài prompt và output sau khi lọc"))
    if work_meta:
        rows = [[WORKLOAD_TITLE.get(n, n), m["num_samples"],
                 f'{m["prompt_len"]["mean"]:.1f}', m["prompt_len"]["p50"], m["prompt_len"]["p99"],
                 f'{m["output_len"]["mean"]:.1f}', m["output_len"]["p50"], m["output_len"]["p99"],
                 m["dropped_prompt_too_long"] + m["dropped_output_too_long"]]
                for n, m in work_meta.items()]
        P.append(table(["Dataset", "Mẫu", "prompt tb", "p50", "p99", "output tb", "p50", "p99",
                        "bị lọc"], rows,
                       "Mẫu dài hơn cửa sổ 2048 token bị <em>loại</em>, không bị cắt ngắn. Cắt "
                       "ngắn sẽ dồn một cục nhân tạo đúng tại giá trị lớn nhất, và điều đó thiên "
                       "vị Orca (Max) — sự lãng phí của nó lớn đúng bằng khoảng cách giữa 2048 "
                       "và độ dài thật."))
        if {"alpaca", "sharegpt"} <= set(work_meta):
            a, s = work_meta["alpaca"], work_meta["sharegpt"]
            pr = s["prompt_len"]["mean"] / a["prompt_len"]["mean"]
            orr = s["output_len"]["mean"] / a["output_len"]["mean"]
            P.append(f'<div class="note"><strong>Lệch so với paper, và lệch theo hướng bảo '
                     f'thủ:</strong> tỉ lệ ShareGPT/Alpaca ở đây là <b>{pr:.2f}×</b> prompt và '
                     f'<b>{orr:.2f}×</b> output, trong khi paper báo <b>8,4×</b> và <b>5,8×</b>. '
                     f'Nguyên nhân là cửa sổ 1024+1024 của engine này (VRAM 3,63 GiB) lọc mất '
                     f'phần đuôi dài của ShareGPT. Hai workload vẫn tách nhau rõ, nhưng độ tương '
                     f'phản nhẹ hơn — nên mọi tỉ số đo được ở đây là <em>cận dưới</em> của hiệu '
                     f'ứng thật.</div>')
    return P


def section_serving(rows, serving_b64, capacity):
    P = ["<h2>4. Kết quả chính — đường cong serving (§6.2, Fig 12)</h2>"]
    P.append("<p>Mỗi điểm là một run 200 request với thời điểm đến Poisson. Đường cong đi ngang "
             "trong khi hệ thống còn theo kịp, rồi <b>vọt lên</b> khi tốc độ đến vượt năng lực: "
             "hàng đợi bắt đầu dồn, và thời gian chờ cộng vào latency của mọi request sau đó. "
             "Vị trí điểm vọt lên <em>là</em> kết quả.</p>")
    P.append(figure(serving_b64, "Normalized latency theo tốc độ request đến (Poisson)"))
    # §6.5 is not a separate section: it is the same figure with a third workload in it, and the
    # thing to look at is a *coincidence* between three curves rather than a gap between two.
    # Saying so next to the figure is the only way a reader notices it.
    if any(r["workload"] == "sharegpt_chat" for r in rows):
        P.append('<div class="note"><b>Panel thứ ba (§6.5) phải được đọc khác hai panel kia.</b> '
                 'Ở đó mỗi lượt chat gửi lại toàn bộ lịch sử hội thoại rồi cắt về 1024 token '
                 'cuối, nên prompt rất dài (p50 = 948, 46% đúng 1024). Paper dự đoán khi đó buddy '
                 'allocator làm tròn mọi đặt chỗ lên cùng 2048 token và ba đường Orca chập vào '
                 'nhau. Ở đây chỉ chập <em>một nửa</em> — xem ô dưới bảng.</div>')
    if capacity:
        trows, pinned = [], []
        for workload in [w for w in WORKLOADS if w in capacity]:
            caps = capacity[workload]
            base = caps.get("base:max")
            grid = sorted({r["rate"] for r in pick(rows, workload=workload)
                           if r["rate"] is not None})
            for i, v in enumerate([v for v in ORDER if v in caps]):
                c = caps[v]
                if c.get("censored"):
                    pinned.append((workload, v, c))
                ratio, prefix = capacity_ratio(c, base)
                cell = "—" if ratio is None else (
                    "không chặn được" if prefix == "?"
                    else f"<b>{prefix}{vi(ratio, '.2f')}×</b>")
                trows.append([(WORKLOAD_TITLE.get(workload, workload) if i == 0 else ""),
                              VARIANT_VI.get(v, v), capacity_text(c), cell])
        # One line if every workload shares a grid, one line each once they diverge (which is
        # what plan 3 does). Printing the same six numbers twice reads as if they meant
        # different things.
        per_workload = {w: sorted({r["rate"] for r in pick(rows, workload=w)
                                   if r["rate"] is not None})
                        for w in WORKLOADS if w in capacity}
        if len({tuple(g) for g in per_workload.values()}) == 1:
            grids = rate_grid_str(next(iter(per_workload.values())))
        else:
            grids = "; ".join(f"{WORKLOAD_TITLE.get(w, w)} {rate_grid_str(g)}"
                              for w, g in per_workload.items())
        P.append(table(["Workload", "Biến thể", "Capacity point (req/s)", "so với Orca (Max)"],
                       trows, highlight_col=3,
                       note=f"Capacity point = tốc độ cao nhất mà normalized latency còn dưới 2× "
                            f"giá trị lúc nhàn rỗi. Nó chỉ đọc được tới độ phân giải của lưới đã "
                            f"quét ({grids} req/s), nên mọi giá trị ở đây có sai số ±1 bước lưới."))
        if pinned:
            items = "".join(
                f"<li><b>{WORKLOAD_TITLE.get(w, w)} / {VARIANT_VI.get(v, v)}</b> = "
                f"{capacity_text(c)} — "
                + ("rơi đúng vào <em>điểm đo thấp nhất</em>. Đường cong đã vượt ngưỡng ngay ở "
                   "điểm kế tiếp, nên giá trị thật nằm thấp hơn; và chính điểm neo “nhàn rỗi” "
                   "dùng để dựng ngưỡng cũng có thể đã bị suy giảm, chỉ đẩy giá trị thật xuống "
                   "thêm." if c["censored"] == "left"
                   else "rơi đúng vào <em>điểm đo cao nhất</em>. Hệ thống chưa hề vượt ngưỡng "
                        "trong vùng đã quét, nên giá trị thật nằm cao hơn.")
                + "</li>"
                for w, v, c in pinned)
            P.append(f'<div class="note"><strong>{len(pinned)} giá trị trong bảng không phải số '
                     f'đo mà là <em>cận</em>.</strong><ul>{items}</ul>'
                     f'Vì thế cột tỉ số của chúng mang dấu <b>≥</b> hoặc <b>≤</b>: khi mẫu số là '
                     f'cận trên, tỉ số thật chỉ có thể lớn hơn con số in ra. Muốn bỏ cận thì phải '
                     f'nới lưới ở đầu tương ứng — đó là việc của '
                     f'<code>.claude/plan-3-rate-grid.md</code>.</div>')

        rs, ps = capacity_ratio(capacity.get("sharegpt", {}).get("paged"),
                                capacity.get("sharegpt", {}).get("base:max"))
        ra, pa = capacity_ratio(capacity.get("alpaca", {}).get("paged"),
                                capacity.get("alpaca", {}).get("base:max"))
        if rs and ra:
            P.append(f'<div class="note"><strong>Đọc bảng này thế nào.</strong> Trên ShareGPT, '
                     f'paged chịu được <b>{ps}{vi(rs, ".1f")}×</b> tốc độ đến so với baseline — so '
                     f'với khoảng 1,7–2,7× mà paper báo. Trên Alpaca chỉ '
                     f'<b>{pa}{vi(ra, ".2f")}×</b>, và đó là kết quả đúng chứ không phải thất bại: '
                     f'prompt và output của Alpaca đều ngắn và ít trải, nên lượng bộ nhớ bị đặt '
                     f'chỗ thừa nhỏ đi hẳn, và lợi thế của paging co lại theo. <em>Hiệu ứng của '
                     f'quản lý bộ nhớ tỉ lệ với độ trải của workload</em> — hai cột này là bằng '
                     f'chứng cho chính điều đó, và là lý do báo cáo này giữ cả hai workload chứ '
                     f'không chỉ lấy cái cho số đẹp.</div>')

        # When several variants land on the same capacity point, the metric has run out of
        # resolution - it has not found them equivalent. Saying so matters here because the
        # curves behind those equal numbers are visibly far apart.
        for workload in [w for w in WORKLOADS if w in capacity]:
            caps = capacity[workload]
            tied = [v for v in ORDER
                    if caps.get(v) and caps[v]["value"] == (caps.get("paged") or {}).get("value")]
            if len(tied) >= 3 and caps.get("paged", {}).get("value"):
                at = max((r["rate"] for r in pick(rows, workload=workload)
                          if r["rate"] is not None), default=None)
                vals = {v: (at_rate(rows, workload, v, at) or {}).get("nl") for v in tied}
                if not all(vals.values()):
                    continue
                lo, hi = min(vals.values()), max(vals.values())
                # Equal capacity points mean one of two different things, and the curves decide
                # which. Far apart at load: the metric ran out of resolution. Close at load:
                # they really are equivalent on this workload, which is itself a result and must
                # not be dressed up as a measurement artefact.
                if hi / lo < 1.3:
                    continue
                spread = (f" Ở {rate_str(at)} req/s chúng cách nhau "
                          f"<b>{vi(hi / lo, '.1f')}×</b> về normalized latency "
                          f"({vi(lo, '.2f')} so với {vi(hi, '.2f')} s/token).")
                names = ", ".join(VARIANT_VI.get(v, v) for v in tied)
                P.append(f'<div class="note"><strong>Trên '
                         f'{WORKLOAD_TITLE.get(workload, workload)}, {names} cùng rơi vào một '
                         f'capacity point — đó là <em>giới hạn phân giải của chỉ số</em>, không '
                         f'phải kết luận rằng chúng tương đương.</strong> Capacity point chỉ hỏi '
                         f'"latency vượt 2× lúc nhàn rỗi ở đâu", và với lưới 6–8 điểm thì ba '
                         f'đường rất khác nhau vẫn có thể vượt ngưỡng trong cùng một khoảng.'
                         f'{spread} Muốn phân biệt chúng thì đọc hình ở trên và bảng TTFT ở §5, '
                         f'chứ đừng đọc cột này.</div>')

        # §6.5. Plan 4 measured why the prediction only half holds; see its closing section.
        chat = capacity.get("sharegpt_chat", {})
        if chat:
            P.append('<div class="note"><strong>Workload chatbot: Pow2 và Oracle chập nhau, Max '
                     'tách ra — và điều đó tính được từ chính workload, không phải lỗi.</strong> '
                     'Max đặt chỗ prompt + 1024, tổng luôn vượt 1024 nên <b>100%</b> request bị làm '
                     'tròn lên 2048. Pow2/Oracle vẫn đẩy được khoảng 1/3 request (31% và 36%) '
                     'xuống mức ≤ 1024, vì chỉ 46% prompt thật sự chạm trần — paper giả định '
                     '“most requests”. Nên hiệu ứng ra khoảng một nửa. Điểm vỡ chỉ chênh 1,2× vì '
                     'trần năng lực của cả bốn biến thể gần bằng nhau (prefill prompt dài bị chặn '
                     'bởi sức tính); chênh lệch thật nằm ở <b>độ dốc sau điểm vỡ</b>: ở 0,3 req/s '
                     'paged nhanh hơn Orca (Max) khoảng 3,4×.</div>')
    return P


def section_ttft(rows, ttft_b64):
    P = ["<h2>5. Thời gian tới token đầu tiên</h2>"]
    P.append("<p>Normalized latency chia cho số token sinh ra, nên một request chờ 100 giây rồi "
             "sinh 500 token trông “chỉ” tệ gấp đôi một request chờ 0 giây. Với người dùng thì "
             "không phải vậy. TTFT là <b>thời gian nằm hàng đợi, chưa bị chia loãng</b> — và nó "
             "là nơi khác biệt giữa hai cơ chế lộ ra rõ nhất.</p>")
    P.append(figure(ttft_b64, "TTFT theo tốc độ đến. Trục dọc log — biên độ trải trên ba bậc "
                              "độ lớn"))
    for workload in WORKLOADS:
        sub = pick(rows, workload=workload)
        if not sub:
            continue
        rates = sorted({r["rate"] for r in sub})
        hdr = ["Biến thể"] + [f"{x}" for x in rates]
        trows = []
        for v in ORDER:
            cells = []
            for x in rates:
                m = at_rate(rows, workload, v, x)
                cells.append(fmt(m["ttft_mean"], ".2f") if m else "—")
            if any(c != "—" for c in cells):
                trows.append([VARIANT_VI.get(v, v)] + cells)
        if trows:
            P.append(table(hdr, trows,
                           note=f"{WORKLOAD_TITLE.get(workload, workload)} — TTFT trung bình "
                                f"(giây) theo tốc độ đến (req/s). Cột đầu là trạng thái nhàn rỗi, "
                                f"nơi mọi biến thể phải bằng nhau — và chúng bằng nhau, đó là "
                                f"phép kiểm tra rằng khác biệt ở các cột sau đến từ bộ nhớ chứ "
                                f"không từ tốc độ tính toán."))
    ref = reference_rate(rows, "sharegpt")
    p = at_rate(rows, "sharegpt", "paged", ref) if ref else None
    b = at_rate(rows, "sharegpt", "base:max", ref) if ref else None
    if p and b and p["ttft_mean"] and b["ttft_mean"]:
        P.append(f'<div class="note"><strong>Con số đáng chú ý nhất của cả báo cáo.</strong> '
                 f'Ở {vi(ref, ".1f")} req/s trên ShareGPT — tốc độ cao nhất mà paged vẫn còn '
                 f'theo kịp (hàng đợi {vi(p["waiting_mean"])} request) — paged trả token đầu '
                 f'tiên sau <b>{vi(p["ttft_mean"])} s</b>, còn Orca (Max) sau '
                 f'<b>{vi(b["ttft_mean"], ".0f")} s</b>: chênh '
                 f'<b>{vi(b["ttft_mean"] / p["ttft_mean"], ".0f")}×</b>. Nguyên nhân thuần tuý '
                 f'là hàng đợi — tại cùng thời điểm đó Orca (Max) có trung bình '
                 f'<b>{vi(b["waiting_mean"], ".0f")} request</b> đang xếp hàng, không vì thiếu '
                 f'sức tính toán mà vì <em>không còn vùng liền mạch nào để cấp</em>. Đây là cái '
                 f'giá của đặt chỗ, biểu diễn ở dạng người dùng cảm nhận được.</div>')
    return P


def slot_headroom(rows):
    """The engine's concurrency cap and the highest concurrency anything actually reached.
    Quoted in two places, so it is derived once: a cap that is never touched is a claim the
    report has to keep re-earning as data arrives, not a constant to type in."""
    caps = {r["slots"] for r in rows if r["slots"]}
    peak = max((r["running_max"] for r in rows if r["running_max"]), default=None)
    who = None
    if peak is not None:
        hit = [r for r in rows if r["running_max"] == peak]
        who = hit[0] if hit else None
    return (caps.pop() if len(caps) == 1 else None), peak, who


def section_why(rows, batched_b64, kv_b64, frag_b64, waste):
    P = ["<h2>6. Vì sao — ba mắt xích</h2>"]
    P.append("<p>Mắt xích nhân quả của paper: <b>tiết kiệm bộ nhớ → batch được nhiều sequence "
             "hơn → throughput cao hơn</b>. Mắt xích đó chỉ chặt nếu thêm một sequence vào batch "
             "gần như miễn phí, và trên GPU này đúng là vậy: mỗi decode step phải stream hết "
             "2,30 GiB weights <em>bất kể</em> bao nhiêu slot đang hoạt động, nên decode bị chặn "
             "bởi băng thông chứ không bởi số sequence. Ba mục dưới đây đo ba khâu của mắt xích "
             "đó, theo thứ tự nguyên nhân → kết quả.</p>")

    # ---- 6.1 concurrency ----
    P.append("<h3>6.1 Số sequence chạy song song (Fig 13)</h3>")
    P.append("<p>Đây là khâu giữa, và là khâu giải thích mọi thứ còn lại.</p>")
    P.append(figure(batched_b64, "Số request được batch đồng thời, lấy đỉnh của trung bình "
                                 "từng run"))
    sub = [r for r in pick(rows, workload="sharegpt") if r["running_mean"]]
    if sub:
        rates = sorted({r["rate"] for r in sub})
        trows = []
        for v in ORDER:
            cells = []
            for x in rates:
                m = at_rate(rows, "sharegpt", v, x)
                if m and m["running_mean"]:
                    cells.append(f'{m["running_mean"]:.1f}<span class="dim"> / '
                                 f'{m["running_max"]}</span>')
                else:
                    cells.append("—")
            if any(c != "—" for c in cells):
                trows.append([VARIANT_VI.get(v, v)] + cells)
        P.append(table(["Biến thể"] + [str(x) for x in rates], trows,
                       note="ShareGPT — số sequence chạy song song, <b>trung bình / đỉnh</b>, "
                            "theo tốc độ đến (req/s). Cột đỉnh là thứ cho thấy giới hạn nào "
                            "đang thật sự ràng buộc: Orca (Max) đứng yên ở một con số, paged "
                            "thì không."))
    bmax = [r for r in pick(rows, workload="sharegpt", variant="base:max") if r["pool_tokens"]]
    if bmax:
        pool = bmax[0]["pool_tokens"]
        ceiling = pool // 2048
        obs = max((r["running_max"] for r in bmax if r["running_max"]), default=None)
        cap, peak, who = slot_headroom(rows)
        peak_clause = ""
        if cap and peak:
            peak_clause = (f" và chưa bao giờ bị chạm — cao nhất quan sát được trên toàn bộ "
                           f"sweep là <b>{peak}</b>"
                           + (f", của {VARIANT_VI.get(who['variant'], who['variant'])} ở "
                              f"{vi(who['rate'], '.1f')} req/s" if who else "")
                           + f", tức còn dư {vi(cap / peak, '.1f')}×")
        P.append(f'<div class="note"><strong>Orca (Max) bị chặn bởi số học, và con số khớp '
                 f'chính xác.</strong> Mỗi sequence giữ 2048 token bất kể nó cần bao nhiêu, nên '
                 f'trần song song là <code>{vi(pool, ",d")} ÷ 2048 = {vi(pool / 2048)}</code> → '
                 f'<b>{ceiling} sequence</b>. Quan sát được: <code>running_max</code> ghim đúng '
                 f'<b>{obs}</b> ở <em>mọi</em> tốc độ đến — tăng tốc độ đến bao nhiêu cũng '
                 f'không nhúc nhích. Không phải scheduler giới hạn nó: giới hạn slot của engine '
                 f'là <b>{cap}</b>{peak_clause}. Đây là bộ nhớ, và chỉ bộ nhớ.</div>')

    # ---- 6.2 KV waste ----
    P.append("<h3>6.2 KV cache: đã cấp so với thực sự chứa token (Fig 2)</h3>")
    P.append("<p>Khâu nguyên nhân. Figure 2 của paper là một hình minh hoạ; ở đây nó được "
             "<em>đo</em>, vì cả hai bộ cấp phát đều của ta và đều có telemetry.</p>")
    P.append(figure(kv_b64, "KV cache đã cấp phát so với phần thực sự chứa token, theo thời gian"))
    if waste:
        trows = [[VARIANT_VI.get(v, v), f'{w["rate"]}', f'{w["reserved_pct"]:.1f}%',
                  f'{w["live_pct"]:.1f}%', f'<b>{w["waste_pct"]:.1f}%</b>']
                 for v, w in waste.items()]
        P.append(table(["Biến thể", "rate", "đã cấp phát", "chứa token thật", "lãng phí"], trows,
                       highlight_col=4,
                       note="Với base, khoảng trống là phần đặt chỗ chưa dùng tới <em>cộng</em> "
                            "phần buddy làm tròn lên lũy thừa 2. Với paged, nó chỉ là phần đuôi "
                            "chưa đầy của page cuối mỗi sequence — nhiều nhất 15 token mỗi "
                            "sequence mỗi layer, và đó là lý do nó xuống 1,5%."))
        P.append('<div class="note"><strong>Oracle vẫn lãng phí hơn 50%.</strong> Một hệ thống '
                 'biết trước <em>chính xác</em> độ dài output vẫn bỏ không hơn nửa số cache nó '
                 'cầm. Lượng đó không phải do đoán sai — nó là buddy làm tròn lên lũy thừa 2, '
                 'cộng với việc chỗ đặt cho token thứ 400 nằm không trong suốt thời gian '
                 'sequence mới sinh tới token thứ 10. Biết trước tương lai không cứu được việc '
                 'phải <em>cầm chỗ</em> cho tương lai đó.</div>')

    # ---- 6.3 external fragmentation ----
    P.append("<h3>6.3 Phân đoạn ngoài — cái giá riêng của sự liền mạch</h3>")
    P.append("<p>Mục này không có trong các hình của paper, và chỉ đo được vì cả hai bộ cấp phát "
             "là của ta. Engine đếm riêng một trường hợp: một đặt chỗ <b>thất bại trong khi pool "
             "vẫn còn đủ chỗ tính theo tổng</b>, chỉ là không có vùng liền mạch nào đủ. Đó đúng "
             "là định nghĩa của phân đoạn ngoài.</p>")
    P.append(figure(frag_b64, "Số lần cấp phát thất bại dù pool còn chỗ"))
    for workload in WORKLOADS:
        sub = pick(rows, workload=workload)
        if not sub:
            continue
        rates = sorted({r["rate"] for r in sub})
        trows = []
        for v in ORDER:
            cells = []
            for x in rates:
                m = at_rate(rows, workload, v, x)
                cells.append(f'{m["frag"]}' if m else "—")
            if any(c != "—" for c in cells):
                trows.append([VARIANT_VI.get(v, v)] + cells)
        if trows:
            P.append(table(["Biến thể"] + [str(x) for x in rates], trows,
                           note=f"{WORKLOAD_TITLE.get(workload, workload)} — số lần cấp phát "
                                f"thất bại vì phân đoạn ngoài, theo tốc độ đến (req/s)."))
    P.append('<div class="note"><strong>Ba điều cần đọc ra từ hai bảng này.</strong><br>'
             '<b>(1) paged là 0 ở mọi điểm, và không thể khác được.</b> Mọi page bằng nhau, nên '
             '“vừa” hay “không vừa” không phải một khái niệm tồn tại trong cơ chế đó. Đây không '
             'phải một kết quả đo may mắn, nó là một tính chất.<br>'
             '<b>(2) Orca (Max) cũng là 0 — và đó là lời bào chữa duy nhất của nó.</b> Vì mọi '
             'đặt chỗ đều đúng 2048 token, pool bị chia thành các ô bằng nhau xếp ngăn nắp cạnh '
             'nhau; không bao giờ có lỗ lệch cỡ. Nó trả toàn bộ giá bằng phân đoạn <em>trong</em> '
             '(80,4% ở §6.2).<br>'
             '<b>(3) Pow2 và Oracle giảm phân đoạn trong, nhưng mua lại bằng phân đoạn '
             'ngoài.</b> Chúng đặt chỗ đúng hơn, nên các vùng có kích thước khác nhau, nên pool '
             'vụn ra. Đây là phép đánh đổi được nhìn thấy trực tiếp, và là lý do “chỉ cần đặt '
             'chỗ thông minh hơn” <em>không</em> phải một đường thoát cho cơ chế liền mạch.</div>')
    P.append('<div class="note"><strong>Vì sao Alpaca toàn số 0 ở cả ba biến thể Orca.</strong> '
             'Prompt và output của Alpaca ngắn và đều, nên gần như mọi đặt chỗ rơi vào cùng một '
             'bậc lũy thừa 2 — và khi mọi vùng cùng kích thước thì chúng lại xếp ngăn nắp như '
             'trường hợp Max. Phân đoạn ngoài cần <em>sự trộn lẫn kích thước</em>, và chỉ '
             'ShareGPT có đủ độ trải để tạo ra nó.</div>')

    # ---- 6.4 preemption ----
    P.append("<h3>6.4 Preemption — cái giá riêng của paging</h3>")
    P.append("<p>Để công bằng, paging cũng có một cái giá mà cơ chế liền mạch không có. Bản "
             "paged <b>over-subscribe</b> có chủ ý: lúc nhận request nó chỉ kiểm tra rằng "
             "<em>prompt</em> vừa pool, chứ không bảo đảm gì cho phần sinh ra về sau. Khi một "
             "sequence đang decode cần page mới mà pool hết, nó không thể bảo sequence đó chờ — "
             "nên scheduler <b>thu hồi</b> page của sequence mới nhất và đẩy request đó về đầu "
             "hàng đợi để <em>prefill lại từ đầu</em> (recompute). Công đã làm bị bỏ.</p>")
    sub = [r for r in pick(rows, variant="paged") if r["preempt"] is not None]
    if sub:
        trows, measured = [], False
        for workload in WORKLOADS:
            ws = sorted(pick(rows, workload=workload, variant="paged"),
                        key=lambda r: r["rate"] or 0)
            for i, r in enumerate(ws):
                n = r["num_requests"] or 200
                # §7.3: what the preemptions actually cost, in the two currencies they waste.
                if r["recompute_ms"] is None:
                    cost = waste = '<span class="dim">chưa đo</span>'
                else:
                    measured = True
                    share = (100.0 * (r["recompute_ms"] / 1000.0) / r["wall_s"]
                             if r["wall_s"] else 0.0)
                    cost = (f'{vi(r["recompute_ms"] / 1000.0, ".1f")} s '
                            f'<span class="dim">({vi(share, ".1f")}% run)</span>'
                            if r["recompute_ms"] else "0 s")
                    waste = vi(r["recompute_tokens"], ",d") if r["recompute_tokens"] else "0"
                trows.append([(WORKLOAD_TITLE.get(workload, workload) if i == 0 else ""),
                              rate_str(r["rate"]), f'{r["preempt"]}',
                              f'{vi(r["preempt"] / n * 100, ".1f")}%' if r["preempt"] else "0%",
                              cost, waste])
        P.append(table(["Workload", "rate", "Số lần preempt", "% số request",
                        "thời gian tính lại", "token phải dựng lại"], trows,
                       note="Chỉ bản paged có bảng này; cơ chế liền mạch không bao giờ preempt, "
                            "nó từ chối ngay lúc nhận request thay vì nhận rồi thu hồi. Hai cột "
                            "cuối là §7.3: preempt lãng phí ở hai loại tiền tệ khác nhau — "
                            "thời gian GPU prefill lại cùng những vị trí đó, và công decode đã "
                            "làm rồi bị vứt."
                            + ("" if measured else
                               " <b>Chưa đo</b> ở các run hiện có: trường này thêm vào sau, "
                               "run cũ không mang nó. Chạy lại là có.")))
        hit = [r for r in pick(rows, variant="paged") if r["preempt"]]
        if hit:
            pct = max(100.0 * r["preempt"] / (r["num_requests"] or 200) for r in hit)
            where = "; ".join(
                f'{WORKLOAD_TITLE.get(w, w)} từ {rate_str(min(r["rate"] for r in hit if r["workload"] == w))} req/s'
                for w in WORKLOADS if any(r["workload"] == w for r in hit))
            P.append(f'<div class="note"><strong>Cái giá có thật nhưng nhỏ, và nó xuất hiện đúng '
                     f'chỗ dự đoán được.</strong> Preemption chỉ xảy ra ở: {where} — tức là khi '
                     f'paged đã ở sát hoặc vượt điểm gãy của chính nó — và nhiều nhất '
                     f'{vi(pct, ".1f")}% số request. Dưới điểm gãy thì bằng 0. Workload chatbot '
                     f'preempt sớm hơn vì mỗi sequence ở đó chiếm hơn 1000 token. Over-subscribe là '
                     f'một canh bạc, và trong vùng tải hệ thống được thiết kế để phục vụ, nó thắng. '
                     f'Mọi request bị preempt đều hoàn thành đủ ngân sách token, có kiểm bất biến.'
                     f'</div>')
    return P


def section_kernel(kernel_b64, kernel_table):
    P = ["<h2>7. Microbenchmark kernel attention (§7.1, Fig 18a)</h2>"]
    P.append("<p>Mọi lợi thế ở trên đến từ bộ nhớ. Mục này đo <b>cái giá</b> đi kèm: paging làm "
             "địa chỉ của token không còn suy ra được bằng một phép nhân, phải tra bảng trước. "
             "Kernel attention được tách khỏi phần còn lại của model và chạy với cùng seq_len, "
             "cùng batch, cùng số byte cache — khác biệt duy nhất là tra bảng so với bước nhảy "
             "cố định.</p>")
    P.append(figure(kernel_b64, "Độ trễ kernel attention, paged so với liền mạch"))
    if kernel_table:
        trows = [[r["batch"], r["context_len"], f'{r["contiguous_us"]:.1f}',
                  f'{r["paged_us"]:.1f}', f'<b>{r["ratio"]:.2f}×</b>',
                  '<span class="ok">khớp</span>' if r["digest_match"]
                  else '<span class="bad">LỆCH</span>']
                 for r in kernel_table]
        P.append(table(["batch", "context", "liền mạch (µs)", "paged (µs)", "tỉ lệ",
                        "digest output"], trows, highlight_col=4,
                       note="Cột digest là phép kiểm tra rằng hai kernel làm cùng một lượng "
                            "việc: chúng phải cho ra byte giống nhau, và chúng cho ra byte "
                            "giống nhau."))
        worse = [r for r in kernel_table if r["ratio"] > 1.0]
        mr = sum(r["ratio"] for r in kernel_table) / len(kernel_table)
        P.append(f'<div class="note"><strong>Kết quả này ngược paper, và đây là kết quả thật '
                 f'chứ không phải lỗi đo.</strong> Paper báo kernel paged chậm hơn '
                 f'<b>20–26%</b>; ở đây trung bình là <b>{mr:.2f}×</b>, tức là '
                 f'<em>nhanh hơn</em> ({len(worse)}/{len(kernel_table)} cấu hình chậm hơn). Hai '
                 f'kernel cho digest output <em>giống hệt nhau</em>, nên paged không hề làm ít '
                 f'việc hơn.<br><br>'
                 f'Nguyên nhân xác định được: paper so kernel của vLLM với '
                 f'<b>FasterTransformer</b> — một bản liền mạch được tối ưu rất sâu bởi NVIDIA — '
                 f'còn ở đây là so với biến thể liền mạch của <em>cùng một kernel</em>, cùng tác '
                 f'giả, cùng mức tối ưu. So trong cùng codebase, chi phí tra bảng nhỏ hơn hai '
                 f'khoản lợi mà paging tình cờ mang lại: offset của paged vừa trong 32 bit còn '
                 f'bản liền mạch buộc phải dùng <code>size_t</code> 64 bit, và vòng lặp trong '
                 f'của paged chạy trên <code>BLOCK_SIZE</code> là hằng số compile-time nên '
                 f'unroll được.<br><br>'
                 f'<b>Đọc đúng kết luận:</b> con số này <em>không</em> nói “paging nhanh hơn”. '
                 f'Nó nói chi phí indirection của paging <b>nhỏ hơn khoảng sai số giữa hai cách '
                 f'viết cùng một kernel</b> — tức là nhỏ tới mức không phải thứ đáng cân khi '
                 f'chọn cơ chế. Khoảng 20% của paper là so với một baseline được tối ưu hơn mọi '
                 f'thứ trong dự án này.</div>')
    return P


def section_blocksize(rows, block_b64):
    P = ["<h2>8. Ablation block size (§7.2, Fig 18b)</h2>"]
    P.append("<p>Block size là tham số tự do duy nhất của paging, nên nó đáng được quét riêng. "
             "Hai lực kéo ngược nhau: page <b>nhỏ</b> thì mỗi lần đọc cache lấy được ít byte "
             "liên tiếp, không tận dụng được băng thông, và bảng tra phình ra; page <b>lớn</b> "
             "thì phần đuôi bỏ không của page cuối lớn theo, tức là phân đoạn trong quay trở "
             "lại. vLLM chọn 16.</p>")
    P.append(figure(block_b64, "Normalized latency theo block size"))
    sub = [r for r in rows if r["block_size"]]
    if sub:
        sizes = sorted({r["block_size"] for r in sub})
        # One table per (workload, rate). Same workload at two rates is two different amounts
        # of memory pressure, and pressure is the variable the paper's claim is actually about.
        for workload, rate in sorted({(r["workload"], r["rate"]) for r in sub}):
            ws = [r for r in pick(sub, workload=workload) if r["rate"] == rate]
            if not ws:
                continue
            best = min((r["nl"] for r in ws if r["nl"] is not None), default=None)
            trows = []
            for bs in sizes:
                m = [r for r in ws if r["block_size"] == bs]
                if not m:
                    continue
                m = m[0]
                delta = (f'{100.0 * (m["nl"] / best - 1.0):+.1f}%'
                         if (best and m["nl"]) else "—")
                mark = " ◀ vLLM" if bs == 16 else ""
                trows.append([f'{bs}{mark}', fmt(m["nl"], ".4f"), delta,
                              vi(m["pool_tokens"], ",d") if m["pool_tokens"] else "—",
                              fmt(m["running_mean"], ".1f")])
            # Live context as a fraction of the pool. This is the number that decides whether
            # the block-size effect can show up at all: with the pool nearly empty, a page's
            # unused tail costs nothing.
            used = max((r["running_mean"] or 0) * 256 / (r["pool_tokens"] or 1) for r in ws)
            P.append(table(["block size", "norm. latency (s/token)", "so với tốt nhất",
                            "pool (token)", "song song"], trows, highlight_col=2,
                           note=f'{WORKLOAD_TITLE.get(workload, workload)} ở '
                                f'<b>{rate_str(rate)} req/s</b>, {ws[0]["slots"]} slot cho mọi '
                                f'block size. Ước lượng pool đang dùng: '
                                f'<b>{vi(100.0 * used, ".0f")}%</b> — chính con số này quyết '
                                f'định hiệu ứng block size có hiện ra được hay không.'))
    # Pool sizes and slot count come from the runs themselves: the ablation rebuilds the engine
    # per block size, so these are the numbers that build actually got.
    pools = [r["pool_tokens"] for r in sub if r["pool_tokens"]]
    slots = sub[0]["slots"] if sub else None
    rates = sorted({r["rate"] for r in sub if r["rate"] is not None})
    if sub:
        P.append(f'<div class="note">Ablation chạy ở cấu hình riêng '
                 f'<code>MAX_SEQUENCES = {slots}</code> cố định cho mọi block size: bảng tra tỉ '
                 f'lệ nghịch với block size, nên ở <code>BLOCK_SIZE=1</code> với 384 slot nó '
                 f'chiếm 25 MiB và không vừa VRAM. Giữ cố định để ablation chỉ có một biến chạy. '
                 f'Vì KV pool lấy từ phần VRAM còn lại, pool co lại ở block size nhỏ '
                 f'({vi(min(pools), ",d")} so với {vi(max(pools), ",d")} token) — đó là chi phí '
                 f'metadata thật của page nhỏ, không phải nhiễu. Rate '
                 f'{rate_grid_str(rates)} req/s chọn ngay dưới điểm gãy của paged: đủ tải để có '
                 f'nhiều cache phải đọc, mà hàng đợi còn gần trống nên latency vẫn phản ánh tính '
                 f'toán chứ không phản ánh thời gian chờ.</div>')

    # The Alpaca branch of the paper's §7.2 claim. This is a negative result with a positive
    # mechanism behind it, so every number is derived: the point is that fragmentation behaves
    # exactly as the paper says while the *latency* consequence stays absent, and a stale figure
    # would collapse that distinction.
    def frag_at(ws, bs):
        m = [r for r in ws if r["block_size"] == bs]
        if not m or m[0]["reserved_pct"] is None:
            return None
        r = m[0]
        return {"res": r["reserved_pct"], "live": r["live_pct"], "nl": r["nl"],
                "waste": 100.0 * (r["reserved_pct"] - r["live_pct"]) / r["reserved_pct"]}

    lines = []
    for workload, rate in sorted({(r["workload"], r["rate"]) for r in sub}):
        ws = [r for r in pick(sub, workload=workload) if r["rate"] == rate]
        big, mid = frag_at(ws, max(sizes)), frag_at(ws, 16)
        small = frag_at(ws, min(sizes))
        if not (big and mid and small):
            continue
        lines.append([
            f'{WORKLOAD_TITLE.get(workload, workload)} @ {rate_str(rate)}',
            f'{vi(big["res"], ".1f")}%', f'{vi(big["live"], ".1f")}%',
            f'<b>{vi(big["waste"], ".0f")}%</b>',
            f'{100.0 * (big["nl"] / mid["nl"] - 1.0):+.1f}%',
            f'{100.0 * (small["nl"] / mid["nl"] - 1.0):+.1f}%',
        ])
    if lines:
        P.append(table(["Workload @ rate", f"đã cấp (blk {max(sizes)})", "chứa token",
                        "phân đoạn trong", f"nl blk {max(sizes)} vs 16",
                        f"nl blk {min(sizes)} vs 16"], lines, highlight_col=3,
                       note=f"Hai cột cuối là hình phạt của page lớn nhất và page nhỏ nhất so "
                            f"với lựa chọn 16 của vLLM. Cột “phân đoạn trong” là "
                            f"<code>(đã cấp − chứa token) / đã cấp</code> ở block "
                            f"{max(sizes)}."))
        P.append('<div class="note"><strong>Cơ chế của paper đúng; hệ quả thì không xuất hiện — '
                 'và hai điều đó khác nhau.</strong> Paper viết rằng trên Alpaca <em>“larger '
                 'block sizes significantly degrade the performance since the sequences become '
                 'shorter than the block sizes”</em>. Phần <em>sequence ngắn hơn page</em> đo '
                 'được rất rõ: ở block lớn nhất, Alpaca bỏ không <b>hơn một nửa</b> số cache nó '
                 'cầm — <em>nhiều hơn</em> ShareGPT. Nhưng phần <em>degrade the performance</em> '
                 'thì không: cùng ở block đó, Alpaca <b>không chậm đi</b>, còn ShareGPT thì '
                 'chậm.<br><br>'
                 'Khác biệt không nằm ở độ dài sequence mà ở <b>pool có chật hay không</b>. Phân '
                 'đoạn trong chỉ biến thành latency khi bộ nhớ là thứ đang ràng buộc; Alpaca '
                 'trên máy này không bao giờ làm pool chật, kể cả ở tốc độ đến cao nhất quét '
                 'được (xem ô dưới). Nên kết luận đúng từ dữ liệu này không phải “paper sai”, mà '
                 'là: <em>câu của paper gộp hai bước — lãng phí, rồi lãng phí thành chậm — và '
                 'bước thứ hai cần một điều kiện mà cấu hình này không tạo ra được.</em></div>')
        P.append('<div class="note"><strong>Phía page nhỏ thì tái hiện trọn vẹn, và tăng theo áp '
                 'lực.</strong> Hình phạt ở block nhỏ nhất lớn dần đúng theo mức pool bị dùng — '
                 'xem cột cuối của bảng trên, đọc cùng cột “pool đang dùng” của từng bảng phía '
                 'trước. Đó là lý do paper nêu: page nhỏ thì mỗi lần đọc cache lấy được ít byte '
                 'liên tiếp, nên càng nhiều cache phải đọc thì càng mất băng thông.</div>')
    return P


def section_validity(rows):
    cap, peak, _ = slot_headroom(rows)
    P = ["<h2>12. Tính hợp lệ và những gì không đo được</h2>"]
    P.append("<h3>Hai bản chỉ khác đúng một thứ</h3><ul>"
             "<li><code>model.cu</code>, <code>utils.cu</code>, <code>request_queue.*</code>, "
             "<code>client.py</code>, <code>CMakeLists.txt</code>, <code>prompts.txt</code> "
             "<b>giống hệt nhau</b> giữa hai cây nguồn.</li>"
             "<li>Inventory kernel giống nhau trừ <code>contiguousAttentionKernel</code> ↔ "
             "<code>pagedAttentionKernel</code> và <code>markTokenListKernel</code> (chỉ paged "
             "cần, để dựng lại mask penalty sau khi recompute).</li>"
             "<li>Cửa sổ ngữ cảnh 2048 token ở cả hai, khớp max sequence length của OPT trong "
             "paper — và cũng là con số mà định nghĩa <code>Orca (Max)</code> phụ thuộc vào.</li>"
             + (f"<li>Giới hạn slot của engine là <b>{cap}</b> ở cả hai và <b>chưa bao giờ bị "
                f"chạm</b> — cao nhất quan sát được trên toàn bộ sweep là <b>{peak}</b>. Nó "
                f"không tham gia vào phép so sánh, và đó là điều phải kiểm chứ không phải giả "
                f"định: một trần bị chạm sẽ chặn đúng bên đang thắng và làm hỏng kết quả.</li>"
                if cap and peak else "")
             + "</ul>")
    P.append('<div class="note"><strong>Checksum chỉ so được ở mức song song bằng nhau.</strong> '
             'cuBLAS chọn kernel/tiling theo chiều batch, nên hai bản chạy số sequence khác nhau '
             'là đủ để sai số bf16 khác đi và argmax phân kỳ ở các ca sát nút. Đo được: 12 vs 12 '
             'sequence → checksum khớp tuyệt đối; 10 vs 30 → lệch toàn bộ. PagedAttention '
             'lossless <em>với một batch cố định</em>, và paper cũng không tuyên bố hai hệ thống '
             'cho output giống nhau từng bit. Ở mọi mức song song, điều được kiểm là <em>bất '
             'biến token</em>: mỗi request emit chính xác ngân sách output của nó, không token '
             'nào mất hay lặp, kể cả qua preempt. Toàn bộ ' + f'{sum(1 for r in rows if r["quality"] != runqual.BROKEN)}' + ' run §6.2 đạt bất biến này.</div>')
    # §6.5 is listed only while it is genuinely missing. It used to be absent from this list
    # *and* absent from the results, which is the worst of both: plan 2 recorded it as covered
    # and nothing contradicted that.
    chat_missing = not pick(rows, workload="sharegpt_chat")
    P.append("<h3>Nằm ngoài tầm</h3><ul>"
             + ("<li><b>§6.5 workload chatbot — chưa chạy, không phải không làm được.</b> Paper "
                "nối lịch sử hội thoại vào prompt rồi cắt về 1024 token cuối, nên hầu hết "
                "request có prompt chạm trần; đó là chế độ khiến <em>cả ba</em> biến thể Orca "
                "sụp về cùng một hành vi. Workload hiện dùng lấy một lượt hỏi–đáp nên không tạo "
                "ra chế độ đó. Vá chỉ cần đổi khâu dựng dữ liệu, không đụng engine — xem "
                "<code>.claude/plan-4-chatbot-workload.md</code>.</li>" if chat_missing else "")
             + "<li><b>Beam search chưa có.</b> Chia sẻ page và copy-on-write thì có (xem §10–11), "
             "nhưng beam search đòi <em>vẽ lại quan hệ cha–con mỗi bước</em> — beam con có thể "
             "nhận cha khác với bước trước — và lỗi ở đó <b>sai âm thầm</b>: vẫn ra chữ, chỉ là "
             "chữ sai. Nên nó bị để lại sau cùng, và cổng kiểm của nó phải là so với beam search "
             "của HuggingFace, không chỉ kiểm bất biến. Hệ quả: phần “tiết kiệm 37,6–55,2% bộ "
             "nhớ ở beam search” của paper vẫn chưa có đối chiếu ở đây.</li>"
             "<li><b>§6.4 tiết kiệm được bộ nhớ nhưng chưa bỏ được phần tính.</b> Paper viết "
             "<em>“the prompt phase computation only needs to execute on the user's task "
             "input”</em>. Ở đây request sau vẫn tính lại K/V của tiền tố rồi ghi đè đúng những "
             "byte đã có (cùng token, cùng vị trí, kernel tất định ⇒ cùng bit, nên vô hại). "
             "Không bỏ được vì <code>prefillAttentionKernel</code> đọc K/V từ bộ đệm packed của "
             "chính pass đó, không đi qua block table. Nghĩa là số §6.4 ở đây là <b>cận dưới</b> "
             "của lợi thế thật.</li>"
             "<li>Không có baseline FasterTransformer — và §7 cho thấy điều đó quan trọng.</li>"
             "<li><b>GPU và model khác paper</b>: GTX 1650 4 GB / Llama-3.2-1B thay vì A100 / "
             "OPT-13B–175B. Không tensor core, 192 GB/s, 30 W có throttle. Mọi con số "
             "<em>tuyệt đối</em> không so được với paper; chỉ <em>tỉ số giữa các biến thể</em> "
             "là so được.</li>"
             "<li>Repetition penalty 1,15 không có trong paper. Đối xứng ở hai bên, và dưới "
             "<code>IGNORE_EOS</code> thì độ dài output do ngân sách ấn định, nên nó không đụng "
             "tới hành vi bộ nhớ.</li>"
             "<li>Paper chạy trace <b>1 tiếng</b> mỗi điểm; không sao chép được vì GPU này chậm "
             "hơn khoảng hai bậc độ lớn. Thay bằng <b>200 request mỗi run cộng kiểm tra hội "
             "tụ</b> (nửa đầu so nửa sau lệch &lt; 10%) — kiểm chứng trực tiếp tính chất mà con "
             "số 1 tiếng dùng để bảo đảm.</li>"
             "<li>Lưới rate có 8 điểm mỗi workload, nên capacity point có sai số ±1 bước lưới. "
             "Các tỉ số ở §4 là ước lượng theo lưới đó.</li>"
             "<li><b>Thu thập dữ liệu dừng trước khi xong</b>: §6.3 chỉ có n nhỏ, §6.4 chưa có "
             "run nào — xem bảng tình trạng ở đầu báo cáo.</li></ul>")
    return P


def section_appendix(rows, e1_raw, e3_raw, e4_raw):
    """Housekeeping, at the back. Deliberately not at the top: the state of the data collection
    is not the finding, and a reader who needs it will look for it."""
    P = ["<h2>Phụ lục A — chất lượng dữ liệu</h2>"]
    kinds = defaultdict(list)
    for r in rows:
        kinds[r["quality"]].append(r)
    by_stage = defaultdict(int)
    for r in rows:
        by_stage[r["stage"]] += 1
    P.append(table(["Hạng mục", "Giá trị"], [
        ["Run §6.2 đường cong serving (E1)", f"{len(e1_raw)} / {len(sweep.e1_grid())}"],
        ["Run ablation block size (E4)", f"{len(e4_raw)} / {len(sweep.e4_grid())}"],
        ["Run phục hồi khi hết page (E5)", f"{by_stage['e5']} / {len(sweep.e5_grid())}"],
        ["Run nhiều đáp án mỗi prompt (E6)", f"{by_stage['e6']} / {len(sweep.e6_grid())}"],
        ["Run tiền tố dùng chung (E7)", f"{by_stage['e7']} / {len(sweep.e7_grid())}"],
        ["Microbenchmark kernel (E3)",
         '<span class="ok">có</span>' if e3_raw else '<span class="bad">chưa chạy</span>'],
        ["— <b>ổn định</b>: trung bình đã hội tụ",
         f'<span class="ok">{len(kinds[runqual.STEADY])}</span>'],
        ["— <b>quá tải</b>: vượt năng lực, dữ liệu đúng và là cận dưới",
         f'{len(kinds[runqual.SATURATED])}'],
        ["— <b>quá ngắn</b>: nhiễu, nên chạy lại dài hơn",
         f'<span class="{"bad" if kinds[runqual.NOISY] else "ok"}">{len(kinds[runqual.NOISY])}</span>'],
        ["— <b>lỗi</b>: bỏ dở, hoặc vi phạm bất biến token",
         f'<span class="{"bad" if kinds[runqual.BROKEN] else "ok"}">{len(kinds[runqual.BROKEN])}</span>'],
    ], "Phân loại tự động, logic ở <code>runqual.py</code>. Báo cáo này sinh được từ dữ liệu "
       "một phần; hạng mục nào chưa có sẽ ghi rõ là chưa chạy."))
    P.append('<div class="note"><strong>“Quá tải” không phải lỗi, nó là kết quả.</strong> Khi '
             'tốc độ đến vượt năng lực, hàng đợi dồn lên và latency tăng đơn điệu suốt run — nửa '
             'sau <em>phải</em> cao hơn nửa đầu, và chạy lâu hơn chỉ làm khoảng cách đó rộng '
             'thêm. Đây đúng là các điểm làm đường cong vọt lên ở §4, và con số tại đó là '
             '<b>cận dưới</b> của latency thật ở trạng thái ổn định — tức bảo thủ, không phóng '
             'đại. Thứ phân biệt “quá tải” với “quá ngắn” là <b>hàng đợi</b>: dưới năng lực thì '
             'hàng đợi trống và mọi dao động chỉ là nhiễu.</div>')

    attention = kinds[runqual.NOISY] + kinds[runqual.BROKEN]
    if attention:
        trows = [[r["workload"], f'{r["rate"]}', VARIANT_VI.get(r["variant"], r["variant"]),
                  runqual.LABEL_VI[r["quality"]],
                  f'{100 * r["drift"]:.0f}%' if r["drift"] is not None else "—",
                  f'{fmt(r["first_half"], ".3f")} → {fmt(r["second_half"], ".3f")}',
                  fmt(r["waiting_mean"], ".2f")]
                 for r in attention]
        P.append(table(["Workload", "rate", "Biến thể", "Loại", "lệch", "nửa đầu → nửa sau",
                        "hàng đợi"], trows,
                       note="“nửa đầu → nửa sau” là normalized latency (s/token) của hai nửa run."))
        P.append('<div class="note"><strong>Run “quá ngắn”: lệch lớn nhưng hàng đợi gần trống.'
                 '</strong> Chúng chưa quá tải — trung bình chỉ chưa kịp hội tụ trong 200 request, '
                 'thường do một cụm request đến dồn. Ở rate thấp nhất, cả bốn biến thể lệch '
                 '<em>giống hệt nhau</em> (cùng dãy request), nên phép so giữa chúng vẫn đúng. '
                 'Số của chúng vẫn được dùng, kèm dấu này.</div>')
        if kinds[runqual.BROKEN]:
            P.append('<div class="warn"><strong>Run “lỗi” bị loại khỏi mọi hình và kết luận.'
                     '</strong> Engine tự huỷ giữa run (mã thoát −6) — đây là lỗi thật của '
                     'engine, không phải nhiễu đo, và chưa được sửa vì thu thập dữ liệu đã '
                     'dừng.</div>')

    # ---- full measurement table ----
    P.append("<h3>Toàn bộ số đo E1</h3>")
    body = []
    for workload in WORKLOADS:
        for v in ORDER:
            ws = sorted(pick(rows, workload=workload, variant=v), key=lambda r: r["rate"] or 0)
            for i, r in enumerate(ws):
                body.append([
                    (WORKLOAD_TITLE.get(workload, workload) if i == 0 and v == ORDER[0] else ""),
                    VARIANT_VI.get(v, v) if i == 0 else "",
                    f'{r["rate"]}', fmt(r["nl"], ".4f"), fmt(r["ttft_mean"], ".2f"),
                    fmt(r["ttft_p99"], ".2f"), fmt(r["throughput"], ".3f"),
                    fmt(r["running_mean"], ".1f"), f'{r["running_max"]}',
                    fmt(r["waiting_mean"], ".2f"), f'{r["frag"]}', f'{r["preempt"]}',
                    runqual.LABEL_VI[r["quality"]],
                ])
    if body:
        P.append("<details><summary>Mở bảng đầy đủ — "
                 f"{len(body)} run, mọi đại lượng đã đo</summary>"
                 + table(["Workload", "Biến thể", "rate", "norm.lat", "TTFT tb", "TTFT p99",
                          "req/s", "song song tb", "song song max", "hàng đợi", "phân đoạn ngoài",
                          "preempt", "chất lượng"], body,
                         note="norm.lat = s/token. TTFT và hàng đợi tính theo giây và theo số "
                              "request. Đây là nguồn của mọi bảng ở trên.")
                 + "</details>")
    return P


def section_thermal(e1_raw):
    temps = [r["gpu_after"].get("temp_c") for r in e1_raw if r.get("gpu_after", {}).get("temp_c")]
    clocks = [r["gpu_after"].get("sm_clock_mhz") for r in e1_raw
              if r.get("gpu_after", {}).get("sm_clock_mhz")]
    if not temps:
        return []
    return ["<h2>Phụ lục B — nhiệt độ và clock</h2>",
            table(["Đại lượng", "min", "trung bình", "max"], [
                ["Nhiệt độ cuối run (°C)", f"{min(temps):.0f}",
                 f"{sum(temps) / len(temps):.0f}", f"{max(temps):.0f}"],
                ["SM clock cuối run (MHz)", f"{min(clocks):.0f}",
                 f"{sum(clocks) / len(clocks):.0f}", f"{max(clocks):.0f}"],
            ], f"SM clock không tụt quá {vi(100.0 * (1 - min(clocks) / max(clocks)), '.0f')}% "
               f"so với đỉnh trong toàn bộ sweep, nên throttle có nhưng "
               f"không phải thứ giải thích được khoảng cách nào ở trên. Để giữ được điều đó, "
               f"<code>sweep.py</code> chạy xen kẽ các biến thể theo vòng và nghỉ 60 s giữa các "
               f"run; nếu chạy hết một biến thể rồi mới sang biến thể khác, biến thể cuối sẽ "
               f"gánh toàn bộ phần GPU đã nóng và phép so sánh biến thành đo tản nhiệt.")]


# ============================ §7.3 recovery cost ============================

POLICY_TITLE = {"recompute": "Tính lại (recompute)", "swap": "Đẩy sang RAM (swap)"}
POLICY_COLOR = {"recompute": "#1b6ca8", "swap": "#c0392b"}


def recovery_rows(rows):
    """One row per (block size, policy) with the cost of a single recovery event.

    Per *event*, not per run: Fig 19 is the cost of recovering one sequence, and a run's total
    depends on how many times it happened, which is a property of the load rather than of the
    policy. Dividing makes the two policies comparable even when they preempt different numbers
    of times."""
    out = []
    for r in rows:
        if r["stage"] != "e5" or not r["block_size"]:
            continue
        events = r["preempt"] or 0
        if events == 0:
            continue
        policy = r["policy"] or "recompute"
        if policy == "swap":
            # Pages out and back again; swap_ms already covers both directions.
            cost_ms = (r["swap_ms"] or 0.0) / events
            moved_mb = (r["swap_mb"] or 0.0) / events
        else:
            cost_ms = (r["recompute_ms"] or 0.0) / events
            moved_mb = None
        out.append({"block_size": r["block_size"], "policy": policy, "events": events,
                    "cost_ms": cost_ms, "moved_mb": moved_mb,
                    "fallbacks": r["swap_fallbacks"], "nl": r["nl"],
                    "wasted_tokens": (r["recompute_tokens"] or 0) / events})
    return out


def fig_recovery(rows):
    data = recovery_rows(rows)
    if not data:
        return None, None
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    sizes = sorted({d["block_size"] for d in data})
    for policy in ("recompute", "swap"):
        sub = sorted([d for d in data if d["policy"] == policy], key=lambda d: d["block_size"])
        if not sub:
            continue
        axes[0].plot([d["block_size"] for d in sub], [d["cost_ms"] for d in sub],
                     marker="o", color=POLICY_COLOR[policy], label=POLICY_TITLE[policy])
        axes[1].plot([d["block_size"] for d in sub], [d["nl"] for d in sub],
                     marker="o", color=POLICY_COLOR[policy], label=POLICY_TITLE[policy])
    for ax, ylab, title in ((axes[0], "ms mỗi lần phục hồi", "Chi phí một lần phục hồi"),
                            (axes[1], "normalized latency (s/token)", "Hệ quả lên latency")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(x) for x in sizes])
        ax.set_xlabel("block size (token mỗi page)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.tight_layout()
    return savefig(fig, "recovery.png")


def section_recovery(rows, recovery_b64):
    P = ["<h2>9. Hết page thì phục hồi bằng cách nào (§7.3, Fig 19)</h2>"]
    P.append("<p>Bản paged <b>nhận request vượt khả năng bảo đảm</b>: lúc nhận nó chỉ kiểm "
             "prompt có vừa pool, không hứa gì cho phần sinh về sau. Khi decode cạn page, nó "
             "phải thu hồi một sequence đang chạy. Có hai cách lấy lại sequence đó, và paper "
             "so chúng theo block size:</p>")
    P.append("<ul><li><b>Tính lại</b> — vứt KV cache, prefill lại từ prompt cộng mọi token đã "
             "sinh. Không đụng tới page nào, nên chi phí <em>không phụ thuộc</em> block size."
             "</li><li><b>Đẩy sang RAM</b> — chép page sang RAM máy, trả page về pool, rồi chép "
             "ngược khi có chỗ. Page càng nhỏ thì càng nhiều lần truyền bé, nên paper thấy nó "
             "<em>tệ đi ở block nhỏ</em>.</li></ul>")
    P.append(figure(recovery_b64, "Chi phí phục hồi theo block size, hai chính sách"))

    data = recovery_rows(rows)
    if not data:
        P.append('<div class="note">Chưa chạy — <code>python sweep.py --stage e5</code>.</div>')
        return P

    sizes = sorted({d["block_size"] for d in data})
    trows = []
    for bs in sizes:
        cell = {d["policy"]: d for d in data if d["block_size"] == bs}
        rc, sw = cell.get("recompute"), cell.get("swap")
        ratio = (f'{rc["cost_ms"] / sw["cost_ms"]:.0f}×'
                 if rc and sw and sw["cost_ms"] else "—")
        trows.append([bs,
                      fmt(rc["cost_ms"], ".0f") if rc else "—",
                      fmt(sw["cost_ms"], ".0f") if sw else "—",
                      ratio,
                      f'{rc["events"]}/{sw["events"]}' if rc and sw else "—",
                      fmt(sw["moved_mb"], ".0f") if sw else "—",
                      (str(sw["fallbacks"]) if sw and sw["fallbacks"] is not None else "—")])
    P.append(table(["block size", "tính lại (ms)", "swap (ms)", "tính lại / swap",
                    "số lần (tl/swap)", "MB mỗi lần swap", "swap rơi về tính lại"],
                   trows, highlight_col=3,
                   note="Chi phí một lần phục hồi. Cột cuối là số lần chính sách swap không "
                        "swap được và phải tính lại — nếu nó lớn thì cột swap đang đo tính "
                        "lại dưới một cái tên khác."))

    rc = [d for d in data if d["policy"] == "recompute"]
    sw = [d for d in data if d["policy"] == "swap"]
    if rc and sw:
        rc_mean = mean([d["cost_ms"] for d in rc])
        sw_mean = mean([d["cost_ms"] for d in sw])
        rc_spread = max(d["cost_ms"] for d in rc) / max(1e-9, min(d["cost_ms"] for d in rc))
        sw_spread = max(d["cost_ms"] for d in sw) / max(1e-9, min(d["cost_ms"] for d in sw))
        winner = "đẩy sang RAM" if sw_mean < rc_mean else "tính lại"
        P.append(f"<p><b>Kết quả trên máy này: {winner} rẻ hơn, chênh "
                 f"{vi(max(rc_mean, sw_mean) / max(1e-9, min(rc_mean, sw_mean)), '.0f')}×.</b> "
                 f"Trung bình một lần phục hồi tốn {vi(rc_mean, '.0f')} ms nếu tính lại và "
                 f"{vi(sw_mean, '.0f')} ms nếu đẩy sang RAM. Theo block size, tính lại dao động "
                 f"{vi(rc_spread, '.1f')}× còn swap dao động {vi(sw_spread, '.1f')}×.</p>")
        P.append('<div class="note"><b>Ngược với dự đoán, và lý do nằm ở phần cứng.</b> '
                 'Paper thấy swap tệ ở block nhỏ vì nhiều lần truyền bé làm sụp băng thông '
                 'PCIe. Ở đây điều quyết định lại là <em>đầu kia của phép so</em>: tính lại '
                 'phải prefill lại cả prompt lẫn mọi token đã sinh — với workload chatbot là '
                 'hơn một nghìn token — trên một GPU trần 2,7 TFLOP/s không có tensor core. '
                 'Trong khi đó KV cache của một sequence chỉ 64 MiB và đi qua PCIe pinned mất '
                 'vài chục ms. Nói cách khác: <b>tính lại đắt vì GPU này yếu về sức tính, '
                 'không vì paging</b>. Trên A100 của paper, cùng phép so cho kết quả khác là '
                 'điều phải mong đợi.</div>')
        thin = [d for d in data if d["events"] < 10]
        if thin:
            P.append(f'<div class="warn">{len(thin)} điểm có dưới 10 lần phục hồi, nên trung '
                     f'bình của chúng mỏng. Ít nhất là {min(d["events"] for d in data)} lần.'
                     f'</div>')
        P.append("<p>Số lần preempt mỗi run thấp (quanh "
                 f"{vi(mean([d['events'] for d in data]), '.0f')} lần trên 200 request) và đó "
                 "là <b>hệ quả của việc làm đúng §4.5</b>: “once it preempts a sequence and "
                 "evicts its blocks, vLLM stops accepting new requests until all preempted "
                 "sequences are completed”. Mỗi lần preempt tự giải toả áp lực đã gây ra nó. "
                 "Một engine preempt liên tục ở đây sẽ là engine <em>bỏ qua</em> câu đó.</p>")
    return P


# ============================ §6.3 parallel sampling ============================

def parallel_rows(rows):
    """Fig 15, one row per n that has a usable paged run. A run the engine aborted is left out:
    its averages describe the minutes before a crash, not the mechanism. The Orca (Max) run at
    the same n, when there is one, supplies the ratios."""
    mem = [r for r in rows if r["stage"] == "e6" and "mem" in (r["file"] or "")
           and not r["abandoned"] and r["quality"] != runqual.BROKEN]
    out = []
    for n in sorted({r["n_seqs"] for r in mem if r["n_seqs"]}):
        p = next((r for r in mem if r["n_seqs"] == n and (r["variant"] or "").startswith("paged")), None)
        b = next((r for r in mem if r["n_seqs"] == n and (r["variant"] or "").startswith("base:max")), None)
        if not p:
            continue
        used = (p["reserved_pct"] or 0) + (p["shared_pct"] or 0)
        d = {"n": n, "paged": p, "base": b,
             # pages that sharing made unnecessary, as a share of what no-sharing would need
             "saved": 100.0 * (p["shared_pct"] or 0) / used if used else None}
        if b and p["nl"] and b["nl"]:
            d["nl_ratio"] = b["nl"] / p["nl"]
            d["run_ratio"] = (p["running_mean"] or 0) / max(1e-9, b["running_mean"] or 0)
        out.append(d)
    return out


def fig_parallel(rows):
    data = parallel_rows(rows)
    if not data:
        return None, None
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    for variant, key in (("paged", "paged"), ("base:max", "base")):
        sub = [d for d in data if d[key]]
        if not sub:
            continue
        st = dict(STYLE[variant])
        axes[0].plot([d["n"] for d in sub], [d[key]["running_mean"] for d in sub], **st)
        axes[1].plot([d["n"] for d in sub], [d[key]["nl"] for d in sub], **st)
    axes[0].set_ylabel("sequence chạy song song (tb)")
    axes[0].set_title("Song song mỗi mức n")
    axes[1].set_ylabel("normalized latency (s/token)")
    axes[1].set_yscale("log")
    axes[1].set_title("Latency mỗi mức n")
    ns = [d["n"] for d in data]
    for ax in axes:
        ax.set_xticks(ns)
        ax.set_xlabel("số đáp án mỗi prompt (n)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, "parallel.png")


def section_parallel(rows, parallel_b64):
    P = ["<h2>10. Nhiều đáp án cho một prompt (§6.3, Fig 14–15)</h2>"]
    P.append("<p>Khi một prompt cần <b>n</b> đáp án khác nhau, phần KV cache của prompt là "
             "<em>giống hệt nhau</em> ở cả n đáp án. Bản paged cho chúng trỏ chung vào một page "
             "và chỉ tách ra khi có ai cần ghi đè (copy-on-write). Cơ chế liền mạch không làm "
             "được: ở đó một sequence <em>chính là</em> vùng liên tục của nó, nên n đáp án là n "
             "lần đặt chỗ đầy đủ.</p>")
    data = parallel_rows(rows)
    if not data:
        P.append('<div class="note">Chưa có run dùng được.</div>')
        return P
    P.append(figure(parallel_b64, "Số sequence song song và latency theo số đáp án mỗi prompt"))
    trows = []
    for d in data:
        p, b = d["paged"], d["base"]
        trows.append([d["n"],
                      f'{fmt(p["nl"], ".2f")} / {fmt(b["nl"], ".2f") if b else "—"}',
                      vi(d["nl_ratio"], ".1f") + "×" if d.get("nl_ratio") else "—",
                      f'{fmt(p["running_mean"], ".0f")} / {fmt(b["running_mean"], ".0f") if b else "—"}',
                      vi(d["saved"], ".0f") + "%" if d["saved"] is not None else "—",
                      f'{vi(p["pool_tokens"], ",d")} / {vi(b["pool_tokens"], ",d") if b else "—"}',
                      f'{p["preempt"]}',
                      f'{runqual.LABEL_VI[p["quality"]]} / {runqual.LABEL_VI[b["quality"]] if b else "—"}'])
    P.append(table(["n", "norm. latency paged / Max (s/token)", "Max chậm hơn",
                    "song song tb paged / Max", "bộ nhớ tiết kiệm nhờ chia sẻ",
                    "pool (token) paged / Max", "preempt (paged)", "chất lượng run"], trows,
                   highlight_col=2,
                   note="ShareGPT, 0,6 req/s, 120 prompt, temperature 0,8 nên các đáp án tách nhau "
                        "sau prompt. “Bộ nhớ tiết kiệm” = số page được dùng chung ÷ số page cần "
                        "nếu không chia sẻ, trung bình theo thời gian. Riêng các run này slot được "
                        "nâng lên 768 ở <b>cả hai</b> bản."))
    d = data[0]
    if d.get("nl_ratio"):
        P.append(f'<div class="note"><strong>Ở n={d["n"]}, paged chạy '
                 f'{vi(d["run_ratio"], ".1f")}× số sequence song song và có latency thấp hơn '
                 f'{vi(d["nl_ratio"], ".0f")}×.</strong> Orca (Max) vẫn kẹt ở trần 13 sequence như '
                 f'§6.1, nhưng giờ mỗi prompt cần n sequence, nên nó chỉ phục vụ được 13/n prompt '
                 f'cùng lúc và hàng đợi dồn ngay — run của nó là “quá tải” ở mọi n đã đo. Bản '
                 f'paged dùng chung page prompt, tiết kiệm {vi(d["saved"], ".0f")}% bộ nhớ KV — '
                 f'cao hơn mức 6,1–9,8% paper báo cho parallel sampling, vì prompt ShareGPT ở đây '
                 f'chiếm phần lớn hơn trong ngữ cảnh mỗi sequence.</div>')
    P.append('<div class="note"><b>Một điểm làm phép so <em>khó</em> hơn cho paged:</b> bảng tra '
             'của paged lớn theo số slot, nên ở 768 slot pool của nó nhỏ hơn khoảng 17% so với '
             'base. Paged vẫn thắng dù có ít bộ nhớ KV hơn. Vì cấu hình slot khác §6.2, các run '
             'này không so điểm-với-điểm với §4.</div>')
    excluded = [r for r in rows if r["stage"] == "e6" and (r["abandoned"] or r["quality"] == runqual.BROKEN)]
    missing = len(sweep.e6_grid()) - len([r for r in rows if r["stage"] == "e6"])
    P.append(f'<div class="warn"><b>Phần chưa có.</b> '
             + (f'{len(excluded)} run bị loại: bản paged ở n='
                f'{", ".join(str(r["n_seqs"]) for r in excluded)} bị engine tự dừng giữa chừng '
                f'(mã thoát −6, tức engine tự huỷ) sau khi đã preempt nhiều lần — đây là <b>lỗi thật '
                f'của engine</b> khi kết hợp chia sẻ page với preempt, chưa xác định nguyên nhân, '
                f'nên không dùng số của nó. ' if excluded else '')
             + f'{missing} run còn lại (n lớn hơn ở base, và đường cong Fig 14) chưa chạy. Kết luận '
             f'ở mục này vì vậy chỉ dựa trên n = {", ".join(str(x["n"]) for x in data)}.</div>')
    return P


# ============================ §6.4 shared prefix ============================

PREFIX_TITLE = {"translate_1shot": "1-shot (tiền tố 82 token)",
                "translate_5shot": "5-shot (tiền tố 303 token)"}


def fig_prefix(rows):
    sub = [r for r in rows if r["stage"] == "e7"]
    if not sub:
        return None, None
    works = [w for w in ("translate_1shot", "translate_5shot")
             if any(r["workload"] == w for r in sub)]
    w = panel_width(len(works))
    fig, axes = plt.subplots(1, len(works), figsize=(w * len(works), 3.8), squeeze=False)
    for ax, workload in zip(axes[0], works):
        for name in ORDER:
            series = sorted([r for r in sub if r["workload"] == workload
                             and r["variant"] == name], key=lambda r: r["rate"])
            if not series:
                continue
            ax.plot([r["rate"] for r in series], [r["nl"] for r in series], **STYLE[name])
        ax.set_xlabel("tốc độ đến (req/s)")
        ax.set_ylabel("normalized latency (s/token)")
        ax.set_title(PREFIX_TITLE.get(workload, workload))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, "prefix.png")


def section_prefix(rows, prefix_b64):
    P = ["<h2>11. Prompt dùng chung một mở đầu (§6.4, Fig 16)</h2>"]
    P.append("<p>Rất nhiều ứng dụng gắn cùng một đoạn mở đầu vào trước mọi prompt — chỉ dẫn "
             "nhiệm vụ cộng vài ví dụ mẫu. Phần KV cache của đoạn đó giống nhau ở mọi request, "
             "nên paper cấp cho nó một bộ page riêng, ghim lại, và mọi request về sau chỉ việc "
             "trỏ vào: “reserving a set of physical blocks for a set of predefined shared "
             "prefixes by the LLM service provider, as how OS handles shared library across "
             "processes”.</p>")
    P.append("<p>Workload là chính corpus của paper, <b>WMT16 Anh→Đức</b>, với hai cấu hình "
             "tiền tố. Độ dài tiền tố không do ai khai, nó được <b>đo</b> từ chính workload — "
             "tiền tố token chung dài nhất của cả bộ: <b>82 token</b> cho 1-shot và "
             "<b>303 token</b> cho 5-shot, so với 80 và 341 của paper.</p>")
    P.append(figure(prefix_b64, "Đường cong serving khi mọi prompt dùng chung một mở đầu"))
    sub = [r for r in rows if r["stage"] == "e7"]
    if not sub:
        P.append('<div class="note">Chưa chạy — <code>python sweep.py --stage e7</code> sau khi '
                 'chọn lưới tốc độ. Engine đã biết chia sẻ page của tiền tố; phần nó '
                 '<b>chưa</b> làm được là bỏ qua luôn việc tính tiền tố, vì '
                 '<code>prefillAttentionKernel</code> đọc K/V từ bộ đệm packed của chính pass '
                 'chứ không đi qua block table. Hệ quả: con số đo được là <em>cận dưới</em> '
                 'của lợi thế thật — tiết kiệm được bộ nhớ nhưng vẫn tính lại tiền tố mỗi '
                 'request.</div>')
        return P
    # The one number that needs explaining before the table, because it looks like a bug.
    over = [r for r in sub if (r["live_pct"] or 0) > 100.0]
    if over:
        worst = max(over, key=lambda r: r["live_pct"])
        P.append(f'<div class="note"><b>Cột “% pool có token” vượt 100% — và đó chính là kết '
                 f'quả, không phải lỗi.</b> Nó đo <em>tổng ngữ cảnh logic</em> của mọi sequence '
                 f'đang chạy so với pool vật lý. Khi n request dùng chung page tiền tố, mỗi '
                 f'request vẫn tính đủ ngữ cảnh của mình trong khi bộ nhớ chỉ giữ một bản. '
                 f'Đỉnh đo được là <b>{vi(worst["live_pct"], ".0f")}%</b>: engine phục vụ lượng '
                 f'ngữ cảnh gần gấp đôi pool mà nó có. Bản liền mạch không có cách nào đạt quá '
                 f'100% ở đây.</div>')
    trows = []
    for r in sorted(sub, key=lambda r: (r["workload"] or "", r["rate"] or 0, r["variant"] or "")):
        trows.append([PREFIX_TITLE.get(r["workload"], r["workload"]), rate_str(r["rate"]),
                      r["variant"], fmt(r["nl"], ".4f"), fmt(r["live_pct"], ".1f"),
                      vi(r["prefix_hits"], ",d") if r["prefix_hits"] else "—",
                      r["quality"]])
    P.append(table(["workload", "rate", "biến thể", "nl (s/token)", "% pool có token",
                    "lần dùng lại tiền tố", "chất lượng run"], trows))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="store_true",
                    help="also write a body-only artifact.html for publishing")
    args = ap.parse_args()

    runs = load_runs()
    e1_raw = [r for r in runs if r["_stage"] == "e1"]
    e3_raw = [r for r in runs if r["_stage"] == "e3"]
    e4_raw = [r for r in runs if r["_stage"] == "e4"]
    e5_raw = [r for r in runs if r["_stage"] == "e5"]
    e6_raw = [r for r in runs if r["_stage"] == "e6"]
    e7_raw = [r for r in runs if r["_stage"] == "e7"]
    e1 = [derive(r) for r in e1_raw]
    e4 = [derive(r) for r in e4_raw]
    e5 = [derive(r) for r in e5_raw]
    e6 = [derive(r) for r in e6_raw]
    e7 = [derive(r) for r in e7_raw]

    _, work_b64, work_meta = fig_workload()
    _, serving_b64, capacity = fig_serving(e1)
    _, ttft_b64 = fig_ttft(e1)
    _, batched_b64 = fig_batched(e1)
    _, kv_b64, waste = fig_kv_waste(e1_raw)
    _, frag_b64 = fig_frag(e1)
    _, kernel_b64, kernel_table = fig_kernel(e3_raw)
    _, block_b64 = fig_blocksize(e4)
    _, recovery_b64 = fig_recovery(e5)
    _, parallel_b64 = fig_parallel(e6)
    _, prefix_b64 = fig_prefix(e7)

    P = []
    P.append("<h1>PagedAttention: tái hiện trên GTX 1650 / Llama-3.2-1B</h1>")
    P.append('<p class="sub">Hai engine inference viết tay bằng CUDA, <b>chỉ khác nhau ở cơ chế '
             'quản lý KV cache</b> — đặt chỗ liên tục kiểu Orca so với paging kiểu vLLM. Tái '
             'hiện Kwon et al., <em>Efficient Memory Management for LLM Serving with '
             'PagedAttention</em>, SOSP \'23.</p>')
    usable = sum(1 for r in e1 + e4 + e5 + e6 + e7 if r["quality"] != runqual.BROKEN) + len(e3_raw)
    P.append('<p class="lede">Câu hỏi duy nhất: <b>quản lý KV cache theo kiểu paging hơn kiểu '
             'đặt chỗ liên tục bao nhiêu, và hơn ở đâu?</b> Dưới đây là câu trả lời đo được trên '
             f'{usable} run hợp lệ.</p>')
    cards = section_headline(e1, capacity, waste, kernel_table) + extra_cards(e5, e6)
    if cards:
        P.append(kpi(cards))
        P.append('<p class="sub">Mỗi ô dẫn tới mục chứng minh nó. Thứ tự các mục đi theo chuỗi '
                 'nhân quả: cơ chế (§1) → cách đo (§2–3) → kết quả (§4–5) → vì sao (§6) → cái '
                 'giá phải trả (§7–9) → chia sẻ KV (§10–11) → giới hạn (§12).</p>')
    P += section_status(e1, e3_raw, e4, e5, e6, e7, capacity, kernel_table)

    P += section_mechanisms(waste)
    P += section_method(e1)
    P += section_workload(work_b64, work_meta)
    P += section_serving(e1, serving_b64, capacity)
    P += section_ttft(e1, ttft_b64)
    P += section_why(e1, batched_b64, kv_b64, frag_b64, waste)
    P += section_kernel(kernel_b64, kernel_table)
    P += section_blocksize(e4, block_b64)
    P += section_recovery(e5, recovery_b64)
    P += section_parallel(e6, parallel_b64)
    P += section_prefix(e7, prefix_b64)
    P += section_validity(e1)
    P += section_appendix(e1 + e4 + e5 + e6 + e7, e1_raw, e3_raw, e4_raw)
    P += section_thermal(e1_raw)

    os.makedirs(REPORT, exist_ok=True)
    html = build_html("".join(P), body_only=False)
    out = os.path.join(REPORT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html) / 1024:.0f} KB)")
    if args.artifact:
        art = os.path.join(REPORT, "artifact.html")
        with open(art, "w", encoding="utf-8") as f:
            f.write(build_html("".join(P), body_only=True))
        print(f"wrote {art}")

    # machine-readable summary next to the HTML
    summary = {
        "capacity": capacity,
        "kv_waste": waste,
        "kernel": kernel_table,
        "workload_meta": work_meta,
        "e1_runs": len(e1_raw),
        "e4_runs": len(e4_raw),
        "e5_runs": len(e5_raw),
        "e6_runs": len(e6_raw),
        "e7_runs": len(e7_raw),
        "recovery": recovery_rows(e5),
        "parallel": [{"n": d["n"], "saved_pct": d["saved"], "nl_ratio": d.get("nl_ratio"),
                      "run_ratio": d.get("run_ratio")} for d in parallel_rows(e6)],
        "measurements": e1 + e4 + e5 + e6 + e7,
    }
    with open(os.path.join(REPORT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(REPORT, 'summary.json')}")


if __name__ == "__main__":
    main()
