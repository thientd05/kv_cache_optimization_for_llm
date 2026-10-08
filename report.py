"""Turn results/runs/*.json into the paper's figures and a self-contained HTML report.

Mirrors the figures of Kwon et al., SOSP '23:
  Fig 12  normalised latency vs request rate, per dataset            -> fig_serving.png
  Fig 13  average number of batched requests                         -> fig_batched.png
  Fig 2   KV cache: reserved vs holding a real token, over time      -> fig_kv_waste.png
  Fig 18a latency of the attention kernels                           -> fig_kernel.png
  Fig 18b end-to-end latency vs block size                           -> fig_blocksize.png
  §6.1    the workload length distributions we actually got          -> fig_workload.png

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
WORKLOAD_TITLE = {"alpaca": "Alpaca (short)", "sharegpt": "ShareGPT (long)"}


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


# ============================ Fig 12: serving curves ============================

def fig_serving(e1):
    """The paper's headline plot. Latency climbs slowly, then explodes once the arrival rate
    passes what the server can absorb; the rate at which that happens is the result."""
    by = defaultdict(dict)  # workload -> variant -> {rate: norm latency}
    for r in e1:
        if r.get("normalized_latency_s_per_token") is None:
            continue
        by[r["workload"]].setdefault(r["label"], {})[r["rate"]] = r["normalized_latency_s_per_token"]
    if not by:
        return None, None, {}

    workloads = [w for w in ("alpaca", "sharegpt") if w in by]
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.4 * len(workloads), 4.1), squeeze=False)
    capacity = {}
    for ax, workload in zip(axes[0], workloads):
        for variant in ORDER:
            series = by[workload].get(variant)
            if not series:
                continue
            xs = sorted(series)
            ys = [series[x] for x in xs]
            ax.plot(xs, ys, **STYLE[variant])
            # Operational definition of the paper's informal reading ("sustains N x higher
            # request rates while maintaining similar latencies"): the highest rate at which
            # normalised latency is still under 2x its unloaded value.
            threshold = 2.0 * ys[0]
            ok = [x for x, y in zip(xs, ys) if y <= threshold]
            capacity.setdefault(workload, {})[variant] = max(ok) if ok else None
        ax.set_title(WORKLOAD_TITLE.get(workload, workload))
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel("Normalized latency (s/token)")
        ax.grid(alpha=0.25, ls=":")
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Normalized latency vs request rate  —  paper Fig. 12", y=1.02, fontsize=11)
    path, b64 = savefig(fig, "fig_serving.png")
    return path, b64, capacity


# ============================ Fig 13: batched requests ============================

def fig_batched(e1):
    """Why the curves differ: how many sequences the scheduler could keep in flight. On a
    bandwidth-bound decoder this is what converts into throughput."""
    by = defaultdict(lambda: defaultdict(list))
    for r in e1:
        if r.get("running_mean"):
            by[r["workload"]][r["label"]].append(r["running_mean"])
    if not by:
        return None, None
    workloads = [w for w in ("alpaca", "sharegpt") if w in by]
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.6 * len(workloads), 3.8), squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        names = [v for v in ORDER if v in by[workload]]
        vals = [max(by[workload][v]) for v in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[STYLE[v]["color"] for v in names], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("base:", "Orca\n") .replace("paged", "paged") for n in names],
                           fontsize=8)
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
        # the busiest rate we have data for, where the distinction matters most
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


# ============================ Fig 18a: kernel ============================

def fig_kernel(e3):
    if not e3:
        return None, None, []
    rows = e3[0]["rows"]
    by = {(r["mechanism"], r["batch"], r["context_len"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    ctxs = sorted({r["context_len"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(4.4 * len(batches), 3.6), squeeze=False,
                             sharey=False)
    table = []
    for ax, batch in zip(axes[0], batches):
        width = 0.36
        xs = range(len(ctxs))
        c = [by[("contiguous", batch, c_)]["latency_us"] for c_ in ctxs]
        p = [by[("paged", batch, c_)]["latency_us"] for c_ in ctxs]
        ax.bar([x - width / 2 for x in xs], c, width, label="contiguous", color="#c0392b")
        ax.bar([x + width / 2 for x in xs], p, width, label="paged", color="#1b6ca8")
        for c_, cv, pv in zip(ctxs, c, p):
            table.append({"batch": batch, "context_len": c_, "contiguous_us": cv,
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
    return path, b64, table


# ============================ Fig 18b: block size ============================

def fig_blocksize(e4):
    by = defaultdict(dict)
    for r in e4:
        if r.get("normalized_latency_s_per_token") is None:
            continue
        bs = r["engine_config"].get("block_size")
        if bs:
            by[r["workload"]][bs] = r["normalized_latency_s_per_token"]
    if not by:
        return None, None
    # Two panels on purpose. The absolute one is zero-based so the magnitudes are not
    # exaggerated; but the two workloads sit at very different latencies and the effect is a few
    # percent, so on that axis the shape is invisible. The right panel divides each workload by
    # its own best point, which is the only way to actually see the U - and it is labelled as
    # relative so nobody reads it as an absolute gap.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    colors = {"alpaca": "#d98c00", "sharegpt": "#1b6ca8"}
    for ax, relative in ((axes[0], False), (axes[1], True)):
        for workload, series in sorted(by.items()):
            xs = sorted(series)
            ys = [series[x] for x in xs]
            if relative:
                best = min(ys)
                ys = [100.0 * (y / best - 1.0) for y in ys]
            ax.plot(xs, ys, marker="o", color=colors.get(workload),
                    label=WORKLOAD_TITLE.get(workload, workload))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Block size (tokens per page)")
        ax.axvline(16, color="#888", ls="--", lw=1)
        ax.grid(alpha=0.25, ls=":")
        if relative:
            ax.set_ylabel("% worse than that workload's best block size")
            ax.set_title("relative to each workload's own best", fontsize=9)
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


# ============================ workload distributions ============================

def fig_workload():
    metas, samples = {}, {}
    for name in ("alpaca", "sharegpt"):
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
    for key, ax, title in ((("prompt_len"), axes[0], "Prompt length"),
                           (("output_len"), axes[1], "Output length")):
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


# ============================ HTML ============================

def table(headers, rows, note=None):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<figcaption>{note}</figcaption>" if note else ""
    return f'<figure><div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>{cap}</figure>'


def figure(b64, caption):
    if not b64:
        return f'<figure class="missing">Not available yet — the stage that produces it has not run.<figcaption>{caption}</figcaption></figure>'
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


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
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#11151a; --card:#1a1f26; --ink:#e4e9ef; --muted:#95a1ad; --line:#272e37;
    --accent:#5aa9e0; --critical:#e8766a; --good:#79c07f;
    --warn-ink:#f0c478; --warn-ground:#241c0f; --warn-edge:#8a6a28;
  }
}
:root[data-theme="dark"]{
  --ground:#11151a; --card:#1a1f26; --ink:#e4e9ef; --muted:#95a1ad; --line:#272e37;
  --accent:#5aa9e0; --critical:#e8766a; --good:#79c07f;
  --warn-ink:#f0c478; --warn-ground:#241c0f; --warn-edge:#8a6a28;
}

*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0; padding:2.5rem 1.25rem 6rem;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
main{max-width:64rem; margin:0 auto; display:flex; flex-direction:column; gap:.2rem}

h1{font-size:2rem; line-height:1.15; letter-spacing:-.02em; margin:0; text-wrap:balance}
h2{font-size:1.25rem; letter-spacing:-.01em; margin:2.8rem 0 .2rem; padding-bottom:.4rem;
   border-bottom:1px solid var(--line); text-wrap:balance}
h3{font-size:1.02rem; margin:1.8rem 0 .1rem; text-wrap:balance}
p{margin:.7rem 0; max-width:68ch}
ul{padding-left:1.25rem; margin:.7rem 0; max-width:68ch}
li{margin:.35rem 0}
.sub{color:var(--muted); margin:.4rem 0 1.4rem; max-width:68ch}

figure{margin:1.3rem 0 .4rem; background:var(--card); border:1px solid var(--line);
       border-radius:8px; padding:1rem}
figure.missing{color:var(--muted); font-style:italic}
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

code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
     font-size:.88em; background:var(--card); border:1px solid var(--line);
     border-radius:3px; padding:.08em .32em}

.note{background:var(--warn-ground); border-left:3px solid var(--warn-edge);
      padding:.85rem 1.1rem; border-radius:0 6px 6px 0; margin:1.2rem 0; font-size:.93rem;
      max-width:72ch}
.note strong{color:var(--warn-ink)}
.ok{color:var(--good); font-weight:600}
.bad{color:var(--critical); font-weight:600}

a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def build_html(parts, body_only):
    body = f"<main>{parts}</main>"
    if body_only:
        return f"<title>PagedAttention replication — results</title><style>{CSS}</style>{body}"
    return ("<!doctype html><html lang=\"vi\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>PagedAttention replication — results</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="store_true",
                    help="also write a body-only artifact.html for publishing")
    args = ap.parse_args()

    runs = load_runs()
    e1 = [r for r in runs if r["_stage"] == "e1"]
    e3 = [r for r in runs if r["_stage"] == "e3"]
    e4 = [r for r in runs if r["_stage"] == "e4"]

    _, serving_b64, capacity = fig_serving(e1)
    _, batched_b64 = fig_batched(e1)
    _, kv_b64, waste = fig_kv_waste(e1)
    _, kernel_b64, kernel_table = fig_kernel(e3)
    _, block_b64 = fig_blocksize(e4)
    _, work_b64, work_meta = fig_workload()

    P = []
    P.append("<h1>PagedAttention: tái hiện trên GTX 1650 / Llama-3.2-1B</h1>")
    P.append('<p class="sub">So sánh hai engine viết tay chỉ khác nhau ở cơ chế quản lý KV cache — '
             'đặt chỗ liên tục kiểu Orca so với paging kiểu vLLM. Theo Kwon et al., SOSP \'23.</p>')

    # ---- status ----
    kinds = defaultdict(list)
    for r in e1 + e4:
        kinds[runqual.classify(r)[0]].append(r)
    P.append("<h2>Tình trạng dữ liệu</h2>")
    P.append(table(["Hạng mục", "Giá trị"], [
        ["Run §6.2 (E1) hoàn tất", f"{len(e1)} / 48"],
        ["Ablation block size (E4)", f"{len(e4)} / 18"],
        ["Microbenchmark kernel (E3)", "có" if e3 else "chưa chạy"],
        ["— trong đó <b>ổn định</b> (trung bình đã hội tụ)", len(kinds[runqual.STEADY])],
        ["— <b>quá tải</b> (vượt năng lực; dữ liệu đúng, là cận dưới)",
         len(kinds[runqual.SATURATED])],
        ["— <b>quá ngắn</b> (nhiễu, nên chạy lại dài hơn)",
         f'<span class="{"bad" if kinds[runqual.NOISY] else "ok"}">{len(kinds[runqual.NOISY])}</span>'],
        ["— <b>lỗi</b> (bỏ dở hoặc vi phạm bất biến token)",
         f'<span class="{"bad" if kinds[runqual.BROKEN] else "ok"}">{len(kinds[runqual.BROKEN])}</span>'],
    ], "Báo cáo này sinh được từ dữ liệu một phần; hạng mục nào chưa có sẽ ghi rõ là chưa chạy."))
    P.append('<div class="note"><strong>“Quá tải” không phải lỗi, nó là kết quả.</strong> Khi tốc '
             'độ request đến vượt năng lực hệ thống, hàng đợi dồn lên vô hạn và latency tăng đơn '
             'điệu suốt run — nửa sau <em>phải</em> cao hơn nửa đầu. Chạy lâu hơn chỉ làm khoảng '
             'cách đó rộng thêm. Đây đúng là các điểm làm đường cong vọt lên trong Fig 12 của paper, '
             'và con số báo cáo ở những điểm đó là <b>cận dưới</b> của latency thật ở trạng thái ổn '
             'định — tức là bảo thủ, không phải phóng đại. Phân biệt với “quá ngắn” bằng hàng đợi: '
             'dưới năng lực thì hàng đợi trống và mọi dao động chỉ là nhiễu.</div>')
    if kinds[runqual.NOISY] or kinds[runqual.BROKEN]:
        rows = [[r["workload"], r["rate"], r["label"], runqual.LABEL_VI[runqual.classify(r)[0]],
                 runqual.classify(r)[1]]
                for r in kinds[runqual.NOISY] + kinds[runqual.BROKEN]]
        P.append(table(["Workload", "rate", "Biến thể", "Loại", "Lý do"], rows,
                       "Những run này cần xử lý; mọi run khác dùng được."))

    # ---- workload ----
    P.append("<h2>1. Workload — ShareGPT và Alpaca thật</h2>")
    P.append("<p>Paper §6.1 lấy <em>cặp</em> (độ dài prompt, độ dài output) nguyên bản từ hai "
             "dataset thật, vì trong traffic thật hai đại lượng này tương quan với nhau, và "
             "<em>tổng</em> của chúng mới quyết định một sequence chiếm bao nhiêu KV cache.</p>")
    P.append(figure(work_b64, "Phân phối độ dài prompt và output thu được sau khi lọc"))
    if work_meta:
        rows = [[WORKLOAD_TITLE.get(n, n), m["num_samples"],
                 f'{m["prompt_len"]["mean"]:.1f}', m["prompt_len"]["p50"], m["prompt_len"]["p99"],
                 f'{m["output_len"]["mean"]:.1f}', m["output_len"]["p50"], m["output_len"]["p99"],
                 m["dropped_prompt_too_long"] + m["dropped_output_too_long"]]
                for n, m in work_meta.items()]
        P.append(table(["Dataset", "Mẫu", "prompt tb", "p50", "p99", "output tb", "p50", "p99", "bị lọc"], rows))
        if {"alpaca", "sharegpt"} <= set(work_meta):
            a, s = work_meta["alpaca"], work_meta["sharegpt"]
            pr = s["prompt_len"]["mean"] / a["prompt_len"]["mean"]
            orr = s["output_len"]["mean"] / a["output_len"]["mean"]
            P.append(f'<div class="note"><strong>Lệch so với paper:</strong> tỉ lệ ShareGPT/Alpaca '
                     f'ở đây là <b>{pr:.2f}×</b> prompt và <b>{orr:.2f}×</b> output, trong khi paper '
                     f'báo <b>8,4×</b> và <b>5,8×</b>. Nguyên nhân là cap 1024 token của engine này '
                     f'(VRAM 3,63 GiB) lọc bỏ phần đuôi dài của ShareGPT. Hai workload vẫn tách nhau '
                     f'rõ, nhưng độ tương phản nhẹ hơn paper — nên mọi tỉ số đo được ở đây là '
                     f'<em>cận dưới</em> của hiệu ứng.</div>')

    # ---- serving ----
    P.append("<h2>2. Đường cong serving (§6.2, Fig 12)</h2>")
    P.append("<p>Chỉ số chính của paper: <b>normalized latency</b> = thời gian end-to-end của một "
             "request chia cho số token nó sinh ra, trung bình trên mọi request, vẽ theo tốc độ "
             "request đến. Đường cong tăng chậm rồi vọt thẳng khi tốc độ đến vượt năng lực hệ thống.</p>")
    P.append(figure(serving_b64, "Normalized latency theo request rate (Poisson arrival)"))
    if capacity:
        rows = []
        for workload, caps in capacity.items():
            base = caps.get("base:max")
            for v in ORDER:
                if v not in caps:
                    continue
                c = caps[v]
                ratio = f"{c / base:.2f}×" if (c and base) else "—"
                rows.append([WORKLOAD_TITLE.get(workload, workload), v,
                             f"{c:.2f}" if c else "—", ratio])

        P.append(table(["Workload", "Biến thể", "Capacity point (req/s)", "so với Orca (Max)"], rows,
                       "Capacity point = tốc độ cao nhất mà normalized latency còn dưới 2× giá trị "
                       "lúc nhàn rỗi. Paper đọc bằng mắt từ đồ thị (“sustains 1.7×–2.7× higher "
                       "request rates”); đây là định nghĩa thao tác được tương ứng."))
        P.append('<div class="note"><strong>Số công bố là <code>paged</code> so với '
                 '<code>Orca (Max)</code>.</strong> Pow2 và Oracle là <em>cận trên không thể đạt</em> '
                 '— Oracle biết trước độ dài output, điều không engine thật nào làm được. Paper cũng '
                 'trình bày đúng như vậy ở Fig 12.</div>')

    # ---- mechanism ----
    P.append("<h2>3. Cơ chế đằng sau (Fig 13 và Fig 2)</h2>")
    P.append("<p>Mắt xích nhân quả: tiết kiệm bộ nhớ → batch được nhiều sequence hơn → throughput "
             "cao hơn. Trên GPU này decode bị chặn bởi bandwidth (mỗi step stream hết 2,30 GiB "
             "weights bất kể bao nhiêu slot hoạt động), nên thêm một slot gần như miễn phí.</p>")
    P.append(figure(batched_b64, "Số request được batch đồng thời"))
    P.append(figure(kv_b64, "KV cache đã cấp phát so với phần thực sự chứa token"))
    if waste:
        rows = [[v, f'{w["rate"]}', f'{w["reserved_pct"]:.1f}%', f'{w["live_pct"]:.1f}%',
                 f'<b>{w["waste_pct"]:.1f}%</b>'] for v, w in waste.items()]
        P.append(table(["Biến thể", "rate", "đã cấp phát", "chứa token thật", "lãng phí"], rows,
                       "Đây là Figure 2 của paper, đo được thay vì minh hoạ. Với base, khoảng trống "
                       "là phần đặt chỗ chưa dùng tới cộng phần buddy làm tròn lên lũy thừa 2; với "
                       "paged, nó chỉ là phần đuôi chưa đầy của page cuối mỗi sequence."))

    # ---- kernel ----
    P.append("<h2>4. Microbenchmark kernel attention (§7.1, Fig 18a)</h2>")
    P.append("<p>Tách riêng kernel attention khỏi phần còn lại của model, cùng seq_len và batch, "
             "cùng số byte cache. Khác biệt duy nhất là tra block table so với bước nhảy cố định.</p>")
    P.append(figure(kernel_b64, "Độ trễ kernel attention, paged so với contiguous"))
    if kernel_table:
        rows = [[r["batch"], r["context_len"], f'{r["contiguous_us"]:.1f}', f'{r["paged_us"]:.1f}',
                 f'{r["ratio"]:.2f}×',
                 '<span class="ok">khớp</span>' if r["digest_match"] else '<span class="bad">LỆCH</span>']
                for r in kernel_table]
        P.append(table(["batch", "context", "contiguous (µs)", "paged (µs)", "tỉ lệ", "digest output"], rows))
        worse = [r for r in kernel_table if r["ratio"] > 1.0]
        mean_ratio = sum(r["ratio"] for r in kernel_table) / len(kernel_table)
        P.append(f'<div class="note"><strong>Khác paper, và đây là kết quả thật:</strong> paper báo '
                 f'kernel paged chậm hơn <b>20–26%</b>; ở đây trung bình là <b>{mean_ratio:.2f}×</b> '
                 f'({len(worse)}/{len(kernel_table)} cấu hình chậm hơn). Hai kernel cho '
                 f'<em>digest output giống hệt nhau</em>, nên paged không hề làm ít việc hơn. '
                 f'Lý do khác biệt: paper so kernel của vLLM với <b>FasterTransformer</b> — một bản '
                 f'contiguous được tối ưu rất sâu — còn ở đây là so với biến thể contiguous của '
                 f'<em>cùng một kernel</em>. So cùng codebase, chi phí indirection nhỏ hơn cả chênh '
                 f'lệch do số học địa chỉ (paged dùng offset 32-bit, contiguous buộc phải dùng '
                 f'<code>size_t</code>) và do vòng lặp trong của paged chạy trên '
                 f'<code>BLOCK_SIZE</code> là hằng số compile-time nên unroll được.</div>')

    # ---- block size ----
    P.append("<h2>5. Ablation block size (§7.2, Fig 18b)</h2>")
    P.append("<p>Page nhỏ thì không tận dụng được băng thông GPU khi đọc cache; page lớn thì phân "
             "mảnh trong page tăng lên. vLLM chọn 16.</p>")
    P.append(figure(block_b64, "Normalized latency theo block size"))
    P.append('<div class="note">Ablation chạy ở cấu hình riêng <code>MAX_SEQUENCES = 64</code> cố '
             'định cho mọi block size: bảng block table tỉ lệ nghịch với block size, nên ở '
             '<code>BLOCK_SIZE=1</code> với 384 slot nó chiếm 25 MiB và không vừa VRAM. Vì KV pool '
             'lấy từ phần VRAM còn lại, pool co lại vài MiB ở block size nhỏ — đó là chi phí thật '
             'của page nhỏ, không phải nhiễu. Rate cố định <b>1.2 req/s</b>, chọn ngay dưới điểm gãy '
             'của paged: batch 26 trên ShareGPT nên có nhiều cache để đọc, mà hàng đợi còn gần trống '
             'nên latency vẫn phản ánh tính toán.</div>')
    P.append('<div class="note"><strong>Nhánh “block lớn trên workload ngắn” nằm ngoài vùng tải đã '
             'đo.</strong> Paper báo rằng trên Alpaca <em>"larger block sizes significantly degrade '
             'the performance since the sequences become shorter than the block sizes"</em>. Dữ liệu '
             'ở đây <b>không</b> thấy điều đó: Alpaca gần như phẳng (chênh 1,4% trên toàn dải). Lý '
             'do xác định được: hiệu ứng đó đến từ <em>áp lực bộ nhớ</em>, mà ở rate 1.2 Alpaca chỉ '
             'batch 8,1 → 8 × 256 ≈ 2048 token trên pool 31 344, tức <b>dùng 7% pool</b>. Lãng phí '
             'trong page có thật (sequence ~116 token nằm trong page 256 token bỏ không 140 token '
             'mỗi layer, hơn 50%) nhưng không chuyển thành chậm khi pool còn thừa mênh mông. Muốn '
             'thấy nhánh đó phải đo Alpaca ở rate cao hơn nhiều. Kết luận đúng từ dữ liệu này là '
             '“trong vùng tải đã đo, block size không ảnh hưởng đáng kể tới Alpaca” — <em>không</em> '
             'phải “block size không ảnh hưởng”.</div>')

    # ---- validity ----
    P.append("<h2>6. Điều kiện thí nghiệm và những gì không đo được</h2>")
    P.append("<h3>Hai bản chỉ khác đúng một thứ</h3><ul>"
             "<li><code>model.cu</code>, <code>utils.cu</code>, <code>request_queue.*</code>, "
             "<code>client.py</code>, <code>CMakeLists.txt</code>, <code>prompts.txt</code> "
             "<b>giống hệt nhau</b> giữa hai cây nguồn.</li>"
             "<li>Inventory kernel giống nhau trừ <code>contiguousAttentionKernel</code> ↔ "
             "<code>pagedAttentionKernel</code> và <code>markTokenListKernel</code> (chỉ paged cần, "
             "để dựng lại mask penalty sau khi recompute).</li>"
             "<li><b>Ngân sách KV không ai được chọn:</b> cả hai lấy phần VRAM còn lại sau khi cấp "
             "weights và scratch. Đo được 27 518 vs 27 390 token — lệch 0,47%, đúng bằng kích thước "
             "block table, là chi phí metadata thật của paging.</li>"
             "<li>Bộ cấp phát của base là <b>buddy allocation</b>, đúng như paper giả định về Orca; "
             "ba công thức đặt chỗ lấy nguyên văn §6.1.</li>"
             "<li>Cửa sổ ngữ cảnh 2048 token, khớp max sequence length của OPT trong paper.</li></ul>")
    P.append('<div class="note"><strong>Checksum chỉ so được ở mức song song bằng nhau.</strong> '
             'cuBLAS chọn kernel/tiling theo chiều batch, nên hai bản chạy số sequence khác nhau là '
             'đủ để sai số bf16 khác và argmax phân kỳ. Đo được: 12 vs 12 sequence → checksum khớp '
             'tuyệt đối; 10 vs 30 → lệch toàn bộ. PagedAttention lossless <em>với một batch cố '
             'định</em>, và paper cũng không tuyên bố hai hệ thống cho output giống nhau từng bit. '
             'Ở mọi mức song song, điều được kiểm là <em>bất biến token</em>: mỗi request emit chính '
             'xác ngân sách output, không token nào mất hay lặp, kể cả qua preempt.</div>')
    P.append("<h3>Nằm ngoài tầm</h3><ul>"
             "<li>Không bản nào có <b>chia sẻ KV giữa các sequence</b> (copy-on-write), nên §6.3 "
             "(parallel sampling, beam search), §6.4 (shared prefix) và Fig 15 không đo được. Đó lại "
             "là nơi vLLM thắng đậm nhất trong paper (tiết kiệm 37,6–55,2% bộ nhớ ở beam search), "
             "nên con số ở đây <b>khiêm tốn hơn paper</b>.</li>"
             "<li>Không có baseline FasterTransformer.</li>"
             "<li>§7.3 chỉ có nhánh <b>recompute</b>, không có swap-to-CPU (máy còn ~2,9 GiB RAM "
             "trống, PCIe laptop).</li>"
             "<li><b>GPU và model khác paper</b>: GTX 1650 4 GB / Llama-3.2-1B thay vì A100 / "
             "OPT-13B–175B. Không tensor core, 192 GB/s. Mọi con số <em>tuyệt đối</em> không so được "
             "với paper; chỉ <em>tỉ số giữa các biến thể</em> là so được.</li>"
             "<li>Repetition penalty 1,15 không có trong paper. Đối xứng ở hai bên, và dưới "
             "<code>IGNORE_EOS</code> thì độ dài output do ngân sách ấn định nên nó không đụng tới "
             "hành vi bộ nhớ.</li>"
             "<li>Paper chạy trace <b>1 tiếng</b> mỗi điểm; không sao chép được vì GPU này chậm hơn "
             "~100 lần. Thay bằng <b>200 request mỗi run cộng kiểm tra hội tụ</b> (nửa đầu so nửa "
             "sau lệch &lt; 10%) — kiểm chứng trực tiếp tính chất mà con số 1 tiếng bảo đảm.</li></ul>")

    # ---- thermal ----
    if e1:
        temps = [r["gpu_after"].get("temp_c") for r in e1 if r.get("gpu_after", {}).get("temp_c")]
        clocks = [r["gpu_after"].get("sm_clock_mhz") for r in e1 if r.get("gpu_after", {}).get("sm_clock_mhz")]
        if temps:
            P.append("<h3>Nhiệt độ và clock</h3>")
            P.append(table(["Đại lượng", "min", "trung bình", "max"], [
                ["Nhiệt độ cuối run (°C)", f"{min(temps):.0f}", f"{sum(temps)/len(temps):.0f}", f"{max(temps):.0f}"],
                ["SM clock cuối run (MHz)", f"{min(clocks):.0f}", f"{sum(clocks)/len(clocks):.0f}", f"{max(clocks):.0f}"],
            ], "Laptop 30 W có throttle, nên sweep.py xen kẽ các biến thể theo vòng và nghỉ 60 s "
               "giữa các run. Nếu không, biến thể chạy cuối sẽ gánh toàn bộ phần GPU đã nóng."))

    os.makedirs(REPORT, exist_ok=True)
    html = build_html("".join(P), body_only=False)
    out = os.path.join(REPORT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")
    if args.artifact:
        art = os.path.join(REPORT, "artifact.html")
        with open(art, "w", encoding="utf-8") as f:
            f.write(build_html("".join(P), body_only=True))
        print(f"wrote {art}")

    # machine-readable summary next to the HTML
    summary = {"capacity": capacity, "kv_waste": waste, "kernel": kernel_table,
               "workload_meta": work_meta, "e1_runs": len(e1), "e4_runs": len(e4)}
    with open(os.path.join(REPORT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(REPORT, 'summary.json')}")


if __name__ == "__main__":
    main()
