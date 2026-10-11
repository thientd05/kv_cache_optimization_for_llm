"""Write the finished experiments up as a paper, in the structure and register of Kwon et al.,
SOSP '23, rather than as the working report.

report.py is the lab notebook: every run, every caveat, every partial stage. This is the
write-up of the experiments that are complete (§6.2 basic sampling, §6.5 chatbot, §7.1-§7.3
ablations), ordered the way the original paper orders them. Every number in the prose is
computed from results/runs/ through report.py's derive(), so the two documents cannot disagree.

    python paper.py            -> results/paper/paper.html and results/paper/paper.pdf
    python paper.py --no-pdf   -> HTML only

Stages that never finished (parallel sampling, beam search, shared prefix) are named as future
work and nothing more: a paper reports what was measured.
"""

import argparse
import html
import io
import os
import re
import shutil
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import report as R
import runqual

OUT = os.path.join(R.REPO_ROOT, "results", "paper")

# Paper look: serif, small, vector. Text is converted to paths so the SVG renders identically
# wherever it is opened, and the PDF stays sharp at any zoom.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.6,
    "svg.fonttype": "path",
})
COL_W = 3.35    # inches, one column of a two-column page

LABEL = {"paged": "vLLM (paged)", "base:max": "Orca (Max)",
         "base:pow2": "Orca (Pow2)", "base:oracle": "Orca (Oracle)"}
STYLE = {
    "paged":       dict(color="#1f5fa8", marker="o", ls="-"),
    "base:max":    dict(color="#b8312f", marker="s", ls="-"),
    "base:pow2":   dict(color="#d4880f", marker="^", ls="--"),
    "base:oracle": dict(color="#4f8a2b", marker="v", ls=":"),
}
WTITLE = {"alpaca": "Alpaca", "sharegpt": "ShareGPT", "sharegpt_chat": "ShareGPT hội thoại"}

v = R.vi          # Vietnamese number formatting: 0,94 and 27 518
mean = R.mean


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def style_ax(ax):
    ax.grid(alpha=0.3, ls=":", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ============================ derived quantities ============================

def ceiling(rows, workload, variant):
    """Highest completion rate the variant reached anywhere on the grid, i.e. the request rate
    it can sustain. Past saturation completions stop tracking arrivals and flatten at this
    value. If the maximum sits at the top of the grid and was still climbing (> 5% over the
    previous point), the grid did not reach saturation and the value is a lower bound."""
    rs = sorted(R.pick(rows, workload=workload, variant=variant), key=lambda r: r["rate"])
    rs = [r for r in rs if r["throughput"]]
    if not rs:
        return None, False
    best = max(rs, key=lambda r: r["throughput"])
    lower = (best is rs[-1] and len(rs) > 1
             and best["throughput"] > 1.05 * rs[-2]["throughput"])
    return best["throughput"], lower


def kv_usage(e1_raw):
    """Paper Fig. 2, measured: of the KV cache a system has handed out, the share holding a
    real token, averaged over the busiest ShareGPT run of each variant."""
    out = {}
    for variant in R.ORDER:
        rs = [r for r in e1_raw if r["workload"] == "sharegpt" and r["label"] == variant]
        if not rs:
            continue
        r = max(rs, key=lambda r: r["rate"])
        steps = [s for s in r["steps"] if s.get("running")]
        res = mean([s["kv_tokens_reserved"] for s in steps])
        live = mean([s["kv_tokens_live"] for s in steps])
        total = steps[0]["kv_tokens_total"]
        out[variant] = {"used": 100.0 * live / res, "reserved_pct": 100.0 * res / total,
                        "live_pct": 100.0 * live / total, "rate": r["rate"]}
    return out


# ============================ figures ============================

def fig_usage(usage):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    names = [n for n in R.ORDER[1:] + ["paged"] if n in usage]
    xs = range(len(names))
    used = [usage[n]["used"] for n in names]
    ax.bar(xs, used, 0.6, color="#3c6e9f", label="chứa token thật")
    ax.bar(xs, [100 - u for u in used], 0.6, bottom=used, color="#e7c9a9",
           hatch="////", edgecolor="#b07a45", lw=0, label="đã cấp nhưng trống")
    for x, u in zip(xs, used):
        ax.text(x, u / 2, f"{u:.1f}".replace(".", ","), ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[n] for n in names])
    ax.set_ylabel("% KV cache đã cấp phát")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2, frameon=False)
    style_ax(ax)
    return svg(fig)


def fig_workload():
    import json
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    for name, color in (("alpaca", "#d4880f"), ("sharegpt", "#1f5fa8")):
        p = os.path.join(R.REPO_ROOT, "workloads", f"{name}.json")
        with open(p, encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        for ax, key in zip(axes, ("prompt_len", "output_len")):
            ax.hist([s[key] for s in samples], bins=50, range=(0, 1024), histtype="step",
                    density=True, color=color, lw=1.0, label=WTITLE[name])
    axes[0].set_xlabel("số token đầu vào")
    axes[1].set_xlabel("số token đầu ra")
    axes[0].set_ylabel("mật độ")
    for ax in axes:
        ax.set_yticks([])
        style_ax(ax)
    axes[1].legend(frameon=False)
    fig.tight_layout(w_pad=0.6)
    return svg(fig)


def fig_serving(rows, ws=("alpaca", "sharegpt")):
    """Paper Fig. 12 (and Fig. 17 when called with the chatbot workload alone). Kept to one
    column: a figure spanning both columns splits the page into two column blocks and breaks
    the left-then-right reading order."""
    fig, axes = plt.subplots(1, len(ws), figsize=(COL_W, 1.9), squeeze=False)
    axes = axes[0]
    for i, (ax, w) in enumerate(zip(axes, ws)):
        for name in R.ORDER:
            rs = sorted(R.pick(rows, workload=w, variant=name), key=lambda r: r["rate"])
            ax.plot([r["rate"] for r in rs], [r["nl"] for r in rs], label=LABEL[name],
                    **STYLE[name])
        ax.set_title(f"({'abc'[i]}) {WTITLE[w]}" if len(ws) > 1 else WTITLE[w])
        ax.set_xlabel("Tốc độ request (req/s)")
        ax.set_ylim(bottom=0)
        style_ax(ax)
    axes[0].set_ylabel("Normalized latency\n(s/token)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.13), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    return svg(fig)


def fig_batched(rows):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=False)
    for i, (ax, w) in enumerate(zip(axes, ("sharegpt", "alpaca"))):
        for name in R.ORDER:
            rs = sorted(R.pick(rows, workload=w, variant=name), key=lambda r: r["rate"])
            ax.plot([r["rate"] for r in rs], [r["running_mean"] for r in rs],
                    label=LABEL[name], **STYLE[name])
        ax.set_title(f"({'ab'[i]}) {WTITLE[w]}")
        ax.set_xlabel("Tốc độ request (req/s)")
        style_ax(ax)
    axes[0].set_ylabel("Số request trong batch (tb)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 1.08), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    return svg(fig)


def fig_ttft(rows):
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    for name in R.ORDER:
        rs = sorted(R.pick(rows, workload="sharegpt", variant=name), key=lambda r: r["rate"])
        ax.plot([r["rate"] for r in rs], [r["ttft_mean"] for r in rs], label=LABEL[name],
                **STYLE[name])
    ax.set_yscale("log")
    ax.set_xlabel("Tốc độ request (req/s)")
    ax.set_ylabel("TTFT trung bình (s)")
    ax.legend(frameon=False, fontsize=6)
    style_ax(ax)
    return svg(fig)


def fig_kernel(kt):
    batches = sorted({r["batch"] for r in kt})
    ctxs = sorted({r["context_len"] for r in kt})
    fig, axes = plt.subplots(1, len(batches), figsize=(COL_W, 1.6))
    w = 0.38
    for i, (ax, b) in enumerate(zip(axes, batches)):
        by = {r["context_len"]: r for r in kt if r["batch"] == b}
        xs = range(len(ctxs))
        ax.bar([x - w / 2 for x in xs], [by[c]["contiguous_us"] for c in ctxs], w,
               color="#b8312f", label="liền mạch")
        ax.bar([x + w / 2 for x in xs], [by[c]["paged_us"] for c in ctxs], w,
               color="#1f5fa8", label="phân trang")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(ctxs)
        ax.set_xlabel("Độ dài ngữ cảnh")
        ax.set_title(f"({'ab'[i]}) batch = {b}")
        style_ax(ax)
    axes[0].set_ylabel("Độ trễ kernel (µs)")
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=0.6)
    return svg(fig)


def fig_blocksize(e4):
    series = R.blocksize_series(e4)
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    marks = {("sharegpt", 1.2): ("#1f5fa8", "o"), ("alpaca", 1.2): ("#d4880f", "^"),
             ("alpaca", 2.5): ("#8a5a10", "v")}
    for key in sorted(series, key=lambda k: (k[0] != "sharegpt", k[1])):
        by = series[key]
        bs = sorted(by)
        color, m = marks.get(key, ("#555", "x"))
        ax.plot(bs, [by[b] for b in bs], color=color, marker=m,
                label=f"{WTITLE[key[0]]} @ {R.rate_str(key[1])} req/s")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({b for by in series.values() for b in by}))
    ax.set_xticklabels([str(b) for b in sorted({b for by in series.values() for b in by})])
    ax.set_xlabel("Block size (token)")
    ax.set_ylabel("Normalized latency\n(s/token)")
    ax.legend(frameon=False, fontsize=6)
    style_ax(ax)
    return svg(fig)


def fig_recovery(rec):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75))
    col = {"recompute": "#1f5fa8", "swap": "#b8312f"}
    name = {"recompute": "Tính lại", "swap": "Swap"}
    sizes = sorted({d["block_size"] for d in rec})
    for policy in ("recompute", "swap"):
        sub = sorted([d for d in rec if d["policy"] == policy], key=lambda d: d["block_size"])
        axes[0].plot([d["block_size"] for d in sub], [d["cost_ms"] for d in sub],
                     color=col[policy], marker="o", label=name[policy])
        axes[1].plot([d["block_size"] for d in sub], [d["nl"] for d in sub],
                     color=col[policy], marker="o", label=name[policy])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("ms mỗi lần phục hồi")
    axes[1].set_ylabel("Normalized latency (s/token)")
    axes[1].set_ylim(bottom=0)
    for i, ax in enumerate(axes):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes[::2])
        ax.set_xticklabels([str(s) for s in sizes[::2]])
        ax.set_xlabel("Block size")
        ax.set_title(f"({'ab'[i]}) " + ("Microbenchmark" if i == 0 else "End-to-end"))
        style_ax(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=0.8)
    return svg(fig)


def fig_mechanisms():
    """Schematic, drawn by hand for one column: the three kinds of waste in a contiguous pool,
    and two requests in a paged pool. Not data."""
    A, Al, B, Bl = "#1f5fa8", "#b9cfe8", "#d4880f", "#f3d9ae"
    out = ['<svg viewBox="0 0 340 290" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Liberation Serif, DejaVu Serif, serif">',
           '<defs><pattern id="h" width="5" height="5" patternUnits="userSpaceOnUse" '
           'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#888" '
           'stroke-width="1.1"/></pattern></defs>']

    def box(x0, x1, y, fill, h=24, hatch=False):
        out.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" fill="{fill}" '
                   f'stroke="#333" stroke-width="0.7"/>')
        if hatch:
            out.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" fill="url(#h)"/>')

    def text(x, y, s, anchor="middle", size=9, weight="normal", fill="#222"):
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                   f'font-weight="{weight}" fill="{fill}">{html.escape(s)}</text>')

    # (a) contiguous
    text(4, 13, "(a) Đặt chỗ liên tục (Orca)", anchor="start", size=11.5, weight="bold")
    y = 22
    box(5, 65, y, A); box(65, 125, y, Al); box(125, 155, y, Al, hatch=True); box(155, 185, y, "#fff")
    box(185, 225, y, B); box(225, 270, y, Bl); box(270, 295, y, Bl, hatch=True); box(295, 335, y, "#fff")
    text(35, y + 16, "A", fill="#fff", weight="bold"); text(205, y + 16, "B", fill="#fff", weight="bold")
    for x0, x1, n in ((65, 125, "1"), (125, 155, "2"), (155, 185, "3"), (225, 270, "1"),
                      (270, 295, "2"), (295, 335, "4")):
        out.append(f'<path d="M{x0 + 2} {y + 28} v4 H{x1 - 2} v-4" fill="none" stroke="#555" '
                   f'stroke-width="0.7"/>')
        text((x0 + x1) / 2, y + 45, n, size=10.5, weight="bold")
    legend = [("Đậm", "prompt và token đã sinh của request A, B"),
              ("1", "đặt chỗ cho token chưa sinh"),
              ("2", "làm tròn lên luỹ thừa 2 (buddy)"),
              ("3", "lỗ trống quá nhỏ: phân mảnh ngoài"),
              ("4", "trống")]
    for i, (k, d) in enumerate(legend):
        yy = y + 64 + i * 15
        text(8, yy, k, anchor="start", size=10, weight="bold")
        text(40, yy, d, anchor="start", size=10, fill="#333")

    # (b) paged
    y = 174
    text(4, y, "(b) Phân trang (PagedAttention), block = 4 token", anchor="start", size=11.5,
         weight="bold")
    layout = ["A", "B", "", "A", "B", "", "A2", "", "B2", "",
              "", "", "", "", "", "", "", "", "", ""]
    for i, who in enumerate(layout):
        x = 5 + (i % 10) * 33
        yy = y + 8 + (i // 10) * 30
        fill = {"A": A, "B": B, "A2": A, "B2": B}.get(who, "#fff")
        if who in ("A2", "B2"):
            box(x, x + 15, yy, fill); box(x + 15, x + 30, yy, Al if who == "A2" else Bl)
        else:
            box(x, x + 30, yy, fill)
    text(4, y + 80, "Mỗi request chỉ giữ các block đã dùng tới; lãng phí duy", anchor="start",
         size=10, fill="#333")
    text(4, y + 94, "nhất là phần đuôi của block cuối. Block table ánh xạ block", anchor="start",
         size=10, fill="#333")
    text(4, y + 108, "logic → block vật lý, nên các block không cần nằm liền nhau.",
         anchor="start", size=10, fill="#333")
    out.append("</svg>")
    return "".join(out)


# ============================ document ============================

CSS = """
@page { size: A4; margin: 18mm 15mm 18mm 15mm; }
:root { --ink:#111; --muted:#555; --rule:#999; }
html { background:#fff; }
body { margin:0; color:var(--ink); background:#fff;
  font-family:"Liberation Serif","Times New Roman","DejaVu Serif",serif;
  font-size:10.5pt; line-height:1.38; text-rendering:optimizeLegibility; }
main { max-width:780px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:17pt; text-align:center; margin:0 0 8px; line-height:1.25; font-weight:bold; }
.authors { text-align:center; margin:0 0 4px; font-size:11pt; }
.affil { text-align:center; color:var(--muted); font-size:9.5pt; margin:0 0 18px; }
.abstract { margin:0 auto 18px; max-width:640px; font-size:9.8pt; }
.abstract h2 { text-align:center; font-size:10.5pt; margin:0 0 4px; }
.abstract p { text-align:justify; margin:0; }
.cols p { text-align:justify; hyphens:auto; margin:0 0 6px; text-indent:1.2em; }
.cols h2 + p, .cols h3 + p, .cols figure + p, .cols table + p, .cols .tbl + p, .cols ul + p { text-indent:0; }
h2 { font-size:11.5pt; margin:14px 0 6px; break-after:avoid; }
h3 { font-size:10.5pt; margin:10px 0 4px; break-after:avoid; }
h3.run { display:inline; font-size:inherit; margin:0; }
ul { margin:4px 0 8px; padding-left:1.4em; } li { margin:0 0 3px; text-align:justify; }
figure { margin:10px 0 12px; break-inside:avoid; }
figure svg { width:100%; height:auto; display:block; }
figcaption { font-size:9pt; line-height:1.3; margin-top:4px; text-align:justify; }
figcaption b { font-weight:bold; }
.tbl { break-inside:avoid; margin:10px 0 12px; }
.tbl .cap { font-size:9pt; margin-bottom:4px; text-align:justify; }
table { border-collapse:collapse; width:100%; font-size:8.6pt; line-height:1.25; }
th, td { padding:2px 5px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead tr:first-child th { border-top:1.2px solid var(--ink); }
thead tr:last-child th { border-bottom:0.7px solid var(--ink); }
tbody tr:last-child td { border-bottom:1.2px solid var(--ink); }
tr.grp td { border-top:0.5px solid var(--rule); }
.wide { column-span:all; }
a.cite { color:#1f4f8a; text-decoration:none; }
a.cite:hover { text-decoration:underline; }
.refs li:target { background:#fff3c4; }
.refs { font-size:8.8pt; padding-left:1.6em; } .cols .refs li { margin-bottom:3px; text-align:left; word-break:break-word; }
.note { font-size:8.6pt; color:var(--muted); }
@media print {
  main { max-width:none; padding:0; }
  .cols { column-count:2; column-gap:7mm; }
  figure.wide, .tbl.wide { margin:8px 0 12px; }
}
"""


def check_layout(pdf):
    """Print a warning for every tall blank run inside a column of the PDF, last page excepted.

    In CSS columns a figure that does not fit the rest of a column jumps to the next one and
    leaves the space above it empty - LaTeX would float it, Chrome cannot. The fix is to move
    that figure one paragraph later in main(); this check is what tells you which page to look at."""
    import glob
    import tempfile
    from PIL import Image
    if not shutil.which("pdftoppm"):
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "50", "-png", pdf, os.path.join(tmp, "p")], check=True)
        pages = sorted(glob.glob(os.path.join(tmp, "p-*.png")))
        bad = 0
        for n, f in enumerate(pages[:-1], 1):
            im = Image.open(f).convert("L")
            w, h = im.size
            px = im.load()
            for col, (x0, x1) in (("trái", (0.07, 0.49)), ("phải", (0.51, 0.93))):
                xs = range(int(w * x0), int(w * x1), 2)
                run = 0
                for y in range(int(h * 0.05), int(h * 0.94)):
                    run = run + 1 if all(px[x, y] > 245 for x in xs) else 0
                    if run == int(h * 0.08):
                        bad += 1
                        print(f"  layout: trang {n}, cột {col} có khoảng trắng lớn quanh "
                              f"{100 * (y - run) / h:.0f}% chiều cao trang")
        print("  layout: không có khoảng trắng lớn" if not bad else
              "  layout: dời hình tương ứng ra sau đoạn văn kế tiếp rồi chạy lại")


def link_citations(doc):
    """[1], [2] in the prose become links to the reference list. Figures are skipped: their SVG
    is matplotlib output and never contains a citation, but it can contain bracketed numbers."""
    parts = re.split(r"(<svg.*?</svg>)", doc, flags=re.S)
    cite = re.compile(r"\[(\d+)\]")
    return "".join(p if p.startswith("<svg") else
                   cite.sub(r'<a class="cite" href="#ref-\1">[\1]</a>', p) for p in parts)


def tbl(cap, head, body, wide=False, groups=()):
    h = "".join(f"<th>{c}</th>" for c in head)
    rows = []
    for i, r in enumerate(body):
        cls = ' class="grp"' if i in groups else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (f'<div class="tbl{" wide" if wide else ""}"><div class="cap">{cap}</div>'
            f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def fig(svgtext, cap, wide=False):
    return f'<figure{" class=\"wide\"" if wide else ""}>{svgtext}<figcaption>{cap}</figcaption></figure>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    runs = R.load_runs()
    raw = {s: [r for r in runs if r["_stage"] == s] for s in ("e1", "e3", "e4", "e5")}
    e1 = [R.derive(r) for r in raw["e1"]]
    e4 = [R.derive(r) for r in raw["e4"]]
    e5 = [R.derive(r) for r in raw["e5"]]
    used_runs = [r for r in e1 + e4 + e5 if r["quality"] != runqual.BROKEN]
    n_runs = len(used_runs) + len(raw["e3"])

    # ---------------- numbers quoted in the prose ----------------
    cfg = {r["label"]: r["engine_config"] for r in raw["e1"]}
    pool_b, pool_p = cfg["base:max"]["kv_pool_tokens"], cfg["paged"]["kv_pool_tokens"]
    usage = kv_usage(raw["e1"])
    ceil = {(w, n): ceiling(e1, w, n) for w in R.WORKLOADS for n in R.ORDER}

    def ceil_s(w, n):
        c, lower = ceil[(w, n)]
        return ("≥ " if lower else "") + v(c, ".2f")

    def ratio(w, a, b):
        (ca, la), (cb, lb) = ceil[(w, a)], ceil[(w, b)]
        if la and lb:
            return "—"   # both still climbing: the grid says nothing about their ratio
        return ("≥ " if la and not lb else "") + v(ca / cb, ".2f") + "×"

    def at(w, n, rate):
        return R.at_rate(e1, w, n, rate)

    sg_top = max(r["rate"] for r in R.pick(e1, workload="sharegpt"))
    run_p, run_m, run_o = (at("sharegpt", n, sg_top)["running_mean"]
                           for n in ("paged", "base:max", "base:oracle"))
    ref = R.reference_rate(e1, "sharegpt")
    tt_p, tt_m, tt_o = (at("sharegpt", n, ref)["ttft_mean"] for n in ("paged", "base:max", "base:oracle"))
    frag = {(w, n): max((r["frag"] for r in R.pick(e1, workload=w, variant=n)), default=0)
            for w in R.WORKLOADS for n in R.ORDER}
    pre = [r for r in R.pick(e1, variant="paged") if r["preempt"]]
    pre_max = max(pre, key=lambda r: r["preempt"] / (r["num_requests"] or 200))
    pre_share = max((100.0 * r["recompute_ms"] / 1000.0 / r["wall_s"]) for r in pre
                    if r["recompute_ms"] is not None and r["wall_s"])
    chat_rate = 0.3
    chat_p, chat_m = at("sharegpt_chat", "paged", chat_rate), at("sharegpt_chat", "base:max", chat_rate)
    usage_paged_top = at("sharegpt", "paged", sg_top)

    _, _, kt = R.fig_kernel(raw["e3"])
    k_ratios = [r["ratio"] for r in kt]
    pen = R.blocksize_series(e4)
    sg_bs = pen[("sharegpt", 1.2)]
    best16_128 = max(sg_bs[b] for b in (16, 32, 64, 128)) / min(sg_bs[b] for b in (16, 32, 64, 128)) - 1
    al_pen = max(by[b] / by[16] - 1 for (w, _), by in pen.items() if w == "alpaca" for b in by)
    rec = R.recovery_rows(e5)
    rc = [d for d in rec if d["policy"] == "recompute"]
    sw = [d for d in rec if d["policy"] == "swap"]
    rc_lo, rc_hi = min(d["cost_ms"] for d in rc), max(d["cost_ms"] for d in rc)
    sw_by = {d["block_size"]: d["cost_ms"] for d in sw}
    rc_by = {d["block_size"]: d["cost_ms"] for d in rc}
    rs_ratio = [rc_by[b] / sw_by[b] for b in sw_by if b in rc_by]
    events = [d["events"] for d in rec]
    nl_rec = [d["nl"] for d in rec]
    clocks = [r["gpu_after"].get("sm_clock_mhz") for r in raw["e1"]
              if (r.get("gpu_after") or {}).get("sm_clock_mhz")]
    import json as _json
    meta = {w: _json.load(open(os.path.join(R.REPO_ROOT, "workloads", f"{w}.json"),
                               encoding="utf-8"))["meta"] for w in R.WORKLOADS}
    pr_ratio = meta["sharegpt"]["prompt_len"]["mean"] / meta["alpaca"]["prompt_len"]["mean"]
    out_ratio = meta["sharegpt"]["output_len"]["mean"] / meta["alpaca"]["output_len"]["mean"]
    kv_mib = pool_b * 32 / 1024

    u = {n: v(usage[n]["used"], ".1f") for n in usage}

    # ---------------- figures ----------------
    F = {
        "usage": fig_usage(usage), "mech": fig_mechanisms(), "work": fig_workload(),
        "serve": fig_serving(e1), "chat": fig_serving(e1, ("sharegpt_chat",)), "batch": fig_batched(e1), "ttft": fig_ttft(e1),
        "kernel": fig_kernel(kt), "block": fig_blocksize(e4), "rec": fig_recovery(rec),
    }

    P = []
    P.append('<h1>Nghiên cứu thuật toán quản lý bộ nhớ PagedAttention với mô hình<br>'
             'Llama-3.2-1B-Instruct trên GPU <span style="white-space:nowrap">'
             'NVIDIA GeForce GTX 1650</span></h1>')
    P.append('<p class="authors">Tạ Đình Thiện — 20235839</p>')
    P.append('<p class="affil">Báo cáo tái hiện — Kwon et al., SOSP \'23</p>')

    # ---------------- abstract ----------------
    P.append(f'''<div class="abstract"><h2>Tóm tắt</h2><p>Kích thước batch của một hệ thống
phục vụ mô hình ngôn ngữ lớn (LLM) bị giới hạn bởi bộ nhớ dành cho KV cache, và do đó cách quản
lý vùng nhớ này quyết định throughput. Kwon và cộng sự đề xuất PagedAttention, quản lý KV cache
theo các block kích thước cố định được cấp phát theo nhu cầu như bộ nhớ ảo của hệ điều hành, và
báo cáo khả năng chịu tải cao hơn 2–4× so với các hệ thống đặt chỗ liên tục. Bài này kiểm chứng
lại kết luận đó trong một môi trường gần như đối lập: một GPU laptop GTX 1650 4 GB không có
tensor core và mô hình Llama-3.2-1B. Hai engine suy luận được viết từ đầu bằng CUDA, chỉ khác
nhau ở cơ chế cấp phát KV cache — một bản đặt chỗ liên tục kiểu Orca với buddy allocator và một
bản phân trang kiểu vLLM. Trên trace ShareGPT, bản phân trang duy trì được tốc độ request cao hơn
{ratio("sharegpt", "paged", "base:max")} so với Orca (Max) và
{ratio("sharegpt", "paged", "base:oracle")} so với Orca (Oracle); {u["paged"]}% KV cache đã cấp
phát của nó chứa token thật, so với {u["base:max"]}–{u["base:oracle"]}% ở các biến thể Orca.
Lợi thế nhỏ hơn bài gốc vì trên phần cứng này throughput của bản phân trang chạm trần tính toán
trước khi chạm trần bộ nhớ. Các thí nghiệm ablation cho thấy kernel attention phân trang không
chậm hơn bản liền mạch cùng cài đặt, block size 16 nằm trong vùng tối ưu, và — ngược với bài
gốc — swap KV cache sang bộ nhớ host rẻ hơn tính lại {v(min(rs_ratio), ".0f")}–{v(max(rs_ratio), ".0f")}×
ở mọi block size vì GPU yếu về sức tính.</p></div>''')

    C = []   # two-column body
    # ---------------- 1 introduction ----------------
    C.append("<h2>1 Giới thiệu</h2>")
    C.append(f'''<p>Phục vụ LLM là một bài toán bộ nhớ nhiều hơn là một bài toán tính toán. Ở
giai đoạn sinh token, mỗi bước decode phải đọc toàn bộ trọng số mô hình từ DRAM của GPU bất kể
đang xử lý bao nhiêu request, nên gộp thêm request vào cùng một batch gần như không tốn thêm thời
gian. Thứ giới hạn kích thước batch là KV cache — trạng thái key/value mà mỗi request phải giữ cho
mọi token trong ngữ cảnh của nó. Với Llama-3.2-1B, mỗi token chiếm 32 KiB KV cache; trên GPU
4 GB dùng trong bài này, sau khi trọng số chiếm 2,30 GiB, phần còn lại chỉ chứa được khoảng
{v(pool_b, ",d")} token ({v(kv_mib, ".0f")} MiB).</p>''')
    C.append(f'''<p>Các hệ thống phục vụ như Orca [2] và FasterTransformer [3] lưu KV cache của mỗi
request trong một vùng nhớ liền mạch, đặt trước cho độ dài tối đa có thể. Vì độ dài đầu ra không
biết trước, phần lớn vùng đặt trước bị bỏ trống trong suốt vòng đời request; bộ cấp phát còn làm
tròn kích thước và để lại các mảnh vụn giữa các vùng. Kwon et al. [1] đo được chỉ
20,4–38,2% KV cache trong các hệ thống này thực sự chứa token. Họ đề xuất PagedAttention: chia KV
cache thành các block nhỏ cố định, cấp phát theo nhu cầu và tra qua một block table, như phân trang
trong bộ nhớ ảo. Hình 1 cho thấy phép đo tương ứng trên hệ thống trong bài:
{u["base:max"]}%, {u["base:pow2"]}% và {u["base:oracle"]}% ở ba biến thể Orca, so với
{u["paged"]}% ở bản phân trang — rất sát con số 20,4/26,8/38,2/96,3% của bài gốc.</p>''')
    C.append(fig(F["usage"], f'''<b>Hình 1.</b> Tỉ lệ KV cache đã cấp phát thực sự chứa token,
trung bình theo thời gian trên trace ShareGPT ở {R.rate_str(usage["paged"]["rate"])} req/s.
Số liệu tự đo từ telemetry của hai bộ cấp phát; thí nghiệm tương ứng với Hình 2 của [1].'''))
    C.append('''<p>Các kết quả của [1] được đo trên GPU A100 với các mô hình OPT 13B–175B — nơi bộ
nhớ khan hiếm nhưng sức tính dồi dào. Bài này hỏi: lợi ích của phân trang còn lại bao nhiêu khi
cả hai đều khan hiếm? GTX 1650 không có tensor core, đạt khoảng 2,7 TFLOP/s ở bf16 và 192 GB/s
băng thông bộ nhớ — thấp hơn A100 khoảng hai bậc độ lớn về tính toán. Đóng góp của bài gồm:</p>''')
    C.append(f'''<ul>
<li>Hai engine suy luận viết tay bằng CUDA cho Llama-3.2-1B, <b>giống hệt nhau trừ cơ chế cấp
phát KV cache</b>; sự giống nhau được kiểm tra tự động ở mỗi lần build.</li>
<li>Tái hiện các thí nghiệm lấy mẫu cơ bản (§6.2), chatbot (§6.5) và ba ablation (§7.1–7.3) của
[1] trên {n_runs} lượt chạy.</li>
<li>Đo trực tiếp những đại lượng mà [1] chỉ minh hoạ: tỉ lệ bộ nhớ đã cấp chứa token theo thời
gian, và số lần cấp phát thất bại do phân mảnh ngoài.</li>
<li>Hai phát hiện khác bài gốc: lợi thế của phân trang bị chặn bởi trần tính toán của GPU, và swap
rẻ hơn tính lại ở mọi block size.</li></ul>''')

    # ---------------- 2 background ----------------
    C.append("<h2>2 Bối cảnh</h2>")
    C.append('''<h3>2.1 Sinh văn bản tự hồi quy và KV cache</h3><p>Một request gồm hai pha. Pha
<i>prefill</i> xử lý toàn bộ prompt trong một lượt và là phép nhân ma trận lớn, bị giới hạn bởi
sức tính. Pha <i>decode</i> sinh từng token một; mỗi bước cần key và value của mọi token trước
đó, nên chúng được lưu lại thành KV cache thay vì tính lại. Pha decode bị giới hạn bởi băng thông
bộ nhớ: với Llama-3.2-1B, mỗi bước phải đọc 2,30 GiB trọng số, tức tối thiểu khoảng 12 ms trên
192 GB/s, bất kể batch có bao nhiêu request. Vì vậy số request xử lý được đồng thời — và
throughput — tỉ lệ gần như trực tiếp với số request mà KV cache chứa được.</p>''')
    C.append('''<h3>2.2 Batching ở mức vòng lặp</h3><p>Orca [2] lập lịch theo từng bước decode thay
vì theo từng batch: request mới được đưa vào ngay khi có chỗ, và request xong được giải phóng ngay.
Cả hai engine trong bài đều dùng cơ chế này, nên khác biệt giữa chúng chỉ nằm ở việc <i>có chỗ</i>
hay không.</p>''')
    C.append('''<h3>2.3 Đặt chỗ liên tục và ba loại lãng phí</h3><p>Khi KV cache của một request
phải nằm liền mạch, hệ thống phải đặt chỗ cho toàn bộ độ dài tương lai ngay khi nhận request
(Hình 2a). Điều đó gây ra ba loại lãng phí [1]: (i) <i>đặt chỗ</i> — vùng dành cho token chưa
sinh; (ii) <i>phân mảnh trong</i> — phần làm tròn của bộ cấp phát và phần đặt dư so với độ dài
thật; (iii) <i>phân mảnh ngoài</i> — khoảng trống giữa các vùng, đủ về tổng nhưng không đủ liền
mạch cho request mới.</p>''')
    C.append('''<h3>2.4 PagedAttention</h3><p>PagedAttention [1] chia KV cache thành các block cố
định (mặc định 16 token). Mỗi request có một block table ánh xạ block logic sang block vật lý; block
mới chỉ được cấp khi block cuối đã đầy (Hình 2b). Lãng phí duy nhất là phần chưa đầy của block cuối,
và vì mọi block cùng kích thước nên không có phân mảnh ngoài. Đổi lại, hệ thống không còn bảo đảm
một request đã nhận sẽ có đủ chỗ đến cuối: khi pool cạn giữa chừng, nó phải <i>preempt</i> một
request và sau đó phục hồi bằng cách tính lại KV cache hoặc swap nó qua bộ nhớ host.</p>''')
    C.append(fig(F["mech"], '''<b>Hình 2.</b> Sơ đồ minh hoạ hai cơ chế. (a) Đặt chỗ liên tục: mỗi request giữ một
vùng cho toàn bộ độ dài tối đa. (b) Phân trang: request chỉ giữ các block đã dùng tới, nằm rải rác
trong pool.'''))

    # ---------------- 3 implementation ----------------
    C.append("<h2>3 Thiết kế và cài đặt</h2>")
    C.append(f'''<h3>3.1 Engine chung</h3><p>Hai engine phục vụ Llama-3.2-1B-Instruct [6] ở bf16 (16
layer, hidden 2048, 32 query head, 8 KV head theo GQA, head dim 64). Mọi kernel — RMSNorm, RoPE,
attention, SwiGLU, lấy mẫu — viết tay bằng CUDA; riêng các phép nhân ma trận dùng cuBLAS. Lập lịch
theo FCFS ở mức vòng lặp; mọi prompt đang chờ được prefill gộp trong một lượt tối đa 3072 token, và
decode chạy tối đa 384 request đồng thời. Cửa sổ ngữ cảnh là 2048 token (1024 cho prompt, 1024 cho
đầu ra), bằng độ dài tối đa của OPT trong [1]. Engine nhận request qua stdin và chỉ báo lại
số token, không gửi văn bản ra ngoài.</p>''')
    C.append('''<h3>3.2 Baseline kiểu Orca</h3><p>Mã nguồn của Orca không công khai, nên giống [1],
Orca được cài đặt lại với ba chính sách đặt chỗ: <b>Orca (Max)</b> đặt chỗ cho 2048 token;
<b>Orca (Pow2)</b> đặt prompt cộng độ dài đầu ra làm tròn lên luỹ thừa 2 kế tiếp; <b>Orca
(Oracle)</b> đặt đúng prompt cộng độ dài đầu ra thật. Pow2 và Oracle cần biết trước độ dài đầu
ra nên không cài đặt được trong thực tế; chúng là cận trên. Như [1] giả định, vùng nhớ được cấp bởi
buddy allocator, nên mọi đặt chỗ còn được làm tròn lên luỹ thừa 2. Một request chỉ được nhận khi
tìm được một vùng liền mạch đủ lớn; mỗi lần thất bại trong khi tổng bộ nhớ trống vẫn đủ được đếm
là một lần phân mảnh ngoài.</p>''')
    C.append('''<h3>3.3 Bản phân trang kiểu vLLM</h3><p>Pool được chia thành block 16 token. Một
request được nhận khi prompt của nó vừa pool; block cho phần sinh ra được cấp dần trong lúc decode.
Khi pool cạn, engine preempt request đến sau cùng, giải phóng toàn bộ block của nó và — theo §4.5
của [1] — ngừng nhận request mới cho đến khi mọi request bị preempt hoàn thành. Hai chính sách phục
hồi được cài đặt: <i>tính lại</i> (prefill lại prompt cùng các token đã sinh) và <i>swap</i> (chép
các block sang bộ nhớ host pinned rồi chép lại khi có chỗ).</p>''')
    C.append(f'''<h3>3.4 Bảo đảm so sánh công bằng</h3><p>Ba cơ chế giữ cho phép so sánh chỉ có
một biến số. Thứ nhất, các hằng số dùng chung của hai engine nằm trong một khối cấu hình phải giống
nhau từng byte, và script build từ chối build nếu hai khối lệch nhau. Thứ hai, không ai chọn kích
thước KV cache: mỗi engine cấp phát trọng số và bộ đệm tạm trước, rồi lấy toàn bộ VRAM còn trống
(trừ 128 MiB dự phòng) làm KV cache. Kết quả là {v(pool_b, ",d")} token cho bản liền mạch và
{v(pool_p, ",d")} token cho bản phân trang; chênh lệch {v(100 * (1 - pool_p / pool_b), ".2f")}%
chính là bộ nhớ của block table. Thứ ba, cả hai dùng cùng file workload, cùng seed và cùng chuỗi
thời điểm đến; EOS bị bỏ qua để mỗi request sinh đúng số token đầu ra ghi trong trace.</p>''')

    # ---------------- 4 evaluation ----------------
    C.append("<h2>4 Đánh giá</h2>")
    C.append('''<h3>4.1 Thiết lập thí nghiệm</h3><p><b>Phần cứng và mô hình.</b> NVIDIA GTX 1650
laptop (Turing, 14 SM, 3,63 GiB VRAM khả dụng, 192 GB/s, giới hạn công suất 30 W, không tensor
core), CUDA 12.0, Llama-3.2-1B-Instruct ở bf16.</p>''')
    wrows = []
    for w in R.WORKLOADS:
        m = meta[w]
        wrows.append([WTITLE[w], v(m["prompt_len"]["mean"], ".0f"), str(m["prompt_len"]["p50"]),
                      str(m["prompt_len"]["p99"]), v(m["output_len"]["mean"], ".0f"),
                      str(m["output_len"]["p50"]), str(m["output_len"]["p99"])])
    C.append(f'''<p><b>Workload.</b> Như [1], độ dài đầu vào và đầu ra của mỗi request lấy từ các
cặp hỏi–đáp thật của ShareGPT [4] và Alpaca [5], đã tokenize bằng tokenizer của Llama (Bảng 1,
Hình 3). Mẫu dài hơn cửa sổ ngữ cảnh bị loại thay vì cắt ngắn, để không tạo ra một cụm nhân tạo tại
độ dài tối đa. Vì cửa sổ ngắn hơn, ShareGPT ở đây chỉ dài hơn Alpaca {v(pr_ratio, ".1f")}× ở đầu
vào và {v(out_ratio, ".1f")}× ở đầu ra, so với 8,4× và 5,8× trong [1]. Workload chatbot (§4.5)
được dựng như §6.5 của [1]: mỗi lượt hội thoại gửi lại toàn bộ lịch sử, cắt về 1024 token cuối.
Thời điểm đến tuân theo phân phối Poisson.</p>''')
    C.append(tbl("<b>Bảng 1.</b> Độ dài (token) của ba workload sau khi lọc, 20 000 mẫu mỗi bộ.",
                 ["Workload", "Vào tb", "p50", "p99", "Ra tb", "p50", "p99"], wrows))
    C.append(f'''<p><b>Chỉ số.</b> Như [1], chỉ số chính là <i>normalized latency</i>: thời gian
end-to-end của mỗi request chia cho số token đầu ra, lấy trung bình trên mọi request. Khả năng chịu
tải của một hệ thống được tóm tắt bằng <i>throughput tối đa duy trì được</i>: số request hoàn thành
mỗi giây cao nhất đạt được trên lưới tốc độ đến; khi quá tải, tốc độ hoàn thành ngừng tăng theo tốc
độ đến và dừng ở đúng giá trị này. Ngoài ra còn đo <i>time to first token</i> (TTFT),
vốn không bị pha loãng bởi độ dài đầu ra.</p>''')
    C.append(fig(F["work"], "<b>Hình 3.</b> Phân phối độ dài đầu vào và đầu ra của ShareGPT và "
                            "Alpaca."))
    C.append(f'''<p><b>Quy trình.</b> Mỗi điểm đo là một lượt chạy 200 request trên lưới 8 tốc độ
đến cho mỗi workload. Thay cho trace 1 giờ của [1], sự hội tụ của mỗi lượt chạy được kiểm tra
bằng cách so normalized latency của nửa đầu với nửa sau. Các biến thể được chạy xen kẽ, nghỉ 60 s
giữa hai lượt, để GPU laptop bị giới hạn nhiệt không thiên vị biến thể chạy sau; xung nhịp SM cuối
mỗi lượt nằm trong {v(min(clocks), ",.0f")}–{v(max(clocks), ",.0f")} MHz.</p>''')

    # 4.2 basic sampling
    C.append("<h3>4.2 Lấy mẫu cơ bản</h3>")
    C.append(fig(F["serve"], f'''<b>Hình 4.</b> Normalized latency theo tốc độ request trên (a)
Alpaca và (b) ShareGPT. Số liệu tự đo; thí nghiệm tương ứng với Hình 12 của [1].'''))
    crows, groups = [], []
    for w in R.WORKLOADS:
        groups.append(len(crows))
        for n in R.ORDER:
            c = ceil[(w, n)][0]
            crows.append([WTITLE[w] if n == "paged" else "", LABEL[n], ceil_s(w, n),
                          ratio(w, "paged", n) if n != "paged" else ""])
    C.append(f'''<p>Hình 4 có cùng hình dạng như Hình 12 của [1]: normalized latency tăng chậm, rồi
tăng vọt khi tốc độ đến vượt khả năng phục vụ và hàng đợi bắt đầu dồn lại. Trên ShareGPT, bản phân
trang duy trì được {ceil_s("sharegpt", "paged")} req/s, cao hơn Orca (Max)
{ratio("sharegpt", "paged", "base:max")}, Orca (Pow2) {ratio("sharegpt", "paged", "base:pow2")} và
Orca (Oracle) {ratio("sharegpt", "paged", "base:oracle")} (Bảng 2). Bài gốc báo cáo 2,7–8× so với
Orca (Max) và 1,7–2,7× so với Orca (Oracle). Thứ tự giữa các hệ thống được giữ nguyên, nhưng biên
độ nhỏ hơn; §6 phân tích nguyên nhân.</p>''')
    C.append(tbl(f'''<b>Bảng 2.</b> Throughput tối đa duy trì được (req/s) và lợi thế của bản phân
trang. Dấu ≥ đánh dấu giá trị vẫn đang tăng ở tốc độ đến cao nhất đã đo, tức là cận dưới; khi
cả hai hệ thống đều chưa bão hoà, tỉ số không xác định được (—).''',
                 ["Workload", "Hệ thống", "req/s", "paged / hệ thống"], crows, groups=groups[1:]))
    C.append(f'''<p>Trên Alpaca, ba biến thể Orca (Pow2), Orca (Oracle) và bản phân trang gần như
trùng nhau và chưa bão hoà ở {R.rate_str(max(r["rate"] for r in R.pick(e1, workload="alpaca")))}
req/s, trong khi Orca (Max) bão hoà ở {ceil_s("alpaca", "base:max")} req/s. Điều này khớp với nhận
xét của [1] rằng khi chuỗi ngắn, lợi thế so với Pow2 và Oracle kém rõ rệt: một request Alpaca
trung bình chỉ dùng khoảng 116 token, nên mọi chính sách đặt chỗ hợp lý đều để pool còn thừa.</p>''')

    # 4.3 why
    C.append("<h3>4.3 Vì sao phân trang phục vụ được nhiều hơn</h3>")
    C.append(fig(F["batch"], f'''<b>Hình 5.</b> Số request trung bình trong batch theo tốc độ
request. Số liệu tự đo; thí nghiệm tương ứng với Hình 13 của [1].'''))
    C.append(f'''<p><b>Kích thước batch.</b> Hình 5 cho thấy mắt xích nhân quả. Ở
{R.rate_str(sg_top)} req/s trên ShareGPT, bản phân trang chạy trung bình {v(run_p, ".1f")} request
đồng thời, gấp {v(run_p / run_o, ".2f")}× Orca (Oracle) và {v(run_p / run_m, ".2f")}× Orca (Max)
(bài gốc: 2,2× và 4,3×). Orca (Max) không bao giờ vượt quá 13 request: ⌊{v(pool_b, ",d")} /
2048⌋ = 13, đúng bằng số vùng 2048 token vừa trong pool.</p>''')
    sg_rates = sorted({r["rate"] for r in R.pick(e1, workload="sharegpt")})
    frows = [[R.rate_str(x)] + [v(at("sharegpt", n, x)["frag"], ",d") for n in R.ORDER]
             for x in sg_rates]
    C.append(f'''<p><b>Bộ nhớ.</b> Hình 1 đo nguyên nhân trực tiếp. Orca (Max) chỉ dùng
{u["base:max"]}% vùng đã cấp; phần còn lại là chỗ đặt trước cho token chưa sinh. Pow2 và Oracle đặt
chỗ chính xác hơn nhưng vẫn lãng phí hơn một nửa, vì vùng dành cho token thứ 400 nằm trống trong
suốt thời gian request mới sinh đến token thứ 10, và vì buddy allocator làm tròn lên luỹ thừa 2.
Bảng 3 cho thấy cái giá thứ hai của việc đặt chỗ chính xác: các vùng kích thước khác nhau làm pool
vụn ra, và Pow2 thất bại tới {v(frag[("sharegpt", "base:pow2")], ",d")} lần, Oracle
{v(frag[("sharegpt", "base:oracle")], ",d")} lần trong một lượt chạy vì phân mảnh ngoài. Orca (Max)
không bị phân mảnh ngoài vì mọi vùng cùng kích thước 2048, và bản phân trang không bị vì mọi block
cùng kích thước 16. Trên Alpaca, không biến thể nào bị phân mảnh ngoài: hầu hết đặt chỗ rơi vào
cùng một bậc luỹ thừa 2.</p>''')
    C.append(tbl('''<b>Bảng 3.</b> Số lần cấp phát thất bại do phân mảnh ngoài (tổng bộ nhớ trống
đủ nhưng không có vùng liền mạch đủ lớn) trên ShareGPT, theo tốc độ request.''',
                 ["req/s"] + [LABEL[n].replace(" (paged)", "") for n in R.ORDER], frows))
    C.append(f'''<p><b>Cái giá của over-subscription.</b> Bản phân trang nhận request mà không bảo đảm
chỗ cho đầu ra, nên đôi khi phải preempt. Preempt chỉ xuất hiện khi hệ thống đã ở sát hoặc vượt
ngưỡng bão hoà: trên ShareGPT từ
{R.rate_str(min(r["rate"] for r in pre if r["workload"] == "sharegpt"))} req/s, trên workload chatbot
từ {R.rate_str(min(r["rate"] for r in pre if r["workload"] == "sharegpt_chat"))} req/s, và nhiều nhất
{v(100 * pre_max["preempt"] / (pre_max["num_requests"] or 200), ".1f")}% số request. Thời gian
tính lại chiếm tối đa {v(pre_share, ".1f")}% thời gian chạy. Mọi request bị preempt đều hoàn thành
đủ số token.</p>''')

    # 4.4 TTFT
    C.append("<h3>4.4 Thời gian đến token đầu tiên</h3>")
    C.append(fig(F["ttft"], '''<b>Hình 6.</b> TTFT trung bình trên ShareGPT, trục tung logarit.'''))
    C.append(f'''<p>Normalized latency chia thời gian chờ cho số token đầu ra, nên che bớt trải
nghiệm của người dùng. Hình 6 cho thấy TTFT, tức thời gian nằm hàng đợi cộng prefill. Ở
{R.rate_str(ref)} req/s — tốc độ cao nhất mà hàng đợi của bản phân trang còn gần rỗng — bản phân
trang trả token đầu tiên sau {v(tt_p, ".2f")} s, Orca (Oracle) sau {v(tt_o, ".1f")} s và Orca (Max)
sau {v(tt_m, ".0f")} s ({v(tt_m / tt_p, ".0f")}×). Ở tốc độ thấp nhất, cả bốn đường trùng nhau,
xác nhận rằng khác biệt ở các tốc độ cao hơn đến từ bộ nhớ chứ không từ tốc độ tính toán.</p>''')

    # 4.5 chatbot
    C.append("<h3>4.5 Chatbot</h3>")
    C.append(fig(F["chat"], '''<b>Hình 7.</b> Normalized latency theo tốc độ request trên workload
chatbot. Số liệu tự đo; thí nghiệm tương ứng với Hình 17 của [1].'''))
    C.append(f'''<p>Hình 7 tái hiện Hình 17 của [1]. Prompt của workload này rất dài (trung vị
{meta["sharegpt_chat"]["prompt_len"]["p50"]} token), và bản phân trang duy trì được
{ceil_s("sharegpt_chat", "paged")} req/s so với {ceil_s("sharegpt_chat", "base:max")} của Orca
(Max) ({ratio("sharegpt_chat", "paged", "base:max")}); bài gốc báo cáo 2×. Khác biệt lớn hơn nằm sau
điểm bão hoà: ở {R.rate_str(chat_rate)} req/s, normalized latency của bản phân trang là
{v(chat_p["nl"], ".2f")} s/token, thấp hơn Orca (Max) {v(chat_m["nl"] / chat_p["nl"], ".1f")}×.</p>''')
    C.append('''<p>[1] quan sát rằng ba biến thể Orca hành xử như nhau ở đây, vì prompt của hầu hết
request chạm trần 1024 token và buddy allocator làm tròn mọi đặt chỗ lên 2048. Ở đây chỉ Pow2 và
Oracle trùng nhau, còn Max tệ hơn. Nguyên nhân tính được từ workload: chỉ 46% prompt chạm trần
1024, và với số còn lại Pow2 và Oracle vẫn đưa được khoảng một phần ba số đặt chỗ xuống mức
≤ 1024 token, trong khi Max làm tròn 100% đặt chỗ lên 2048.</p>''')

    # ---------------- 5 ablations ----------------
    C.append("<h2>5 Ablation</h2>")
    C.append(f'''<h3>5.1 Kernel attention</h3><p>Truy cập qua block table thêm một bước tra bảng và
làm KV cache của một request nằm rải rác. Để đo riêng chi phí này, kernel attention decode được chạy
tách khỏi phần còn lại của mô hình, với cùng độ dài ngữ cảnh, cùng batch và cùng dữ liệu;
hai kernel cho ra kết quả giống nhau từng bit. Trái với mức chậm hơn 20–26% trong [1], kernel phân
trang ở đây chạy bằng {v(min(k_ratios), ".2f")}–{v(max(k_ratios), ".2f")}× thời gian của kernel liền
mạch (Hình 8). Khác biệt nằm ở baseline: [1] so với kernel của FasterTransformer, được tối ưu riêng
rất sâu, còn ở đây hai kernel cùng một cài đặt, chỉ khác cách tính địa chỉ. Offset trong một block
vừa với số nguyên 32 bit và vòng lặp trong chạy trên block size là hằng số lúc biên dịch, trong khi
bản liền mạch phải dùng offset 64 bit. Kết luận đúng không phải "phân trang nhanh hơn" mà là chi phí
tra bảng nhỏ hơn khác biệt giữa hai cách viết cùng một kernel.</p>''')
    C.append(fig(F["kernel"], '''<b>Hình 8.</b> Độ trễ kernel attention decode, phân trang so với
liền mạch, cùng số byte KV cache. Số liệu tự đo; thí nghiệm tương ứng với Hình 18a của [1].'''))
    C.append(fig(F["block"], '''<b>Hình 9.</b> Normalized latency theo block size, với tốc độ
request cố định. Số liệu tự đo; thí nghiệm tương ứng với Hình 18b của [1].'''))
    C.append(f'''<h3>5.2 Block size</h3><p>Block nhỏ làm mỗi lần đọc KV cache lấy được ít byte liên
tiếp; block lớn làm phần trống của block cuối lớn lên. Hình 9 quét block size từ 1 đến 256 với số slot
cố định 64 để block table vừa VRAM ở block size 1. Trên ShareGPT, block size từ 16 đến 128 cho kết quả
tốt nhất và cách nhau không quá {v(100 * best16_128, ".1f")}%, đúng như [1]; block 1 chậm hơn
{v(100 * (sg_bs[1] / sg_bs[16] - 1), ".1f")}% và block 256 chậm hơn
{v(100 * (sg_bs[256] / sg_bs[16] - 1), ".1f")}% so với 16. Trên Alpaca, [1] thấy block lớn làm hiệu
năng giảm mạnh; ở đây mọi block size nằm trong khoảng {v(100 * al_pen, ".1f")}% của nhau. Phân mảnh
trong vẫn xảy ra — ở block 256, hơn một nửa bộ nhớ đã cấp cho Alpaca bị trống — nhưng chỉ biến thành
độ trễ khi pool chật, và với chuỗi ngắn của Alpaca pool không bao giờ đầy quá khoảng 17% ở các tốc độ
đã đo.</p>''')
    C.append(f'''<h3>5.3 Tính lại so với swap</h3><p>Workload chatbot được chạy ở 1,2 req/s, nơi
pool bị dùng gần hết, với từng chính sách phục hồi và từng block size, mỗi điểm có
{min(events)}–{max(events)} lần preempt. Xu hướng khớp với [1]: chi phí tính lại gần như không phụ
thuộc block size ({v(rc_lo / 1000, ".2f")}–{v(rc_hi / 1000, ".2f")} s), còn chi phí swap giảm khi
block lớn lên, từ {v(sw_by[min(sw_by)], ".0f")} ms ở block {min(sw_by)} xuống
{v(sw_by[max(sw_by)], ".1f")} ms ở block {max(sw_by)}, vì block nhỏ tạo ra nhiều lượt truyền PCIe
nhỏ. Nhưng điểm giao nhau không xuất hiện: [1] thấy tính lại hiệu quả hơn ở block nhỏ và không bao
giờ đắt hơn swap quá 20%, còn ở đây tính lại đắt hơn swap {v(min(rs_ratio), ".0f")}–{v(max(rs_ratio), ".0f")}×
ở mọi block size. Lý do là phía tính toán: tính lại phải prefill lại hơn một nghìn token trên một GPU
không có tensor core, trong khi KV cache của một request chỉ khoảng 64 MiB và đi qua PCIe trong vài
chục ms. Vì preempt hiếm, ảnh hưởng end-to-end nhỏ: normalized latency của mọi cấu hình nằm trong
{v(min(nl_rec), ".1f")}–{v(max(nl_rec), ".1f")} s/token (Hình 10b).</p>''')
    C.append(fig(F["rec"], f'''<b>Hình 10.</b> (a) Chi phí một lần phục hồi request bị preempt theo
block size. (b) Normalized latency end-to-end trên workload chatbot ở 1,2 req/s. Số liệu tự đo;
thí nghiệm tương ứng với Hình 19 của [1].'''))

    # ---------------- 6 discussion ----------------
    C.append("<h2>6 Thảo luận</h2>")
    C.append(f'''<p><b>Vì sao lợi thế nhỏ hơn bài gốc.</b> Phân trang tăng số request vừa trong
bộ nhớ, nhưng điều đó chỉ thành throughput khi bộ nhớ là nút thắt. Trên ShareGPT ở
{R.rate_str(sg_top)} req/s, bản phân trang đã bão hoà trong khi KV cache của nó chỉ được dùng trung
bình {v(usage_paged_top["live_pct"], ".0f")}%: nút thắt đã chuyển sang sức tính, chủ yếu là prefill
— phép nhân ma trận chạy trên lõi FP32 ở khoảng 2,7 TFLOP/s. Trên A100 với mô hình 13B trở lên, sức
tính dư thừa hơn nhiều so với bộ nhớ, nên mỗi request thêm vào batch đều chuyển thành throughput và
lợi thế của phân trang hiện ra đầy đủ. Workload chatbot, với prompt dài nhất và prefill nặng nhất,
có lợi thế nhỏ nhất, nhất quán với giải thích này. Nói cách khác, lợi ích của phân trang bị chặn trên
bởi tỉ số giữa trần tính toán của GPU và trần bộ nhớ của baseline.</p>''')
    C.append('''<p><b>Ngoài phạm vi.</b> Bài chưa đánh giá các tình huống chia sẻ KV cache của [1]:
parallel sampling, beam search và tiền tố dùng chung (§6.3–6.4). Engine phân trang đã hỗ trợ chia
sẻ block với copy-on-write, nhưng các thí nghiệm tương ứng chưa được hoàn tất. Bài cũng không có
baseline FasterTransformer.</p>''')

    # ---------------- 7 conclusion ----------------
    C.append("<h2>7 Kết luận</h2>")
    C.append(f'''<p>Trên một GPU 4 GB không có tensor core, PagedAttention giữ được kết luận định
tính của [1]: nó loại bỏ gần như toàn bộ lãng phí KV cache ({u["paged"]}% bộ nhớ đã cấp chứa token,
so với {u["base:max"]}–{u["base:oracle"]}% ở các baseline), tăng số request xử lý đồng thời, không
có phân mảnh ngoài, và nhờ đó duy trì được tốc độ request cao hơn
{ratio("sharegpt", "paged", "base:max")} so với Orca (Max) trên ShareGPT. Biên độ nhỏ hơn bài gốc vì
trên phần cứng yếu về tính toán, bộ nhớ thôi là nút thắt trước khi phân trang dùng hết lợi thế của
nó. Chi phí của phân trang — tra block table, preempt khi over-subscribe — đo được là nhỏ. Kết quả
cũng cho thấy lựa chọn chính sách phục hồi phụ thuộc phần cứng: trên GPU yếu về tính toán, swap
luôn thắng tính lại.</p>''')

    refs = [
        "Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. "
        "Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language "
        "Model Serving with PagedAttention. In <i>SOSP '23</i>.",
        "Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: "
        "A Distributed Serving System for Transformer-Based Generative Models. In <i>OSDI '22</i>.",
        "NVIDIA. FasterTransformer. https://github.com/NVIDIA/FasterTransformer.",
        "ShareGPT Team. 2023. https://sharegpt.com/.",
        "Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy "
        "Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-following LLaMA "
        "model. https://github.com/tatsu-lab/stanford_alpaca.",
        "Meta AI. 2024. Llama 3.2: Lightweight Models. https://huggingface.co/meta-llama/"
        "Llama-3.2-1B-Instruct.",
    ]
    C.append('<h2>Tài liệu tham khảo</h2><ol class="refs">'
             + "".join(f'<li id="ref-{i}">{r}</li>' for i, r in enumerate(refs, 1)) + "</ol>")

    body = link_citations("".join(P) + f'<div class="cols">{"".join(C)}</div>')
    doc = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Nghiên cứu PagedAttention</title><style>{CSS}</style></head>'
           f'<body><main>{body}</main></body></html>')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {path}  ({len(doc) / 1024:.0f} KB)")

    if not args.no_pdf:
        chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser")
                       if shutil.which(c)), None)
        if not chrome:
            print("no Chrome/Chromium found; skipping PDF")
            return
        pdf = os.path.join(OUT, "paper.pdf")
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", f"file://{path}"],
                       check=True, capture_output=True, timeout=120)
        print(f"wrote {pdf}")
        check_layout(pdf)


if __name__ == "__main__":
    main()
