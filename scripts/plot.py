"""Figures for the report, from the data in results/.

    python3 scripts/summarize.py      # results/timings.txt -> results/timings_median.csv
    python3 scripts/plot.py           # -> report/figs/*.pdf and *.png

Figures:
  time_<graph>    time per phase against threads p, one panel per phase
  speedup         speedup T_seq / T(p) of the whole gradient and of the walk
  depth           walk time against the quadtree depth L (sequential)
  scaling_n       sequential time per phase against n, across the graphs
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                       # files only, no window
import matplotlib.pyplot as plt
from matplotlib import ticker

OUT = "report/figs"
PHASES = [("P", "P"), ("attr", "attraction"), ("build", "tree build"),
          ("mass", "centers of mass"), ("walk", "walk")]
TOTAL = ["P", "attr", "build", "mass", "walk", "grad"]   # the whole gradient

# One color + one marker per version (validated categorical slots 1-3; the
# markers keep the lines apart in grayscale print). seq is a reference line.
STYLE = {"omp":  dict(color="#2a78d6", marker="o", label="OpenMP"),
         "cilk": dict(color="#eb6834", marker="s", label="OpenCilk"),
         "pth":  dict(color="#1baf7a", marker="^", label="Pthreads")}
SEQ  = dict(color="#52514e", linestyle="--", linewidth=1.2, label="sequential")
IDEAL = dict(color="#898781", linestyle=":", linewidth=1.2, label="ideal")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load_medians(path="results/timings_median.csv"):
    """(graph, version, p) -> row of medians (floats where numeric)."""
    rows = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
        rows[(r["graph"], r["version"], int(r["p"]))] = r
    return rows


def seconds(x, _pos=None):
    """Tick label in plain units: 2 ms, 0.5 s, 50 us."""
    if x >= 1:
        return f"{x:g} s"
    if x >= 1e-3:
        return f"{x * 1e3:g} ms"
    return f"{x * 1e6:g} \u00b5s"


def time_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(seconds))
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def thread_axis(ax, ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.minorticks_off()
    ax.set_xlabel("threads p")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


def fig_time(rows, graph):
    """Time per phase vs p: one panel per phase, own y scale (log)."""
    ps = sorted({p for (g, v, p) in rows if g == graph and v != "seq"})
    fig, axes = plt.subplots(1, len(PHASES), figsize=(7.2, 1.9))
    for ax, (key, title) in zip(axes, PHASES):
        seq = rows[(graph, "seq", 1)][key]
        ax.axhline(seq, **SEQ)
        for v, st in STYLE.items():
            ax.plot(ps, [rows[(graph, v, p)][key] for p in ps], **st)
        time_axis(ax)
        ax.margins(y=0.12)                  # keep end labels off the axis
        ax.set_title(title)
        thread_axis(ax, ps)
        ax.tick_params(axis="x", labelsize=6)  # narrow panels: keep "8" and "12" apart
        ax.set_xlabel("")
    axes[0].set_ylabel("time (median of 5)")
    fig.supxlabel("threads p", fontsize=8, y=-0.08)
    fig.subplots_adjust(wspace=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("")
    save(fig, f"time_{graph}")


def fig_speedup(rows):
    """Speedup T_seq / T(p) of the whole gradient (all phases), per graph."""
    graphs = sorted({g for (g, v, p) in rows})
    fig, axes = plt.subplots(1, len(graphs), figsize=(7.2, 2.3), sharey=True)
    if len(graphs) == 1:
        axes = [axes]
    for ax, graph in zip(axes, graphs):
        ps = sorted({p for (g, v, p) in rows if g == graph and v != "seq"})
        seq = rows[(graph, "seq", 1)]
        t_seq = sum(seq[k] for k in TOTAL)
        ax.plot(ps, ps, **IDEAL)
        for v, st in STYLE.items():
            total = [t_seq / sum(rows[(graph, v, p)][k] for k in TOTAL) for p in ps]
            ax.plot(ps, total, **st)
        ax.set_title(f"{graph} (n = {int(seq['n']):,})")
        thread_axis(ax, ps)
        ax.set_yscale("log", base=2)
        ax.set_yticks(ps)
        ax.set_yticklabels([str(p) for p in ps])
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    axes[0].set_ylabel("speedup over sequential\n(whole gradient, all phases)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    save(fig, "speedup")


def fig_depth(path="results/depth_sweep_seq.txt"):
    """Walk time vs quadtree depth L, one line per graph (log y)."""
    data = defaultdict(list)
    for line in open(path):
        m = re.match(r"(\S+) tree \(L = (\d+)\).*walk ([\d.]+) s", line)
        if m:
            data[m.group(1)].append((int(m.group(2)), float(m.group(3))))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    # graphs, not versions: neutral ink, told apart by line style and marker
    graph_style = [dict(color=INK, marker="o", linestyle="-"),
                   dict(color="#52514e", marker="s", linestyle="--"),
                   dict(color=MUTED, marker="^", linestyle=":")]
    for (graph, pts), st in zip(sorted(data.items()), graph_style):
        L, t = zip(*sorted(pts))
        best = min(pts, key=lambda x: x[1])
        ax.plot(L, t, label=f"{graph} (best L = {best[0]})", **st)
        ax.plot(*best, marker="o", markersize=10, markerfacecolor="none",
                markeredgecolor=st["color"], linestyle="none")
    time_axis(ax)
    ax.set_xlabel("quadtree depth L (2^L x 2^L leaves)")
    ax.set_ylabel("walk time, sequential")
    ax.legend(frameon=False)
    save(fig, "depth")


def fig_scaling_n(rows):
    """Sequential time per phase vs n across the graphs (log-log)."""
    graphs = sorted({g for (g, v, p) in rows}, key=lambda g: rows[(g, "seq", 1)]["n"])
    if len(graphs) < 2:
        return
    n = [rows[(g, "seq", 1)]["n"] for g in graphs]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (key, title), c, m in zip(PHASES, colors, markers):
        ax.plot(n, [rows[(g, "seq", 1)][key] for g in graphs],
                color=c, marker=m, label=title)
    ax.set_xscale("log")
    time_axis(ax)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.set_xticks(n)
    ax.set_xticklabels([f"{g}\n{int(x):,}" for g, x in zip(graphs, n)], fontsize=6)
    ax.minorticks_off()
    ax.set_xlabel("graph, n")
    ax.set_ylabel("sequential time")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "scaling_n")


if __name__ == "__main__":
    rows = load_medians()
    for graph in sorted({g for (g, v, p) in rows}):
        fig_time(rows, graph)
    fig_speedup(rows)
    fig_depth()
    fig_scaling_n(rows)
