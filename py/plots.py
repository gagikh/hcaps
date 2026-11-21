"""Plots of the benchmark results (replaces QCustomPlot, tst/graph_plot and tst/plotting_utils).

    python py/plots.py [--data plots/data] [--out plots] [--show]
    python py/plots.py --store          after a benchmark run: keep it and draw it

--store copies the CSV files of the last run (plots/data) into <repo>/<gpu>/<run>/data and draws
the graphs into <repo>/<gpu>/<run>: <gpu> is the folder of that GPU (4090, 5090, a100-sxm, h100,
h200, ... from the device name in the files, --gpu to override), <run> the start time of the run
('# run:' in the files). Earlier runs stay untouched; redraw any of them with
    python py/plots.py --data 4090/2026-09-26_14-05-33/data --out 4090/2026-09-26_14-05-33

The C++ benchmarks (test_gemm_benchmark) write CSV files into plots/data (tst/results.hpp):

    series files  relative_costs, communication_costs, batched_gemm_vs_gemm, warp_and_l2_usage:
                  one graph with a curve per series; relative_costs is drawn as "Costs" (the
                  absolute values) and "Relative Costs" (each CAPS series / cuBLAS, log scale,
                  < 1 = CAPS faster); runs that also timed our own GEMM on the whole matrix
                  (series "OWN wgmma" / "OWN mma", R = 0) get "Relative Costs vs our <kernel>"
                  too: CAPS against its own leaf kernel, the saving of the recursion alone;
                  runs with "CUBLAS best" (cuBLAS's fastest heuristic candidate, chosen by
                  timing) get "Relative Costs vs cuBLAS best", the fair comparison
    stats files   cublas + caps_cublas: the performance / precision comparisons of the old
                  plotting_utils (best, normalized best, per R and per range); initialization_time:
                  min / max / avg per R

Every graph is saved as <out>/<title>.png (the old names: spaces and ,+=/() replaced by _).
Line styles as before: min solid, max dashed, avg dotted.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

import matplotlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# common.hpp color_table (stored as BGR there)
COLOR_TABLE_BGR = [
    (89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98), (130, 172, 179),
    (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45), (44, 52, 10), (101, 158, 121), (179, 124, 12),
    (25, 33, 189), (45, 115, 11), (73, 197, 184), (62, 225, 221), (32, 46, 52), (20, 165, 16), (54, 15, 57),
    (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106), (42, 10, 96), (7, 164, 128), (98, 213, 120),
    (40, 5, 219), (54, 25, 150), (251, 74, 172), (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25),
    (191, 106, 197), (8, 15, 134), (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253), (155, 22, 122),
    (218, 130, 77), (164, 102, 79), (43, 152, 125), (185, 124, 151), (95, 159, 238), (128, 89, 85), (228, 6, 60),
    (6, 41, 210), (11, 1, 133), (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165), (32, 111, 29),
    (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94), (55, 35, 92), (129, 140, 233),
    (70, 250, 116), (61, 209, 152), (216, 21, 138), (100, 0, 176), (3, 42, 70), (151, 13, 44), (216, 102, 88),
    (125, 216, 93), (171, 236, 47), (253, 127, 103), (205, 137, 244), (193, 137, 224), (36, 152, 214),
    (17, 50, 238), (154, 165, 67), (114, 129, 60), (119, 24, 48), (73, 8, 110)]
RED, BLUE, GREEN = "#ff0000", "#0000ff", "#00ff00"


def color(i: int) -> str:
    b, g, r = COLOR_TABLE_BGR[i % len(COLOR_TABLE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


# ------------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------------

def read_result(path: str):
    """(meta, rows): '# key: value' header lines, then the CSV rows as dicts."""
    meta, lines = {}, []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            elif line.strip():
                lines.append(line)
    return meta, list(csv.DictReader(lines))


def series_of(rows):
    series = defaultdict(dict)
    for r in rows:
        series[r["series"]][int(r["N"])] = float(r["value"])
    return series


def stats_of(rows, kind):
    """range -> R -> N -> {min, max, avg, std, rel} for type == kind"""
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["type"] == kind:
            out[int(r["range"])][int(r["R"])][int(r["N"])] = {k: float(r[k]) for k in ("min", "max", "avg", "std", "rel")}
    return out


def mean_max_avg(stats, R):
    """test_utils calculate_mean_max_avg(): per N over all ranges: min of min, max of max,
    mean of avg and rel. Returns (x, min, max, avg, rel)."""
    per_n = defaultdict(list)
    for rng in sorted(stats):
        for N, s in stats[rng].get(R, {}).items():
            per_n[N].append(s)
    x = sorted(per_n)
    return (x, [min(s["min"] for s in per_n[n]) for n in x], [max(s["max"] for s in per_n[n]) for n in x],
            [sum(s["avg"] for s in per_n[n]) / len(per_n[n]) for n in x],
            [sum(s["rel"] for s in per_n[n]) / len(per_n[n]) for n in x])


def best(stats):
    """test_utils calculate_best(): per N the smallest max over all ranges and R."""
    result = {}
    for rng in stats.values():
        for per_r in rng.values():
            for N, s in per_r.items():
                result[N] = min(result.get(N, float("inf")), s["max"])
    return dict(sorted(result.items()))


def recursions(stats):
    return sorted({R for rng in stats.values() for R in rng})


# ------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------

class Plotter:
    def __init__(self, out_dir, device, show=False):
        self.out_dir, self.device, self.show = out_dir, device, show
        self.written = []

    def figure(self, title, y_label):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
        ax.set_title(f"{title}, Device={self.device}", fontsize=14, fontweight="bold")
        ax.set_xlabel("dimension", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="major", alpha=0.3)
        return fig, ax

    @staticmethod
    def min_max_avg(ax, c, text, x, minv, maxv, avgv):
        if minv:
            ax.plot(x, minv, color=c, linestyle="-", linewidth=2, label=f"min, {text}")
        if maxv:
            ax.plot(x, maxv, color=c, linestyle="--", linewidth=2, label=f"max, {text}")
        if avgv:
            ax.plot(x, avgv, color=c, linestyle=":", linewidth=2, label=f"avg, {text}")

    def save(self, fig, ax, test_id):
        import matplotlib.pyplot as plt
        xs = sorted({int(v) for line in ax.get_lines() for v in line.get_xdata()})
        if xs:
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in xs])
        # cuBLAS first, then BFS (solid), then DFSCaps (dashed / dotted); two columns when DFSCaps is there,
        # so each column is one pipeline
        handles, labels = ax.get_legend_handles_labels()
        # the baselines (cuBLAS, cuBLAS best, our R = 0 GEMM, the "= 1" line) first
        rank = lambda label: 0 if ("cuBLAS" in label or label.startswith("our ") or label.endswith("= 1")) else 2 if label.startswith("DFSCaps") else 1
        order = sorted(range(len(labels)), key=lambda i: (rank(labels[i]), labels[i]))
        two = any(lbl.startswith("DFSCaps") for lbl in labels)
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                  ncol=2 if two else 1, fontsize=12, handlelength=4, framealpha=0.95)
        fig.tight_layout()
        self.save_figure(fig, test_id)

    def save_figure(self, fig, test_id):
        """Write the finished figure (layout and legend done by the caller) as <test_id>.png"""
        import matplotlib.pyplot as plt
        name = re.sub(r"[ /+,=()]", "_", test_id) + ".png"
        path = os.path.join(self.out_dir, name)
        fig.savefig(path)
        self.written.append(path)
        if self.show:
            plt.show()
        plt.close(fig)


def series_label(name: str) -> str:
    """Legend text of a relative_costs series: the CAPS series are BFSCaps"""
    if name.startswith("CAPS "):
        return "BFSCaps " + name[len("CAPS "):] + "  (solid)"
    if name.startswith("DFSCaps overlapped "):
        return name + "  (dotted)"
    if name.startswith("DFSCaps "):
        return name + "  (dashed)"
    if name == "CUBLAS":
        return "cuBLAS"
    if name == "CUBLAS best":  # the fastest candidate of cuBLASLt's heuristic (fair baseline)
        return "cuBLAS best"
    if name.startswith("OWN "):  # our own GEMM on the whole matrix (R = 0): the second baseline
        return "our " + name[len("OWN "):] + " (R=0)"
    return name


def baseline_style(name: str):
    """(colour, line style, marker) of the baselines (series without R): cuBLAS as a user gets it
    (cublasHgemm) red, cuBLAS's fastest candidate dark red dotted, our R = 0 GEMM black dash-dotted"""
    if name == "CUBLAS best":
        return "darkred", ":", "v"
    if name.startswith("OWN "):
        return "black", "-.", "D"
    return RED, "-", "o"


def own_series_name(series):
    """The "OWN <kernel>" series of relative_costs (our leaf GEMM on the whole N x N matrix,
    R = 0), or None for runs that did not measure it"""
    return next((name for name in sorted(series) if name.startswith("OWN ")), None)


def plot_series(p: Plotter, meta, rows):
    fig, ax = p.figure(meta["title"], meta["y_label"])
    for name, points in series_of(rows).items():
        m = re.search(r"R=(\d+)", name)
        c = color(int(m.group(1))) if m else RED
        x = sorted(points)
        style = (":" if name.startswith("DFSCaps overlapped") else "--" if name.startswith("DFSCaps") else "-")  # DFSCaps: dashed (overlapped dotted), colour of its R
        if not m:  # a baseline: cuBLAS, cuBLAS best, our R = 0 GEMM
            c, style, _ = baseline_style(name)
        ax.plot(x, [points[n] for n in x], color=c, linestyle=style, linewidth=2, label=series_label(name))
    p.save(fig, ax, meta["title"])


def plot_relative(p: Plotter, meta, rows, base_name: str, test_id: str):
    """Every other series of relative_costs divided by the base series at the same N (time or
    energy ratio, < 1: faster than the base), on a log y axis so that small and large N are equally
    readable; the dotted line at 1 is the base. Each point is labelled with its ratio.
    Base "CUBLAS": the "Relative Costs" graph. Base "OWN <kernel>" (our GEMM on the whole matrix,
    R = 0): CAPS against its own leaf kernel, i.e. the saving of the recursion alone. The other
    baseline appears as a black dash-dotted line."""
    series = series_of(rows)
    base = series.get(base_name)
    if not base:
        return
    base_text = series_label(base_name)
    measure = "time" if "time" in meta["y_label"] else "energy"
    fig, ax = p.figure(f"Relative Costs ({measure} / {base_text})", meta["y_label"].split(" (")[0] + " / " + base_text)
    ax.set_yscale("log", base=2)
    lowest, highest = 1.0, 1.0
    all_x = set()
    for name, points in sorted(series.items()):
        if name == base_name:
            continue
        x = sorted(n for n in points if n in base and base[n] > 0)
        y = [points[n] / base[n] for n in x]
        if not x:
            continue
        all_x.update(x)
        m = re.search(r"R=(\d+)", name)
        if m:
            c = color(int(m.group(1)))
            style = (":" if name.startswith("DFSCaps overlapped") else "--" if name.startswith("DFSCaps") else "-")  # DFSCaps: dashed (overlapped dotted), colour of its R
            marker = "s" if style == "--" else "o"
        else:  # the other baselines (cuBLAS, cuBLAS best, our R = 0 GEMM)
            c, style, marker = baseline_style(name)
        ax.plot(x, y, color=c, linestyle=style, linewidth=2, marker=marker, label=series_label(name))
        for n, v in zip(x, y):
            ax.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(4, 4), fontsize=9, color=c)
        lowest, highest = min(lowest, min(y)), max(highest, max(y))
    if not all_x:
        return
    # the cuBLAS line as data over the measured N (axhline would add x = 0 .. 1 to the ticks of save())
    x = sorted(all_x)
    ax.plot(x, [1.0] * len(x), color=RED, linestyle=":", linewidth=2.5, label=f"{base_text} = 1")
    ticks = [t for t in (0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8, 16) if lowest / 1.2 <= t <= highest * 1.2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    p.save(fig, ax, test_id)


def plot_min_max_avg_per_r(p: Plotter, meta, rows):
    perf = stats_of(rows, "performance")
    fig, ax = p.figure(meta["title"], meta["y_label"])
    for R in recursions(perf):
        x, mn, mx, avg, _ = mean_max_avg(perf, R)
        p.min_max_avg(ax, color(R), f"R={R}", x, mn, mx, avg)
    p.save(fig, ax, meta["title"])


def plot_benchmark(p: Plotter, cublas_meta, cublas_rows, caps_rows):
    """The comparisons test_gemm_benchmark used to draw (same titles, same order)."""
    units = cublas_meta["y_label"]
    ref_perf, ref_prec = stats_of(cublas_rows, "performance"), stats_of(cublas_rows, "precision")
    caps_perf, caps_prec = stats_of(caps_rows, "performance"), stats_of(caps_rows, "precision")

    # best_performance / normalized_best_performance
    b = best(caps_perf)
    rx, rmin, rmax, ravg, _ = mean_max_avg(ref_perf, 1)
    fig, ax = p.figure("Compare caps vs cublas", units)
    ax.plot(list(b), list(b.values()), color=BLUE, linewidth=2, label="CAPS+CUBLAS, best R")
    p.min_max_avg(ax, RED, "CUBLAS", rx, rmin, rmax, ravg)
    p.save(fig, ax, "Compare caps vs cublas")

    ref_avg = dict(zip(rx, ravg))
    fig, ax = p.figure("Compare caps vs cublas(relative)", "relative performance")
    nx = [n for n in b if ref_avg.get(n, 0) > 0]
    ax.plot(nx, [b[n] / ref_avg[n] for n in nx], color=BLUE, linewidth=2, label="CAPS+CUBLAS, best R / CUBLAS")
    p.save(fig, ax, "Compare caps vs cublas(relative)")

    # compare_perf3, per R
    for R in recursions(caps_perf) or recursions(ref_perf):
        title = f"Energy Consumption, R={R}"
        fig, ax = p.figure(title, units)
        for c, name, src in ((BLUE, "CUBLAS", ref_perf), (RED, "CAPS+CUBLAS", caps_perf)):
            x, mn, mx, avg, _ = mean_max_avg(src, R)
            p.min_max_avg(ax, c, name, x, mn, mx, avg)
        p.save(fig, ax, title)

    # compare_precision2, per range and R
    for rng in sorted(caps_prec):
        for R in sorted(caps_prec[rng]):
            if not ref_prec.get(rng, {}).get(R):
                continue
            title = f"Precision (vs mkl), R={R}, range={rng}"
            fig, ax = p.figure(title, "error")
            for c, name, src in ((BLUE, "CUBLAS", ref_prec[rng][R]), (RED, "CAPS/CAPS+CUBLAS", caps_prec[rng][R])):
                x = sorted(src)
                p.min_max_avg(ax, c, name, x, [], [src[n]["max"] for n in x], [src[n]["avg"] for n in x])
            p.save(fig, ax, title)

    # draw_perf_stats
    title = "CAPS+CUBLAS performance"
    fig, ax = p.figure(title, units)
    for R in recursions(caps_perf):
        x, mn, mx, avg, _ = mean_max_avg(caps_perf, R)
        p.min_max_avg(ax, color(R), f"R={R}", x, mn, mx, avg)
    p.min_max_avg(ax, RED, "CUBLAS", rx, rmin, rmax, ravg)
    p.save(fig, ax, title)

    # draw_precision_stats
    title = "CAPS+CUBLAS vs mkl precision"
    fig, ax = p.figure(title, "error")
    color_id = 0
    for rng in sorted(caps_prec):
        for R in sorted(caps_prec[rng]):
            src = caps_prec[rng][R]
            x = sorted(src)
            p.min_max_avg(ax, color(color_id), f"R={R}, range={rng}", x, [], [src[n]["max"] for n in x],
                          [src[n]["avg"] for n in x])
            color_id += 1
    p.save(fig, ax, title)

    # compare_precision, per range and R (max error)
    for rng in sorted(caps_prec):
        for R in sorted(caps_prec[rng]):
            if not ref_prec.get(rng, {}).get(R):
                continue
            title = f"CAPS+CUBLAS vs CUBLAS precision, R={R}, range={rng}"
            fig, ax = p.figure(title, "max error")
            for style, name, src in (("-", "caps+cublas", caps_prec[rng][R]), (":", "cublas", ref_prec[rng][R])):
                x = sorted(src)
                ax.plot(x, [src[n]["max"] for n in x], color=BLUE, linestyle=style, linewidth=2, label=name)
            p.save(fig, ax, title)

    # compare_perf, per R
    for R in recursions(caps_perf):
        title = f"CAPS+CUBLAS vs CUBLAS performance, R={R}"
        fig, ax = p.figure(title, units)
        for c, name, src in ((BLUE, "caps+cublas", caps_perf), (GREEN, "cublas", ref_perf)):
            x, mn, mx, avg, _ = mean_max_avg(src, R)
            p.min_max_avg(ax, c, name, x, mn, mx, avg)
        p.save(fig, ax, title)


def plot_own_vs_cublas(p: Plotter, meta, rows, own_name: str):
    """Our GEMM on the whole matrix (R = 0, series "OWN <kernel>") against both cuBLAS baselines:
    our time / cuBLAS as a user gets it (cublasHgemm, red) and our time / cuBLAS's fastest heuristic
    candidate (dark red dotted), log y axis, each point labelled; < 1: ours is faster. The plain GEMM
    comparison without CAPS (the "hgemm" question), e.g. Relative_Costs_our_wgmma_vs_cuBLAS.png. Also cuBLAS's
    fastest candidate / cuBLAS (gray dashed): how much the default cublasHgemm choice leaves."""
    series = series_of(rows)
    own = series.get(own_name)
    if not own:
        return
    own_text = own_name[len("OWN "):]
    measure = "time" if "time" in meta["y_label"] else "energy"
    fig, ax = p.figure(f"Relative Costs: our {own_text} (R=0) vs cuBLAS ({measure} ratio)",
                       meta["y_label"].split(" (")[0] + f" of our {own_text} / cuBLAS")
    ax.set_yscale("log", base=2)
    lowest, highest = 1.0, 1.0
    all_x = set()
    for base_name in ("CUBLAS", "CUBLAS best"):
        base = series.get(base_name)
        if not base:
            continue
        x = sorted(n for n in own if n in base and base[n] > 0)
        if not x:
            continue
        y = [own[n] / base[n] for n in x]
        all_x.update(x)
        c, style, marker = baseline_style(base_name)
        ax.plot(x, y, color=c, linestyle=style, linewidth=2, marker=marker,
                label=f"our {own_text} / {series_label(base_name)}")
        for n, v in zip(x, y):
            ax.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(4, 4), fontsize=9, color=c)
        lowest, highest = min(lowest, min(y)), max(highest, max(y))
    if not all_x:
        return
    # cuBLAS best / cuBLAS: the gain of picking cuBLAS's fastest candidate over its default
    plain, fastest = series.get("CUBLAS"), series.get("CUBLAS best")
    if plain and fastest:
        x = sorted(n for n in fastest if n in plain and plain[n] > 0)
        if x:
            y = [fastest[n] / plain[n] for n in x]
            ax.plot(x, y, color="gray", linestyle="--", linewidth=2, marker="^", label="cuBLAS best / cuBLAS")
            for n, v in zip(x, y):
                ax.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(4, -12), fontsize=9, color="gray")
            lowest, highest = min(lowest, min(y)), max(highest, max(y))
            all_x.update(x)
    x = sorted(all_x)
    ax.plot(x, [1.0] * len(x), color="gray", linestyle=":", linewidth=2.5, label="cuBLAS = 1")
    ticks = [t for t in (0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 3, 4, 6, 8, 16) if lowest / 1.2 <= t <= highest * 1.2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    p.save(fig, ax, f"Relative Costs our {own_text} vs cuBLAS")  # Relative_Costs_our_wgmma_vs_cuBLAS.png


def plot_batched(p: Plotter, meta, rows):
    """batched_costs.csv (tst/gemm_benchmark.cpp batched_costs()): batches of 7^R leaf products, one
    image per R (Relative_Costs_batched_R<r>_wgmma_vs_cuBLAS.png), x = leaf size, all images on the
    same log y range so they compare directly. In each the same four ratios, the same colour and
    style each: our wgmma / cuBLAS strided batched, our wgmma / cuBLAS best batched, cuBLAS best /
    cuBLAS batched, cuBLAS pointer-array / strided batched (how CAPS would call cuBLAS); < 1: the
    first is faster; our two ratios labelled at every point, the legend above the axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    series = series_of(rows)
    rs = sorted({int(m.group(1)) for name in series for m in [re.search(r"R=(\d+)", name)] if m})
    if not rs:
        return
    measure = "time" if "time" in meta["y_label"] else "energy"
    pairs = [("OWN wgmma", "CUBLAS batched", "#1f77b4", "-", "o", "our wgmma / cuBLAS batched"),
             ("OWN wgmma", "CUBLAS best batched", "#d62728", "-", "v", "our wgmma / cuBLAS best batched"),
             ("CUBLAS best batched", "CUBLAS batched", "#7f7f7f", "--", "s", "cuBLAS best / cuBLAS batched"),
             ("CUBLAS pointers", "CUBLAS batched", "#9467bd", ":", "D", "cuBLAS pointer array / strided")]

    def ratios(r):
        """[(pair, x, y)] of batch R = r"""
        out = []
        for pair in pairs:
            num, den = series.get(f"{pair[0]} R={r}"), series.get(f"{pair[1]} R={r}")
            if not num or not den:
                continue
            x = sorted(n for n in num if n in den and den[n] > 0)
            if x:
                out.append((pair, x, [num[n] / den[n] for n in x]))
        return out

    # one y range for all images
    values = [v for r in rs for _, _, y in ratios(r) for v in y]
    lowest, highest = min(values + [1.0]), max(values + [1.0])
    ticks = [t for t in (0.25, 0.5, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 3, 4) if lowest / 1.1 <= t <= highest * 1.1]
    handles = [Line2D([0], [0], color=c, linestyle=style, linewidth=2, marker=marker) for _, _, c, style, marker, _ in pairs]
    handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.5))
    labels = [text for *_, text in pairs] + ["= 1 (equal)"]
    for r in rs:
        data = ratios(r)
        if not data:
            continue
        title = f"Relative Costs, batched, R = {r}: {7 ** r:,} leaf products, our wgmma vs cuBLAS ({measure} ratio)"
        fig, ax = p.figure(title, f"{measure} ratio")
        ax.set_title("")  # the title goes above the legend (suptitle), the legend between it and the axes
        fig.suptitle(f"{title}, Device={p.device}", fontsize=14, fontweight="bold", y=0.985)
        ax.set_xlabel("leaf size", fontsize=12)
        ax.set_yscale("log", base=2)
        xs = set()
        for (top, bottom, c, style, marker, _), x, y in data:
            xs.update(x)
            ax.plot(x, y, color=c, linestyle=style, linewidth=2, marker=marker)
            if top == "OWN wgmma":  # label our two ratios, the second below its point
                dy = 6 if bottom == "CUBLAS batched" else -14
                for n, v in zip(x, y):
                    ax.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(4, dy), fontsize=10, color=c)
        xs = sorted(xs)
        ax.plot(xs, [1.0] * len(xs), color="black", linestyle=":", linewidth=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_ylim(lowest / 1.05, highest * 1.05)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, 0.955))
        fig.tight_layout(rect=(0, 0, 1, 0.915))
        p.save_figure(fig, f"Relative Costs batched R{r} wgmma vs cuBLAS")  # Relative_Costs_batched_R<r>_wgmma_vs_cuBLAS.png


def plot_all(data_dir: str, out_dir: str, show: bool = False):
    if not show:
        matplotlib.use("Agg")
    os.makedirs(out_dir, exist_ok=True)
    files = {os.path.splitext(f)[0]: os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(".csv")}
    loaded = {name: read_result(path) for name, path in files.items()}
    device = next((m.get("device", "") for m, _ in loaded.values()), "")
    p = Plotter(out_dir, device, show)
    for name, (meta, rows) in loaded.items():
        p.device = meta.get("device", device)
        if meta.get("kind") == "series":
            if name == "relative_costs":
                # absolute times / energies: "Costs" (runs before 2026-10-26 titled them "Relative
                # Costs"); the ratio to cuBLAS is the "Relative Costs" graph
                meta = dict(meta, title="Costs")
            plot_series(p, meta, rows)
            if name == "batched_costs":  # Relative_Costs_batched_R<r>_wgmma_vs_cuBLAS.png
                plot_batched(p, meta, rows)
            if name == "relative_costs":
                plot_relative(p, meta, rows, "CUBLAS", "Relative Costs")
                own = own_series_name(series_of(rows))
                if own:  # e.g. Relative_Costs_vs_our_wgmma.png, Relative_Costs_our_wgmma_vs_cuBLAS.png
                    plot_relative(p, meta, rows, own, "Relative Costs vs our " + own[len("OWN "):])
                    plot_own_vs_cublas(p, meta, rows, own)
                if "CUBLAS best" in series_of(rows):  # Relative_Costs_vs_cuBLAS_best.png
                    plot_relative(p, meta, rows, "CUBLAS best", "Relative Costs vs cuBLAS best")
        elif name == "initialization_time":
            plot_min_max_avg_per_r(p, meta, rows)
    if "cublas" in loaded and "caps_cublas" in loaded:
        p.device = loaded["cublas"][0].get("device", device)
        plot_benchmark(p, loaded["cublas"][0], loaded["cublas"][1], loaded["caps_cublas"][1])
    return p.written


def gpu_folder(device: str) -> str:
    """Folder name of a GPU: "NVIDIA GeForce RTX 4090" -> 4090, "NVIDIA H100 80GB HBM3" -> h100,
    "NVIDIA A100-SXM4-80GB" -> a100-sxm, "NVIDIA GeForce RTX 3080 Ti Laptop GPU" -> 3080ti-laptop."""
    name = device.strip()
    m = re.search(r"RTX\s*(?:PRO\s*)?(\d{4})(\s*Ti)?", name, re.I)
    if m:
        folder = m.group(1) + ("ti" if m.group(2) else "")
        return folder + ("-laptop" if re.search(r"laptop", name, re.I) else "")
    m = re.search(r"\b([ABHL]\d{2,3})\b(?:-?(SXM|PCIE))?", name, re.I)
    if m:
        return m.group(1).lower() + (f"-{m.group(2).lower()}" if m.group(2) else "")
    slug = re.sub(r"^(nvidia\s+)?(geforce\s+)?", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-") or "unknown-gpu"


def store(data_dir: str, root: str, gpu: str | None = None, show: bool = False):
    """Copy the run in data_dir to <root>/<gpu>/<run>/data and draw it into <root>/<gpu>/<run>.
    Returns the run folder."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        raise RuntimeError(f"no result files in {data_dir}")
    metas = {f: read_result(os.path.join(data_dir, f))[0] for f in files}
    devices = {m.get("device", "") for m in metas.values()}
    runs = {m.get("run", "") for m in metas.values()}
    if len(devices) > 1 or len(runs) > 1:
        raise RuntimeError(f"{data_dir} mixes runs: devices {sorted(devices)}, runs {sorted(runs)}; "
                           "remove the old files and run the benchmark again")
    device, run = devices.pop(), runs.pop()
    if not run:  # files from before the run key: use the newest modification time
        import datetime
        newest = max(os.path.getmtime(os.path.join(data_dir, f)) for f in files)
        run = datetime.datetime.fromtimestamp(newest).strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(root, gpu or gpu_folder(device), run)
    if os.path.exists(os.path.join(folder, "data")):
        raise RuntimeError(f"{folder} exists already: this run was stored before")
    os.makedirs(os.path.join(folder, "data"))
    for f in files:
        shutil.copy2(os.path.join(data_dir, f), os.path.join(folder, "data", f))
    plot_all(os.path.join(folder, "data"), folder, show)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(REPO, "plots", "data"))
    parser.add_argument("--out", default=os.path.join(REPO, "plots"))
    parser.add_argument("--show", action="store_true", help="also show every graph in a window")
    parser.add_argument("--store", nargs="?", const=REPO, metavar="ROOT",
                        help="keep the run: copy it to ROOT/<gpu>/<run> (ROOT default: the repository) and draw it there")
    parser.add_argument("--gpu", help="GPU folder name for --store (default: from the device name)")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.data):
        print(f"no results in {args.data}: run ./bin/test_gemm_benchmark first")
        return 1
    if args.store:
        try:
            folder = store(args.data, args.store, args.gpu, args.show)
        except RuntimeError as e:
            print(f"error: {e}")
            return 1
        print(f"stored in {folder}")
        return 0
    for path in plot_all(args.data, args.out, args.show):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
