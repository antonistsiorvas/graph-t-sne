"""Prediction against measurement for coPapersDBLP.

    python3 scripts/compare.py   -> results/prediction_vs_measured_coPapersDBLP.md

Reads the prediction (results/prediction_coPapersDBLP.csv, committed before the
run) and the measured medians (results/timings_median.csv).
"""
import csv

G = "coPapersDBLP"
PHASES = ["P", "mirror", "attr", "build", "mass", "walk", "grad"]
TOTAL = ["P", "attr", "build", "mass", "walk", "grad"]
NAMES = {"P": "P (incl. mirror)", "mirror": "  of which mirror", "attr": "attraction",
         "build": "tree build", "mass": "centers of mass", "walk": "walk", "grad": "gradient"}

pred = {(r["version"], int(r["p"])): {k: float(r[k]) for k in PHASES}
        for r in csv.DictReader(open(f"results/prediction_{G}.csv"))}
meas = {(r["version"], int(r["p"])): {k: float(r[k]) for k in PHASES}
        for r in csv.DictReader(open("results/timings_median.csv")) if r["graph"] == G}
for d in list(pred.values()) + list(meas.values()):
    d["total"] = sum(d[k] for k in TOTAL)

def ms(x):
    return f"{x * 1e3:,.0f}" if x >= 0.01 else f"{x * 1e3:.1f}"

lines = [f"# {G}: prediction against measurement (ms, medians of 5 runs)", "",
         "Prediction: `results/prediction_" + G + ".md` (committed before the run). "
         "Measured: `results/timings_median.csv`. Error = measured / predicted - 1.", ""]
for v, p in [("seq", 1), ("omp", 12), ("cilk", 12), ("pth", 12)]:
    lines += [f"## {v}, p = {p}", "",
              "| phase | predicted | measured | error |", "|---|---|---|---|"]
    for ph in PHASES + ["total"]:
        a, b = pred[(v, p)][ph], meas[(v, p)][ph]
        lines.append(f"| {NAMES.get(ph, ph)} | {ms(a)} | {ms(b)} | {b / a - 1:+.0%} |")
    lines.append("")
open(f"results/prediction_vs_measured_{G}.md", "w").write("\n".join(lines))
print("\n".join(lines))
