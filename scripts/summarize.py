"""Median (and min, max) of every phase per graph, version and thread count.

    python3 scripts/summarize.py            reads results/timings.txt
                                            writes results/timings_median.csv
"""
import csv
import statistics
from collections import defaultdict

PHASES = ["P", "mirror", "attr", "build", "mass", "walk", "grad", "direct"]

runs = defaultdict(list)                 # (graph, version, p) -> list of runs
for line in open("results/timings.txt"):
    if not line.startswith("TIME"):
        continue
    f = dict(kv.split("=") for kv in line.split()[1:])
    runs[(f["graph"], f["version"], int(f["p"]))].append(f)

rows = []
for (graph, version, p), rs in sorted(runs.items()):
    row = {"graph": graph, "version": version, "p": p, "runs": len(rs),
           "n": rs[0]["n"], "nnz": rs[0]["nnz"], "L": rs[0]["L"]}
    for ph in PHASES:
        x = [float(r[ph]) for r in rs]
        row[ph] = statistics.median(x)
        row[ph + "_min"], row[ph + "_max"] = min(x), max(x)
    rows.append(row)

with open("results/timings_median.csv", "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# short table on the screen: median seconds per phase
print(f"{'graph':15} {'ver':5} {'p':>3} {'runs':>4} " + " ".join(f"{ph:>8}" for ph in PHASES))
for r in rows:
    print(f"{r['graph']:15} {r['version']:5} {r['p']:3} {r['runs']:4} "
          + " ".join(f"{r[ph]:8.4f}" for ph in PHASES))
print("wrote results/timings_median.csv")
