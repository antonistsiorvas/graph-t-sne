"""The embedding after iterating (optional part of the README).

    python3 scripts/plot_embedding.py <graph>     reads results/embed_<graph>.txt
                                                  -> report/figs/embed_<graph>.png

Left: the points only, as t-SNE placed them (it was never told the labels).
Right: the same points colored by label. With dozens of classes no palette can
give each a distinct, nameable color, so the colors only separate neighboring
islands (a hue wheel over the label number). In com-Amazon label 0 means
"in no top community" (95% of the nodes): those are drawn light gray, underneath.
In k49 label 0 is a class like the others (classes 0-48).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

graph = sys.argv[1]
x, y, lab = [], [], []
for line in open(f"results/embed_{graph}.txt"):
    a, b, c = line.split()
    x.append(float(a)); y.append(float(b)); lab.append(int(c))

zero_is_none = graph == "com-Amazon"            # README: Amazon's 0 = no community
classes = sorted(set(lab) - {0}) if zero_is_none else sorted(set(lab))
hue = {c: k / max(1, len(classes)) for k, c in enumerate(classes)}
size = 0.3 if len(x) > 100_000 else 0.8

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.5))
left.scatter(x, y, s=size, c="#0b0b0b", linewidths=0)
left.set_title("points only")

unl = [i for i, c in enumerate(lab) if zero_is_none and c == 0]
lbl = [i for i, c in enumerate(lab) if not (zero_is_none and c == 0)]
if unl:
    right.scatter([x[i] for i in unl], [y[i] for i in unl], s=size, c="#d6d5ce", linewidths=0)
right.scatter([x[i] for i in lbl], [y[i] for i in lbl], s=size * 2,
              c=[hue[lab[i]] for i in lbl], cmap="hsv", vmin=0, vmax=1, linewidths=0)
right.set_title(f"colored by label ({len(classes)} classes"
                + (f"; gray: no label, {len(unl):,} points" if unl else "") + ")")

for ax in (left, right):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle(f"{graph}: t-SNE embedding (n = {len(x):,})", fontsize=10)
os.makedirs("report/figs", exist_ok=True)
fig.savefig(f"report/figs/embed_{graph}.png", dpi=200, bbox_inches="tight")
print(f"wrote report/figs/embed_{graph}.png")
