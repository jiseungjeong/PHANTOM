#!/usr/bin/env python3
"""On-device calibration cost per model (single panel, clean).

Just the calibration-cost bars from the old why_knn left panel — no 'append' /
'full refit' annotations, no 'KNN = append' subtitle. KNN highlighted (it only
appends a vector; others need a full refit).

Output: figures/calib_cost.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#2a9d52"
GRAY = "#9aa0c0"

models = ["KNN\n(1-NN)", "SVC\n(RBF)", "ExtraTrees"]
calib_ms = [0.07, 26.0, 843.0]
colors = [GREEN, GRAY, GRAY]

fig, ax = plt.subplots(figsize=(7.5, 6))
fig.patch.set_facecolor("white")

bars = ax.bar(models, calib_ms, color=colors, width=0.62, zorder=3)
ax.set_yscale("log")
ax.set_ylabel("Calibration cost / sample (ms, log)", fontsize=13, fontweight="bold")
ax.set_title("On-device calibration cost", fontsize=15, fontweight="bold", pad=14)

for b, v in zip(bars, calib_ms):
    label = f"{v:.2f} ms" if v < 1 else f"{v:.0f} ms"
    ax.text(b.get_x() + b.get_width() / 2, v * 1.5, label,
            ha="center", fontsize=13, fontweight="bold", color="#222")

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.tick_params(axis="x", labelsize=12)

plt.tight_layout()
out = "figures/calib_cost.png"
plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print(f"saved -> {out}")
