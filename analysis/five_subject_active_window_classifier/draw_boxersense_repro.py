#!/usr/bin/env python3
"""Draw the BoxerSense reproduction figure (3 tiers + paper reference).
Reads reports/boxersense_repro.json. Output: figures/boxersense_repro.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "reports" / "boxersense_repro.json").read_text())
t = d["tiers"]

PURPLE_OFF = "#b8a8d8"   # calib OFF (lighter)
PURPLE_ON = "#6b4ea0"    # calib ON / 1-demo (saturated)
PAPER = "#c9a84a"        # muted gold for the paper reference
RED = "#c0392b"

# three reproduction tiers, each with calib OFF (m0) and 1-demo (m1)
tiers_plot = [
    ("Repro R1\nwrist · 3 rear punch\n(most faithful)", t["R1_wrist_3rear"]),
    ("Repro R2\nwrist · 6 punch", t["R2_wrist_6punch"]),
    ("Repro R3\n3 IMU · 6 punch\n(our extension)", t["R3_3imu_6punch"]),
]

fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.patch.set_facecolor("white")

x = np.arange(len(tiers_plot))
w = 0.36
off_vals = [tt["calib_m0"] for _, tt in tiers_plot]
on_vals = [tt["calib_m1"] for _, tt in tiers_plot]

b_off = ax.bar(x - w / 2, off_vals, w, color=PURPLE_OFF, zorder=3, label="no calibration (m=0)")
b_on = ax.bar(x + w / 2, on_vals, w, color=PURPLE_ON, zorder=3, label="1-demo calibration (m=1)")

for i, (_, tt) in enumerate(tiers_plot):
    ax.text(x[i] - w / 2, off_vals[i] + 0.012, f"{off_vals[i]:.2f}", ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color="#6a6a7a", zorder=5)
    ax.text(x[i] + w / 2, on_vals[i] + 0.012, f"{on_vals[i]:.2f}", ha="center", va="bottom",
            fontsize=13.5, fontweight="bold", color=PURPLE_ON, zorder=5)
    # +delta arrow per tier
    ax.annotate("", xy=(x[i] + w / 2, on_vals[i]), xytext=(x[i] - w / 2, off_vals[i]),
                arrowprops=dict(arrowstyle="->", color="#8a7ab0", lw=1.3), zorder=4)
    ax.text(x[i] - w / 2 - 0.04, (off_vals[i] + on_vals[i]) / 2, f"+{on_vals[i]-off_vals[i]:.2f}",
            ha="right", va="center", fontsize=9.5, color="#8a7ab0", fontweight="bold")
    ax.text(x[i] + w / 2, on_vals[i] - 0.035, tt["best_model"], ha="center", va="top",
            fontsize=8.5, color="white", fontweight="bold", zorder=6)

# paper reference line (PI 0.91, their cleaner data)
paper_v = d["paper"]["R1_wrist_3rear"]["PI_macro_f1"]
ax.axhline(paper_v, ls="--", color=PAPER, lw=2, zorder=2)
ax.text(0.95, paper_v + 0.014, f"BoxerSense paper (PI {paper_v:.2f}) — cleaner data, 10 ppl",
        ha="center", fontsize=9.5, color="#9a7d2a", fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels([lbl for lbl, _ in tiers_plot], fontsize=11)
ax.set_ylabel("LOSO Macro-F1 (unseen person)", fontsize=12.5, fontweight="bold")
ax.set_yticks([0, .2, .4, .6, .8, 1.0])
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10.5, loc="upper left", framealpha=0.9)

ax.set_title("BoxerSense Reproduced on Our Data — Calibration Closes Much of the Gap",
             fontsize=15, fontweight="bold", color=PURPLE_ON, pad=16)

# honest footnote
fig.text(0.5, -0.02,
         "Faithful: 7-stat×6-axis features, sklearn-default SVM/RF/KNN, leave-one-person-out. "
         "1-demo = one calibration file per action (file-separated, no leakage).\n"
         "Gap to paper: their data rests 4s between punches & excludes non-punch motion (clean), 10 subjects; "
         "ours is natural continuous shadowboxing, 5 subjects (harder). "
         "Not reproducible: Butterworth dual-threshold segmentation, upper-back sensor.",
         ha="center", fontsize=8.6, color="#666", style="italic")

plt.tight_layout()
out = HERE / "figures" / "boxersense_repro.png"
plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print(f"saved -> {out}")
