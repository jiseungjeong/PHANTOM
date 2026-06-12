#!/usr/bin/env python3
"""Calibration curve (purple, star at m=1) — WITH y-axis numbers.

Reads reports/calibration.json (shoe4_mean: m -> macro-F1).
m=0 is shown as an isolated dot (no line 0->1, since 1 demo is the recommended point);
m=1..5 connected. Star marks m=1.

Output: figures/calibration_curve.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "reports" / "calibration.json").read_text())
curve = {int(k): v for k, v in d["shoe4_mean"].items()}

ms = sorted(curve)                       # 0..5
PURPLE = "#5b3fd6"
GOLD = "#f4a836"

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")

# m=1..5 line
xs = [m for m in ms if m >= 1]
ys = [curve[m] for m in xs]
ax.plot(xs, ys, "-o", color=PURPLE, lw=2.6, ms=9, zorder=3)

# m=0 isolated dot (no connecting line to m=1)
ax.plot(0, curve[0], "o", color=PURPLE, ms=9, zorder=3)

# star at m=1 (recommended)
ax.plot(1, curve[1], marker="*", color=GOLD, ms=24, mec=PURPLE, mew=1.4, zorder=5)

ax.set_xlabel("Calibration Demos per Action (m)", fontsize=14, fontweight="bold")
ax.set_ylabel("LOSO Macro-F1", fontsize=14, fontweight="bold")
ax.set_title("LOSO Macro-F1 (Unseen Person)", fontsize=16, fontweight="bold", color=PURPLE, pad=14)

ax.set_xticks(ms)
ax.set_xticklabels(ms, fontsize=12)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([f"{t:.1f}" for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=12)

ax.grid(axis="y", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out = HERE / "figures" / "calibration_curve.png"
plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print(f"saved -> {out}")
