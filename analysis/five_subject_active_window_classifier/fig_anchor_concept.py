"""Presentation figure: anchor-first 2-stage combo segmentation, on REAL combo data.

Stage 1 (strikes) are placed first on their dominant-sensor peaks (Lead*->Watch,
Rear*->iPhone); Stage 2 places the defense on an AirPods peak in the interval the
label position implies. Drawn on one real 3-action combo with sensor overlays.

Run: python3 analysis/five_subject_active_window_classifier/fig_anchor_concept.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase6_combo_probe as P

THIS = Path(__file__).resolve()

# one clear 3-action combo with a defense between/edge: Right Roll-Rear Hook-Lead Hook
TARGET = ("JUHO", "Right Roll-Rear Hook-Lead Hook")


def main() -> None:
    rows = P.collect()
    pick = None
    for r in rows:
        if (r[0], r[1]) == TARGET:
            pick = r; break
    if pick is None:
        print("target combo not found"); return
    subj, label, n, times, env, peaks, fname, data, jump = pick
    windows, cut = P.anchor_segment(data, label, jump)
    t, e, dur, _ = P.bp.motion_envelope(data, 0.0)
    src = P.per_source_envelopes(data)
    actions = label.split("-")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(t, e, color="#1f4e79", lw=1.6, label="combined", zorder=3)
    for sname, c in [("Watch", "#1b9e77"), ("iPhone", "#d95f02"), ("AirPods", "#7570b3")]:
        s = src.get(sname)
        if s is not None and len(s) == len(t):
            ax.plot(t, s, color=c, lw=1.0, alpha=0.55, label=sname)
    ymax = max(e.max(), 1e-6)
    if cut > 0:
        ax.axvspan(0, cut, color="#ff5555", alpha=0.10)
        ax.axvline(cut, color="#d11", ls="--", lw=1.0)
        ax.text(cut / 2, ymax * 0.5, "jump\ncut", ha="center", va="center",
                fontsize=8, color="#a00")

    cols = {"strike": "#9ecae1", "defense": "#a1d99b"}
    # stagger label heights so adjacent windows' text doesn't overlap
    label_y = [1.16, 1.02, 1.16, 1.02]
    for k, (act, s, en, _conf) in enumerate(windows):
        kind = "defense" if act in P.DEFENSE else "strike"
        ax.axvspan(s, s + P.WIN_S, color=cols[kind], alpha=0.45, zorder=1)
        stage = "(1) strike anchor" if kind == "strike" else "(2) defense -> AirPods"
        sensor = P.ACTION_SENSOR.get(act) or "combined"
        yy = label_y[k % len(label_y)] * ymax
        cx = s + P.WIN_S / 2
        ax.annotate(f"{act}\n[{sensor}]\n{stage}", xy=(cx, ymax * 0.55),
                    xytext=(cx, yy), ha="center", va="bottom", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))

    ax.set_title("Anchor-first combo segmentation (real data: %s | %s)\n"
                 "①strikes on dominant-sensor peaks  →  ②defense on AirPods peak in-between"
                 % (subj, label), fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.set_ylim(0, ymax * 1.32)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = THIS.parent / "reports" / "fig_anchor_concept.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out.name}  (windows={windows})")


if __name__ == "__main__":
    main()
