#!/usr/bin/env python3
"""Jump-cut verification visualization (read-only).

Segmentation crops actions starting AFTER the sync jump. This verifies, by eye,
that the jump-cut time sits at the correct boundary (the end of the initial
synchronization wobble).

Drawn with the same FUSED motion_envelope used for the actual jump-cut/crop
(gyro 62% + accel 38%, 3-sensor 72%max + 28%mean). To also show the pre-jump
region, envelope is computed with jump_cut_s=0 (full signal), then the real
jump_cut position is marked with a line. This is the same curve as in
segmentation_grid.png, so the two figures connect at a glance.

Each subplot = one (subject, action):
  - fused envelope (full, including jump, blue)
  - jump_cut_s position = red dashed line (left shaded = discarded jump region,
    right = the 5 action repetitions)
If the jump line is AFTER the initial large jump peak and BEFORE the regular
action peaks, it is OK.

Output: reports/jumpcut_grid.png  (excluded by .gitignore)
Run: python3 analysis/five_subject_active_window_classifier/viz_jumpcut.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
OUT_DIR = THIS.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"] = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp

SINGLE_LABELS = ["Lead Jab", "Lead Hook", "Lead Uppercut",
                 "Rear Jab", "Rear Hook", "Rear Uppercut",
                 "Left Slip", "Right Slip", "Left Roll", "Right Roll"]
COLS = ["SEUNG", "JIHO", "JUHO", "SHIM"]


def find_file(subject: str, label: str) -> Path | None:
    folder = DATA_DIR / f"{subject.lower()}_data"
    cands = sorted(p for p in folder.glob(f"{subject}_{label}_*.csv") if "_Watch_" not in p.name)
    return cands[0] if cands else None


def full_envelope(data):
    """Fused envelope over the full signal including the jump. Computed with jump_cut=0."""
    times, env, dur, _ = bp.motion_envelope(data, 0.0)
    return (times, env) if len(env) else (None, None)


def main() -> None:
    nrows, ncols = len(SINGLE_LABELS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 1.8 * nrows), squeeze=False)
    print("# Jump-cut verification visualization")
    bad = []
    for r, label in enumerate(SINGLE_LABELS):
        for c, subj in enumerate(COLS):
            ax = axes[r][c]
            path = find_file(subj, label)
            if path is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_xticks([]); ax.set_yticks([]); continue
            data = rbc.read_csv_sources(path)
            _, meta = rbc.extract_features(path)
            jump = float(meta.get("jump_cut_s", 0.0))
            t, env = full_envelope(data)
            if t is None:
                ax.text(0.5, 0.5, "no env", ha="center", va="center"); ax.set_xticks([]); ax.set_yticks([]); continue
            ax.plot(t, env, color="#1f4e79", lw=1.0)                 # fused envelope (full, including jump)
            ax.axvline(jump, color="red", lw=1.6, ls="--")           # jump-cut boundary
            ax.axvspan(0, jump, color="red", alpha=0.08)             # discarded region
            ax.set_xticks([]); ax.set_yticks([])
            tot = t[-1] if len(t) else 1.0
            ok = not (jump <= 0.05 or jump > 0.6 * tot)
            ax.set_title(f"{subj} | {label}  cut={jump:.1f}s" + ("" if ok else " (!)"),
                         fontsize=7, color="black" if ok else "red")
            if not ok:
                bad.append((subj, label, jump, tot))
    fig.suptitle("Jump-cut verification — FUSED envelope (gyro62%+acc38%, 3-sensor); "
                 "red dashed = jump_cut (left shaded = discarded)",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT_DIR / "jumpcut_grid.png"
    fig.savefig(out, dpi=110)
    print(f"  saved -> {out}")
    if bad:
        print(f"  (!) suspect cuts ({len(bad)}): " + ", ".join(f"{s}/{l}(cut={j:.1f}/{t:.1f}s)" for s,l,j,t in bad))
    else:
        print("  OK: all cuts within reasonable range (0 < cut < 60% of length)")


if __name__ == "__main__":
    main()
