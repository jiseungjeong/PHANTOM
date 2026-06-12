#!/usr/bin/env python3
"""Per-class F1, calib 0 AND 1, for BoxerSense-features vs PHANTOM — one chart.

Punch classes (6): 4 bars each — BoxerSense calib0 / calib1, PHANTOM calib0 / calib1.
Defense classes (4): 2 bars — PHANTOM calib0 / calib1 only. The BoxerSense slots are
LEFT EMPTY on purpose: BoxerSense is a punch-only method; drawing a defense bar for it
would falsely imply it can track defense. The empty slots ARE the message.

Honest framing:
  - Gray = BoxerSense feature (126) + their best model (SVM). Purple = PHANTOM (1-NN).
    The MODELS differ, so the punch comparison is partly a model effect, NOT a clean
    feature test -> we LABEL this on the chart ("model differs — may be unfair") and
    point to the fair test (phase35_fair_feature: features tie under one model).
  - Gold marker on the 3 REAR punches = BoxerSense PAPER PI F1 (SVM, wrist, 10 ppl,
    3-class) — shown only where the paper actually reports it.
  - Our real edge is COVERAGE (defense, only us) + a model fit for on-device calibration.
  - All numbers: file-separated LOSO, per-class F1, pooled over folds. calib 1 = 1 demo/action.

Output: figures/coverage_calib.png
Run:    python3 analysis/five_subject_active_window_classifier/draw_coverage_calib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluation import load_dataset, split_cohort
from phase45_calibration_extra import run_curve, build_index
import phase3_models as p3

PUNCHES = ["Lead Jab", "Lead Hook", "Lead Uppercut", "Rear Jab", "Rear Hook", "Rear Uppercut"]
DEFENSE = ["Left Slip", "Right Slip", "Left Roll", "Right Roll"]
ORDER = PUNCHES + DEFENSE

STATS = ["mean", "p50", "std", "min", "max", "p25", "p75"]
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
DEVICES = ["a", "i", "w"]

# BoxerSense paper, Table 4 (PI case, SVM) — per-class F1 for the 3 rear punches.
# Their main result; reported only for rear (config-1, wrist, 10 ppl, 3-class).
PAPER_REAR_F1 = {"Rear Jab": 0.93, "Rear Hook": 0.85, "Rear Uppercut": 0.93}


def boxer_cols(names):
    idx = {n: i for i, n in enumerate(names)}
    return [idx[f"{d}_{ax}_{st}"] for d in DEVICES for ax in AXES for st in STATS
            if f"{d}_{ax}_{st}" in idx]


def boxer_pipe():
    """BoxerSense feature (126) + their best model (SVM), our segmentation/data.
    NOTE: model differs from PHANTOM (SVC vs 1-NN) -> punch comparison is NOT a clean
    feature-only test (the gap is partly model). Flagged on the chart as 'may be unfair'."""
    return Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                     ("var", VarianceThreshold()), ("scale", MinMaxScaler()),
                     ("clf", SVC())])   # paper best (PI)


def per_class_at(make_pipe, X, y, subj, files, labels, m):
    """pooled per-class F1 at calibration level m (file-separated)."""
    subjects = sorted(set(subj.tolist()))
    by = build_index(y, subj, files)
    _, (yt, yp) = run_curve(make_pipe, X, y, subj, files, labels, subjects, by,
                            collect_preds_at=m)
    return {lab: float(f1_score(yt, yp, labels=[lab], average="macro", zero_division=0))
            for lab in labels}


CACHE = HERE / "reports" / "coverage_calib_values.json"


def compute_values():
    """Heavy step (4 calibration curves). Returns the per-class dicts."""
    import json
    Xa, ya, subja, filesa, feats, _ = load_dataset()
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)
    labels = sorted(set(y.tolist()))

    # PHANTOM = deployed pipeline (3 IMU, full feature set, tree-selection inside)
    ph0 = per_class_at(lambda: p3.knn(sel=True), X, y, subj, files, labels, 0)
    ph1 = per_class_at(lambda: p3.knn(sel=True), X, y, subj, files, labels, 1)

    # BoxerSense = their features (subset of columns) + their best model
    cols = boxer_cols(feats)
    Xb = X[:, cols]
    bx0 = per_class_at(boxer_pipe, Xb, y, subj, files, labels, 0)
    bx1 = per_class_at(boxer_pipe, Xb, y, subj, files, labels, 1)

    vals = {"ph0": ph0, "ph1": ph1, "bx0": bx0, "bx1": bx1}
    CACHE.write_text(json.dumps(vals, indent=2, ensure_ascii=False))
    print(f"computed + cached -> {CACHE}")
    return vals


def load_values():
    """Fast step: load cached values; recompute only if missing or --recompute."""
    import json
    if "--recompute" in sys.argv or not CACHE.exists():
        return compute_values()
    print(f"loaded cache <- {CACHE}  (pass --recompute to refresh)")
    return json.loads(CACHE.read_text())


def main():
    v = load_values()
    ph0, ph1, bx0, bx1 = v["ph0"], v["ph1"], v["bx0"], v["bx1"]

    # ── draw ──────────────────────────────────────────────────────────────────
    GRAY0, GRAY1 = "#cfcad8", "#8a8398"      # BoxerSense calib0 / calib1
    PUR0, PUR1 = "#b8a8d8", "#6b4ea0"        # PHANTOM calib0 / calib1
    BLUE = "#3a6ea5"

    fig, ax = plt.subplots(figsize=(14, 6.4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(ORDER))
    w = 0.2

    # shade defense region
    ax.axvspan(len(PUNCHES) - 0.5, len(ORDER) - 0.5, color="#eef2f8", zorder=0)

    GOLD = "#c9a84a"
    for i, lab in enumerate(ORDER):
        is_punch = lab in PUNCHES
        # BoxerSense bars only for punches
        if is_punch:
            ax.bar(x[i] - 1.5 * w, bx0[lab], w, color=GRAY0, zorder=3)
            ax.bar(x[i] - 0.5 * w, bx1[lab], w, color=GRAY1, zorder=3)
        # PHANTOM bars always
        ax.bar(x[i] + 0.5 * w, ph0[lab], w, color=PUR0, zorder=3)
        ax.bar(x[i] + 1.5 * w, ph1[lab], w, color=PUR1, zorder=3)
        # BoxerSense PAPER PI F1 — only on the 3 rear punches (where the paper reports it)
        if lab in PAPER_REAR_F1:
            pv = PAPER_REAR_F1[lab]
            ax.bar(x[i], pv, 4 * w, color="none", edgecolor=GOLD, lw=2.2,
                   hatch="///", zorder=2)
            ax.text(x[i] - 0.7 * w, pv + 0.022, f"paper {pv:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#9a7d2a", fontweight="bold", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in ORDER], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("per-class F1 (file-separated LOSO)", fontsize=12, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # legend (4 bars + paper marker)
    from matplotlib.patches import Patch
    handles = [
        Patch(color=GRAY0, label="BoxerSense (SVM) — no calib"),
        Patch(color=GRAY1, label="BoxerSense (SVM) — 1 demo"),
        Patch(color=PUR0, label="PHANTOM (1-NN) — no calib"),
        Patch(color=PUR1, label="PHANTOM (1-NN) — 1 demo"),
        Patch(facecolor="none", edgecolor=GOLD, hatch="///", label="BoxerSense paper, unseen · no calib"),
    ]
    ax.legend(handles=handles, fontsize=9.5, loc="upper center", ncol=5,
              framealpha=0.92, bbox_to_anchor=(0.5, 1.11))

    ax.set_title("Punches: comparable  ·  Defense: only PHANTOM tracks it",
                 fontsize=15, fontweight="bold", color="#3d2a66", pad=40)

    fig.text(0.5, -0.035,
             "⚠ Models differ (BoxerSense=SVM, PHANTOM=1-NN) → the punch gap is partly a MODEL effect, not a clean feature test "
             "(under one model the two feature sets tie ±0.01, see appendix).\n"
             "Gold hatch = BoxerSense paper, unseen (no calib) F1 (SVM, wrist, 10 ppl, 3-class) — shown only on rear punches, where the paper reports it. "
             "Our edge: defense coverage (only us) + a model fit for on-device calibration. File-separated LOSO.",
             ha="center", fontsize=8.4, color="#666", style="italic")

    plt.tight_layout()
    out = HERE / "figures" / "coverage_calib.png"
    plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")

    # print numbers for the outline
    print("\n# per-class F1 (calib0 / calib1)")
    print(f"{'class':16s} {'BX0':>6s} {'BX1':>6s} {'PH0':>6s} {'PH1':>6s}")
    for lab in ORDER:
        b0 = f"{bx0[lab]:.2f}" if lab in PUNCHES else "  -"
        b1 = f"{bx1[lab]:.2f}" if lab in PUNCHES else "  -"
        print(f"{lab:16s} {b0:>6s} {b1:>6s} {ph0[lab]:6.2f} {ph1[lab]:6.2f}")


if __name__ == "__main__":
    main()
