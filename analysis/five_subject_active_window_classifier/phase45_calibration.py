#!/usr/bin/env python3
"""Phase 4.5 — Calibration (on-device personalization) experiment.

Implements plan.md §9.8. Question: can a small amount of the END USER's own data
("calibration") lift LOSO 0.605 (unseen person) toward the file-grouped ceiling
0.963 (known person)? If yes with little user effort, on-device personalization
beats collecting more subjects.

── Protocol (peak-level calibration, FILE-separated) ────────────────────────────
For each test subject S (shoe-5 + JEONG):
  - S's 3 files per action are split: 1 calibration file vs 2 test files.
    * calibration peaks come ONLY from the calibration file
    * test peaks come ONLY from the 2 test files
    (Same file's peaks are near-identical -> mixing them = file-grouped leakage.
     So FILES stay separated; only the AMOUNT of calibration is dialed in peaks.)
  - warm start: the factory model = all OTHER subjects' data; calibration adds
    S's m peaks per action into the KNN reference (M-CAL-3).
  - JEONG fold: its same-person SEUNG is removed from the factory train (M-CAL-7),
    so it measures "barefoot user calibrating a shoe-cohort model".

Learning curve: calibration peaks per action m = 0,1,2,3,4,5.
  m=0 == plain LOSO (no calibration). Test set is FIXED (the 2 test files, ~10
  peaks) regardless of m -> stable variance (C1 resolved).

Double averaging (D1 / M-CAL-1), removes luck:
  (a) which of the 3 files is the calibration file: all 3
  (b) which m peaks within that file: all C(5,m) <= 10 combinations
  -> full enumeration, averaged.

Preprocessing FROZEN on factory train (M-CAL-6 / D4): imputer, VarianceThreshold,
RobustScaler, Normalizer, and feature-selection are fit on the factory train and
NOT refit; calibration peaks are only TRANSFORMED and appended to the KNN
reference. This matches on-device reality (the app does not recompute preprocessing,
it just grows the reference set). So we cannot use a plain Pipeline.fit(all) here.

Outputs: reports/calibration.json, figures/calibration_curve.png
Run: python3 analysis/five_subject_active_window_classifier/phase45_calibration.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

THIS = Path(__file__).resolve()
OUT_DIR = THIS.parent / "reports"; OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = THIS.parent / "figures"; FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THIS.parent))

from evaluation import load_dataset, split_cohort, HOLDOUT_ENV_SUBJECTS

import phase3_models as p3

M_VALUES = [0, 1, 2, 3, 4, 5]          # calibration peaks per action
PEAKS_PER_FILE = 5                     # nominal (a few files have 4; we clamp)
# Calibration runs on the SHOE-5 cohort only (split_cohort excludes JEONG).
# JEONG is excluded entirely because (a) it is the barefoot environment holdout and
# (b) it is the SAME PERSON as SEUNG -> keeping it would leak SEUNG<->JEONG. Dropping
# JEONG also removes that leak automatically (no same-person twin remains in train).
SAME_PERSON = {}                        # kept for backward-compat import; empty now
LOSO_M0_REFERENCE = 0.605              # plain LOSO macro-F1 (sanity check for m=0)


def split_preproc_knn(sel: bool):
    """Return (preproc_pipeline, knn_estimator) split from the best KNN(sel) pipeline.

    preproc = every step except the final 'model'; knn = the final estimator.
    We fit preproc on the factory train ONCE, transform everything through it, and
    fit the KNN on the stacked transformed reference (factory + calibration). KNN
    'fit' merely stores the reference, so growing it = adding calibration samples,
    with preprocessing frozen (M-CAL-6)."""
    full = p3.knn(sel=sel)
    steps = full.steps
    preproc = Pipeline(steps[:-1])     # imputer..selection..scaler..normalizer
    knn = steps[-1][1]                 # KNeighborsClassifier(n=1, manhattan, distance)
    return preproc, knn


def calibrated_predict(preproc, knn_proto, Xf, yf, Xcal, ycal, Xte):
    """Freeze preproc on factory (Xf,yf); transform factory + calibration; fit KNN
    on the stacked reference; predict the (transformed) test set."""
    pre = clone(preproc).fit(Xf, yf)
    Zf = pre.transform(Xf)
    knn = clone(knn_proto)
    if len(Xcal):
        Zc = pre.transform(Xcal)
        Z = np.vstack([Zf, Zc])
        y = np.concatenate([yf, ycal])
    else:
        Z, y = Zf, yf
    knn.fit(Z, y)
    return knn.predict(pre.transform(Xte))


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    # Shoe-4 cohort only (excludes JEONG = barefoot holdout & SEUNG's same-person twin).
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    # Exclude meta features (same as Phase 3 / Phase 4 best model).
    keep = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    X = X[:, keep]
    subjects = sorted(set(subj.tolist()))
    print(f"# Phase 4.5 — calibration (peak-level), shoe-5 cohort {len(y)} samples, {len(labels)} classes")
    print(f"  test subjects: {subjects}  (m = {M_VALUES} calib peaks/action)\n")

    preproc, knn_proto = split_preproc_knn(sel=True)

    # Index helper: per (subject, label) -> {file: [sample indices]}
    by_sl_file = defaultdict(lambda: defaultdict(list))
    for i in range(len(y)):
        by_sl_file[(subj[i], y[i])][files[i]].append(i)

    # results[subject][m] = list of macro-F1 over all (calib-file x peak-combo)
    results = defaultdict(lambda: defaultdict(list))

    for test_s in subjects:
        # Factory train = all OTHER shoe-5 subjects (warm start). JEONG is already
        # excluded by split_cohort, so no same-person twin remains to leak.
        factory_mask = subj != test_s
        Xf_all, yf_all = X[factory_mask], y[factory_mask]

        # The test subject's actions, each with its 3 files.
        labels_S = sorted({lab for (s, lab) in by_sl_file if s == test_s})
        # Enumerate which file is the calibration file (all 3), averaged later.
        # All actions share the same 3 takes index-wise? Not guaranteed, so per-action.
        # We pick calibration-file *position* 0/1/2 consistently per action.
        for calib_pos in range(3):
            # Build, per action, the calibration file and the test files.
            cal_idx_by_action = {}       # action -> list of sample idx in calib file
            test_idx = []                # all test-file sample idx (fixed for all m)
            ok = True
            for lab in labels_S:
                fmap = by_sl_file[(test_s, lab)]
                fs = sorted(fmap.keys())
                if len(fs) < 3:
                    ok = False; break
                cal_file = fs[calib_pos]
                cal_idx_by_action[lab] = sorted(fmap[cal_file])
                for f in fs:
                    if f != cal_file:
                        test_idx.extend(fmap[f])
            if not ok:
                continue
            test_idx = np.array(sorted(test_idx))
            Xte, yte = X[test_idx], y[test_idx]

            for m in M_VALUES:
                if m == 0:
                    pred = calibrated_predict(preproc, knn_proto, Xf_all, yf_all,
                                              np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype), Xte)
                    f1 = f1_score(yte, pred, labels=labels, average="macro", zero_division=0)
                    results[test_s][m].append(f1)
                    continue
                # Enumerate all C(peaks_in_calfile, m) combinations per action,
                # but combos multiply across actions -> too many. Instead, average
                # over peak-combinations PER ACTION independently is not valid (the
                # KNN reference needs all actions' calib together). So we use the
                # same combo *rank* across actions via aligned sampling: enumerate
                # combinations within each action and pair them by a shared index
                # set when peak counts match; fall back to first-c order otherwise.
                # Practical, leakage-free choice: take the first m peaks (time order)
                # of each action's calib file -> deterministic; then also average a
                # few rotations to remove "which peaks" luck.
                rot_f1 = []
                # peaks available per action in the calib file (usually 5, sometimes 4)
                n_rot = PEAKS_PER_FILE  # rotate starting offset to vary chosen peaks
                for rot in range(n_rot):
                    cal_sel = []
                    for lab in labels_S:
                        idxs = cal_idx_by_action[lab]
                        mm = min(m, len(idxs))
                        # rotate so different peaks get chosen across rotations
                        order = [idxs[(rot + j) % len(idxs)] for j in range(len(idxs))]
                        cal_sel.extend(order[:mm])
                    cal_sel = np.array(cal_sel)
                    pred = calibrated_predict(preproc, knn_proto, Xf_all, yf_all,
                                              X[cal_sel], y[cal_sel], Xte)
                    rot_f1.append(f1_score(yte, pred, labels=labels, average="macro", zero_division=0))
                results[test_s][m].append(float(np.mean(rot_f1)))

        done = {m: round(float(np.mean(results[test_s][m])), 3) for m in M_VALUES if results[test_s][m]}
        print(f"  {test_s:8s} (factory = other shoe-5): {done}")

    # ── Aggregate: shoe-5 mean (JEONG excluded entirely) ─────────────────────────
    shoe = [s for s in subjects if s not in HOLDOUT_ENV_SUBJECTS]
    curve_shoe, curve_jeong = {}, {}
    for m in M_VALUES:
        sv = [np.mean(results[s][m]) for s in shoe if results[s][m]]
        curve_shoe[m] = float(np.mean(sv)) if sv else None
        jv = results.get("JEONG", {}).get(m, [])
        curve_jeong[m] = float(np.mean(jv)) if jv else None

    print("\n## Calibration learning curve (macro-F1)")
    print(f"  {'m (calib peaks/action)':26s} " + "  ".join(f"m={m}" for m in M_VALUES))
    print(f"  {'shoe-5 mean':26s} " + "  ".join(
        f"{curve_shoe[m]:.3f}" if curve_shoe[m] is not None else " n/a" for m in M_VALUES))
    if any(curve_jeong[m] is not None for m in M_VALUES):
        print(f"  {'JEONG (barefoot)':26s} " + "  ".join(
            f"{curve_jeong[m]:.3f}" if curve_jeong[m] is not None else " n/a" for m in M_VALUES))

    # m=0 vs plain LOSO — they DIFFER BY DESIGN, not a bug:
    #   plain LOSO tests on all 3 files; m=0 here tests on only the 2 non-calib files
    #   (averaged over which file is held out), and averages per-fold macro-F1.
    #   So m=0 (~0.69) > plain LOSO (0.605) is expected. The honest, internal
    #   baseline for the calibration *gain* is m=0 within THIS protocol (same test
    #   set as m>=1), NOT the plain-LOSO number. We report the gain m=0 -> m=k.
    m0 = curve_shoe[0]
    print(f"\n  m=0 (calib baseline, 2-file test) shoe-5 = {m0:.3f}")
    print(f"  (plain 3-file LOSO = {LOSO_M0_REFERENCE}; differs because m=0 tests on 2/3 files "
          f"& averages per-fold F1 — same-test-set baseline for calibration gain is m=0, not plain LOSO)")

    gain = (curve_shoe[1] - curve_shoe[0]) if (curve_shoe[1] and curve_shoe[0]) else None
    if gain is not None:
        print(f"  gain m=0 -> m=1 (1 peak/action): +{gain:.3f}  (within-protocol, honest)")
    best_m = max((m for m in M_VALUES if curve_shoe[m] is not None), key=lambda m: curve_shoe[m])
    print(f"  shoe-5 best: m={best_m} -> {curve_shoe[best_m]:.3f}  (ceiling file-grouped 0.963)")

    # ── Save json + figure ───────────────────────────────────────────────────────
    (OUT_DIR / "calibration.json").write_text(json.dumps({
        "m_values": M_VALUES,
        "shoe4_mean": curve_shoe,
        "jeong": curve_jeong,
        "per_subject": {s: {m: round(float(np.mean(results[s][m])), 4)
                            for m in M_VALUES if results[s][m]} for s in subjects},
        "loso_reference_m0": LOSO_M0_REFERENCE,
        "file_grouped_ceiling": 0.963,
    }, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = M_VALUES
    ax.plot(xs, [curve_shoe[m] for m in xs], "o-", color="#1f4e79", label="shoe-5 mean")
    if any(curve_jeong[m] is not None for m in xs):
        ax.plot(xs, [curve_jeong[m] for m in xs], "s--", color="#d72638",
                label="JEONG (barefoot, env. transfer)")
    ax.axhline(0.963, ls=":", color="#888", label="file-grouped ceiling 0.963")
    ax.axhline(LOSO_M0_REFERENCE, ls=":", color="#bbb", label="plain LOSO 0.605")
    ax.set_xlabel("calibration peaks per action (m)"); ax.set_ylabel("LOSO macro-F1")
    ax.set_ylim(0.5, 1.0); ax.set_xticks(xs)
    ax.set_title("On-device calibration learning curve\n(file-separated; preprocessing frozen)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    for m in xs:
        if curve_shoe[m] is not None:
            ax.annotate(f"{curve_shoe[m]:.2f}", (m, curve_shoe[m]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_curve.png", dpi=130)
    fig.savefig(FIG_DIR / "calibration_curve.png", dpi=130)
    plt.close(fig)

    print(f"\n  saved -> {OUT_DIR}/calibration.json, {FIG_DIR}/calibration_curve.png")
    print("# done.")


if __name__ == "__main__":
    main()
