#!/usr/bin/env python3
"""Phase 3.5 — Baseline comparison: BoxerSense vs ours (for the presentation).

Paper: Hanada et al., "BoxerSense: Punch Detection and Classification Using IMUs",
       ABC 2021 / Springer SIST 291 (2022), DOI 10.1007/978-981-19-0361-8_6.

Goal: evaluate the BoxerSense method under **exactly our conditions** (our 5-subject
      data, shoe-4 cohort, same segmentation, same LOSO) -> fair comparison against
      our best (KNN+sel, macro-F1 0.605).

Fairness: on the same action segments (choose_real_peaks+uniform_windows), only the
          **features + model** are swapped to BoxerSense's. Any difference comes
          purely from the feature/model choice.

── BoxerSense spec (extracted from the paper) ──────────────────────────────────
 features (§3.4): 7 stats per axis = mean, median, std, min, max, 25%ile, 75%ile.
   3-axis accel + 3-axis gyro = 6 axes -> 42 features/IMU. (We apply it to 3 IMUs -> 7x6x3 = 126)
 models (§3.4, §4.4): SVM / RF / KNN, scikit-learn default params.
   - RF/KNN are distance-based, so features are [0,1] normalized (stated in the paper).
 segmentation (§3.3): Butterworth(2nd,1Hz)+dual-threshold. -> differs from our
   structure (jump + 5 reps), so not usable as-is. **Share our segmentation** (fair-comparison decision).
 evaluation: paper uses PD 10-fold + PI leave-one-person-out. Ours = shoe-4 LOSO (=PI) primary metric.

NOTE: all 126 BoxerSense features **already exist** in our single_dataset.json
   (verified, 0 missing). They were extracted from the same segments, so we just
   select them by name (no recomputation needed).

Run: python3 analysis/five_subject_active_window_classifier/phase35_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import load_dataset, split_cohort, loso

# ── BoxerSense feature spec ──────────────────────────────────────────────────
DEVICES = ["a", "i", "w"]                       # AirPods, iPhone, Watch (our 3 IMUs)
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]     # 3 accel + 3 gyro
STATS = ["mean", "p50", "std", "min", "max", "p25", "p75"]   # paper's 7 stats (median=p50)


def boxersense_features(all_feature_names):
    """Select only the BoxerSense 126 (7 stats x 6 axes x 3 devices) from our feature names."""
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    cols, picked = [], []
    for dev in DEVICES:
        for ax in AXES:
            for st in STATS:
                fn = f"{dev}_{ax}_{st}"
                if fn in name_to_idx:
                    cols.append(name_to_idx[fn]); picked.append(fn)
    return cols, picked


# ── BoxerSense models (sklearn default params, paper §3.4) ─────────────────────
#   RF/KNN: [0,1] normalization (stated in the paper). SVM: defaults (the paper does
#   not specify scaling -> including normalization is safe since RBF is distance-based).
#   VarianceThreshold removes 0-variance axes for devices without a gyro.

def _base(scaled: bool):
    steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0)),
             ("variance", VarianceThreshold())]
    if scaled:
        steps.append(("scaler", MinMaxScaler()))   # paper: [0,1] normalization
    return steps


def svm():
    return Pipeline(_base(True) + [("model", SVC())])               # sklearn default = RBF, C=1


def rf():
    return Pipeline(_base(False) + [("model", RandomForestClassifier(random_state=0))])


def knn():
    return Pipeline(_base(True) + [("model", KNeighborsClassifier())])   # default k=5


MODELS = {"SVM": svm, "RandomForest": rf, "KNN": knn}


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (Xc, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    cols, picked = boxersense_features(feats)
    X = Xc[:, cols]
    print("# Phase 3.5 — baseline: BoxerSense (applied to our setup)")
    print(f"  shoe-4 cohort {len(y)} samples / {len(labels)} classes")
    print(f"  BoxerSense features: {len(picked)} (7 stats x 6 axes x 3 devices, target 126)\n")

    print(f"  {'model':14s} {'LOSO acc':>9s} {'macro-F1':>9s} {'min~max':>12s}")
    results = {}
    for name, make in MODELS.items():
        lo = loso(make, X, y, subj, labels)
        results[name] = lo
        rng = f"{lo['per_subject_min']:.2f}~{lo['per_subject_max']:.2f}"
        print(f"  {name:14s} {lo['per_subject_mean']:9.4f} {lo['macro_f1']:9.4f} {rng:>12s}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    bf1 = results[best]["macro_f1"]
    print(f"\n  * BoxerSense best: {best} — LOSO macro-F1 {bf1:.4f}")

    print(f"\n## best({best}) per-class F1 (ascending)")
    for lab, f in sorted(results[best]["per_class_f1"].items(), key=lambda x: x[1]):
        print(f"    {lab:16s} F1={f:.3f}")

    # ── Comparison table vs our method ───────────────────────────────────────
    OUR_BEST = 0.605   # Phase 3: KNN + feature-selection, shoe-4 LOSO macro-F1
    print("\n## Comparison (same data, segmentation, LOSO)")
    print(f"  {'method':28s} {'feature':>9s} {'LOSO macro-F1':>14s}")
    print(f"  {'BoxerSense (' + best + ')':28s} {len(picked):>9d} {bf1:>14.3f}")
    print(f"  {'Ours (KNN+feature-selection)':28s} {'~240':>9s} {OUR_BEST:>14.3f}")
    diff = OUR_BEST - bf1
    verdict = ("ours wins" if diff > 0.02 else
               "BoxerSense wins" if diff < -0.02 else "tie")
    print(f"  -> diff {diff:+.3f} ({verdict})")
    print("\n  NOTE: this is the score of 'the original paper's method applied to our 3-IMU,")
    print("        10-class task'. NOT a direct comparison with the paper's reported number")
    print("        (PI SVM 91% / but single wrist, 3 punch classes).")


if __name__ == "__main__":
    main()
