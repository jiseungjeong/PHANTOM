#!/usr/bin/env python3
"""Faithful reproduction of BoxerSense (Hanada et al., 2021) on OUR data.

Paper spec (verified against the PDF, reallly-boxersense.pdf):
  §3.4 features : 7 stats per axis (mean, median, std, min, max, 25%ile, 75%ile)
                  on 3-axis accel + 3-axis gyro = 6 axes -> 42 features / IMU.
  §3.4 models   : SVM / RF / KNN, scikit-learn DEFAULT params.
                  RF & KNN: features normalized to [0,1] (paper states this).
  §4   eval     : main result (PI 0.911) = ONE wrist IMU, 3 REAR punches (RS,RH,RU),
                  10 participants, leave-one-person-out (= our LOSO).
  §3.3 segment  : Butterworth(2nd,1Hz)+dual-threshold. We CANNOT reuse (our data is
                  jump-synced 5-rep segments) -> we share OUR segmentation (the only
                  honest option; isolates feature/model differences, not segmentation).

This script evaluates three reproduction tiers, each labelled by fidelity:
  R1  wrist-only IMU, 3 rear punches      -> MOST faithful (matches paper main exactly
                                             in #sensors and #classes)
  R2  wrist-only IMU, 6 punches (lead+rear)-> paper's config-2 wrist case
  R3  3 IMUs (our extension), our 10 actions-> NOT in the paper; our own extension

Output: reports/boxersense_repro.json  (numbers for the slide + a fidelity table)
Run:    python3 analysis/five_subject_active_window_classifier/phase35_boxersense_repro.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import load_dataset, split_cohort, loso
from phase45_calibration_extra import run_curve, build_index

OUT = Path(__file__).resolve().parent / "reports"

# ── paper feature spec ────────────────────────────────────────────────────────
STATS = ["mean", "p50", "std", "min", "max", "p25", "p75"]   # paper's 7 (median=p50)
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]                  # 3 accel + 3 gyro

# our action -> punch grouping
REAR_PUNCHES = ["Rear Jab", "Rear Hook", "Rear Uppercut"]                 # paper main (3 cls)
ALL_PUNCHES = ["Lead Jab", "Lead Hook", "Lead Uppercut",
               "Rear Jab", "Rear Hook", "Rear Uppercut"]                  # paper config-2 (6 cls)


def feat_cols(all_names, devices):
    """Column indices for 7stats x 6axes x len(devices), in paper order."""
    idx = {n: i for i, n in enumerate(all_names)}
    cols, picked = [], []
    for dev in devices:
        for ax in AXES:
            for st in STATS:
                fn = f"{dev}_{ax}_{st}"
                if fn in idx:
                    cols.append(idx[fn]); picked.append(fn)
    return cols, picked


# ── paper models (sklearn defaults; RF/KNN normalized to [0,1]) ────────────────
def _pipe(scaled, model):
    steps = [("imp", SimpleImputer(strategy="constant", fill_value=0)),
             ("var", VarianceThreshold())]
    if scaled:
        steps.append(("scale", MinMaxScaler()))   # paper: [0,1]
    steps.append(("clf", model))
    return Pipeline(steps)


def models():
    return {
        "SVM": lambda: _pipe(True, SVC()),                       # default RBF, C=1 (paper best)
        "RandomForest": lambda: _pipe(False, RandomForestClassifier(random_state=0)),
        "KNN": lambda: _pipe(True, KNeighborsClassifier()),      # default k=5
    }


def run_tier(name, devices, keep_labels, X_all, y, subj, files, feat_names):
    cols, picked = feat_cols(feat_names, devices)
    mask = np.isin(y, keep_labels)
    X = X_all[mask][:, cols]
    yy = y[mask]
    ss = subj[mask]
    ff = files[mask]
    labs = sorted(set(yy.tolist()))
    res = {}
    for mname, make in models().items():
        lo = loso(make, X, yy, ss, labs)
        res[mname] = {"macro_f1": lo["macro_f1"], "accuracy": lo["accuracy"],
                      "per_subject_mean": lo["per_subject_mean"]}
    best = max(res, key=lambda k: res[k]["macro_f1"])

    # ── 1-demo calibration (file-separated) on the BEST model of this tier ─────
    # run_curve freezes preproc on factory, refits the model on (factory + m calib peaks).
    by_sl_file = build_index(yy, ss, ff)
    subjects = sorted(set(ss.tolist()))
    curve = run_curve(models()[best], X, yy, ss, ff, labs, subjects, by_sl_file)
    # average per-subject macro-F1 at each m (mirror the deployed calibration metric)
    calib = {}
    for m in (0, 1):
        per_subj = [np.mean(curve[s][m]) for s in subjects if m in curve[s] and curve[s][m]]
        calib[m] = float(np.mean(per_subj)) if per_subj else None

    return {"devices": devices, "n_features": len(picked), "n_classes": len(labs),
            "classes": labs, "n_samples": int(mask.sum()),
            "per_model": res, "best_model": best,
            "best_macro_f1": res[best]["macro_f1"], "best_accuracy": res[best]["accuracy"],
            "calib_m0": calib[0], "calib_m1": calib[1]}


def main():
    Xa, ya, subja, filesa, feats, _ = load_dataset()
    (Xc, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)   # shoe-5 cohort
    print(f"# BoxerSense reproduction on our shoe-5 cohort: {len(y)} samples, "
          f"subjects={sorted(set(subj))}\n")

    tiers = {
        "R1_wrist_3rear": run_tier("R1", ["w"], REAR_PUNCHES, Xc, y, subj, files, feats),
        "R2_wrist_6punch": run_tier("R2", ["w"], ALL_PUNCHES, Xc, y, subj, files, feats),
        "R3_3imu_6punch": run_tier("R3", ["a", "i", "w"], ALL_PUNCHES, Xc, y, subj, files, feats),
    }

    paper = {  # from the PDF, PI = leave-one-person-out (matches our LOSO)
        "R1_wrist_3rear": {"PI_macro_f1": 0.906, "PI_acc": 0.911, "best": "SVM",
                           "setup": "1 wrist IMU, 3 rear punches, 10 people"},
    }

    for k, t in tiers.items():
        print(f"## {k}: {t['devices']} | {t['n_classes']} cls | {t['n_features']} feat "
              f"| {t['n_samples']} samp")
        for m, r in t["per_model"].items():
            print(f"    {m:14s} macro-F1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f}")
        print(f"    -> best {t['best_model']} macro-F1 {t['best_macro_f1']:.3f}")
        print(f"    -> calibration ({t['best_model']}): m0 {t['calib_m0']:.3f} -> m1 {t['calib_m1']:.3f}")
        if k in paper:
            p = paper[k]
            print(f"    [paper PI: {p['PI_macro_f1']:.3f} F1 / {p['PI_acc']:.3f} acc "
                  f"({p['best']}, {p['setup']})]")
        print()

    OUT.mkdir(exist_ok=True)
    (OUT / "boxersense_repro.json").write_text(
        json.dumps({"tiers": tiers, "paper": paper}, indent=2, ensure_ascii=False))
    print(f"saved -> {OUT}/boxersense_repro.json")


if __name__ == "__main__":
    main()
