#!/usr/bin/env python3
"""Phase 3 — model comparison (shoe-5 cohort). Design finalized: reflects M1/M3/M4/M5.

- 4 models: KNN(baseline) / ExtraTrees / SVC(RBF) / HistGBT
- 2 feature sets (M4): full (489) / tree-importance selection (top K)
- evaluation (M5): simple LOSO (shoe-5), best by group-mean macro-F1. Not nested.
- exclude meta features (M1): label-irrelevant meta such as repetition_count.
- HistGBT max_iter 300->150 (small-data overfit / speed).
- verdict (M3): >=0.70 achieved / 0.62~0.70 honest report / ~0.60 plateau -> data limit.

Feature selection is inside the pipeline, so it is selected **only on each LOSO fold's
train** (no leakage).

Run: python3 analysis/five_subject_active_window_classifier/phase3_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import load_dataset, split_cohort, loso

SEED = 20260604

# M1: label-irrelevant meta features (segmentation byproducts) — excluded from training.
META_FEATURES = {
    "single_repetition_sample", "repetition_index", "repetition_count",
    "whole_file_duration_s", "active_peak_time_s", "active_peak_score",
    "active_window_duration_s", "active_window_coverage_pct",
    "active_segmentation_method_peak",
}


# ── Preprocessing block ───────────────────────────────────────────────────────

def _pre():
    return [("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("variance", VarianceThreshold())]


def _selector():
    # Tree-importance-based selection. Drops features below 'median' importance (~half).
    et = ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    return ("select", SelectFromModel(et, threshold="median"))


# ── Model factories (full = no selection / sel = tree-importance selection) ─────

def knn(sel):
    steps = _pre()
    if sel: steps.append(_selector())
    steps += [("scaler", RobustScaler()), ("normalizer", Normalizer("l2")),
              ("model", KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan"))]
    return Pipeline(steps)


def extratrees(sel):
    steps = _pre()
    if sel: steps.append(_selector())
    steps.append(("model", ExtraTreesClassifier(
        n_estimators=300, max_features=0.5, min_samples_leaf=2,
        class_weight="balanced", random_state=SEED, n_jobs=-1)))
    return Pipeline(steps)


def svc(sel):
    steps = _pre()
    if sel: steps.append(_selector())
    steps += [("scaler", StandardScaler()),
              ("model", SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced"))]
    return Pipeline(steps)


def histgbt(sel):
    steps = _pre()
    if sel: steps.append(_selector())
    steps.append(("model", HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, l2_regularization=1.0,
        class_weight="balanced", random_state=SEED)))   # 300->150 (M2)
    return Pipeline(steps)


MODELS = {"KNN(n=1)": knn, "ExtraTrees": extratrees, "SVC(RBF)": svc, "HistGBT": histgbt}


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    # M1: exclude meta features
    keep = [i for i, n in enumerate(feats) if n not in META_FEATURES]
    X = X[:, keep]
    print(f"# Phase 3 — model comparison (shoe-5, {len(y)} samples)")
    print(f"  feature: {len(feats)} -> meta excluded {len(keep)}  / {len(labels)} classes")
    print(f"  baseline ref: KNN shoe-5 LOSO macro-F1 ~= 0.57\n")

    print(f"  {'model':12s} {'feature':>10s} {'LOSO acc':>9s} {'macro-F1':>9s} {'min~max':>12s}")
    results = {}
    for name, make in MODELS.items():
        for sel, tag in ((False, "full"), (True, "sel")):
            lo = loso(lambda s=sel, mk=make: mk(s), X, y, subj, labels)
            results[(name, tag)] = lo
            rng = f"{lo['per_subject_min']:.2f}~{lo['per_subject_max']:.2f}"
            print(f"  {name:12s} {tag:>10s} {lo['per_subject_mean']:9.4f} "
                  f"{lo['macro_f1']:9.4f} {rng:>12s}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    bf1 = results[best]["macro_f1"]
    print(f"\n  * best: {best[0]} ({best[1]}) — LOSO macro-F1 {bf1:.4f}")

    # M3 verdict
    if bf1 >= 0.70:
        verdict = "[GOAL] target (0.70) achieved"
    elif bf1 >= 0.62:
        verdict = "[OK] meaningful improvement — honestly report as 'this level with current data'"
    else:
        verdict = "[FLAT] plateau (~0.60) — not a model bottleneck = data limit (4 subjects)"
    print(f"  verdict (M3): {verdict}  (baseline 0.57 -> {bf1:.3f})")

    print(f"\n## best({best[0]}/{best[1]}) per-class F1 (ascending)")
    for lab, f in sorted(results[best]["per_class_f1"].items(), key=lambda x: x[1]):
        print(f"    {lab:16s} F1={f:.3f}")

    print(f"\n## feature selection effect (full vs sel, macro-F1)")
    for name in MODELS:
        ff = results[(name, "full")]["macro_f1"]; sf = results[(name, "sel")]["macro_f1"]
        arrow = "↑" if sf > ff else ("↓" if sf < ff else "=")
        print(f"    {name:12s} full {ff:.3f} → sel {sf:.3f}  {arrow}")


if __name__ == "__main__":
    main()
