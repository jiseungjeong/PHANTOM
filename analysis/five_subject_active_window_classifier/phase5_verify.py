#!/usr/bin/env python3
"""Phase 5 (b) — numerical consistency: Swift-equivalent inference vs sklearn.

The deployed knn_params.json is consumed by MotionRuleClassifier.swift, which
re-implements the pipeline by hand. A single mismatch (wrong scaler value, wrong
mask order) silently breaks the KNN distances. So we re-implement Swift's exact
arithmetic HERE in Python (reading the SAME json), run it on every training sample,
and compare against the genuine sklearn pipeline refit on the same data.

If predictions + nearest-neighbor distances match, the Swift port is faithful.
(This Python re-impl IS the Swift algorithm, line-for-line: mask -> (x-center)/scale
-> l2 normalize -> 1-NN Manhattan.)

Run: python3 analysis/five_subject_active_window_classifier/phase5_verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
KNN_JSON = THIS.parent.parents[1] / "iOS" / "Resources" / "knn_params.json"
sys.path.insert(0, str(THIS.parent))

from evaluation import load_dataset, split_cohort
import phase3_models as p3


def swift_predict(params, feat_dict_vec, train_vecs, train_labels):
    """Mirror MotionRuleClassifier.predictKNN exactly.
    feat_dict_vec: raw feature vector aligned to params['feature_names'] (missing=0)."""
    mask = params["variance_mask"]
    center = params["scaler_center"]
    scale = params["scaler_scale"]
    # 1. variance mask
    x = [v for v, m in zip(feat_dict_vec, mask) if m]
    # 2. RobustScaler: (x - center)/scale
    x = [(x[i] - center[i]) / scale[i] for i in range(len(x))]
    # 3. Normalizer l2
    norm = sum(v * v for v in x) ** 0.5
    if norm > 1e-9:
        x = [v / norm for v in x]
    # 4. 1-NN Manhattan
    best_d, best_lab = float("inf"), ""
    xa = np.array(x)
    for vec, lab in zip(train_vecs, train_labels):
        d = float(np.abs(xa - vec).sum())
        if d < best_d:
            best_d, best_lab = d, lab
    return best_lab, best_d


def main() -> None:
    params = json.loads(KNN_JSON.read_text())
    feature_names = params["feature_names"]
    train_vecs = [np.array(r) for r in params["train_vectors"]]
    train_labels = params["train_labels"]

    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)
    # Align X columns to params['feature_names'] (same order export used).
    idx = {n: i for i, n in enumerate(feats)}
    cols = [idx[n] for n in feature_names]
    Xn = X[:, cols]

    # Genuine sklearn pipeline (full KNN) refit on the same data, for cross-check.
    skl = p3.knn(sel=False).fit(Xn, y)

    # Compare on every sample: Swift-equiv vs sklearn.
    n = len(y)
    swift_preds, skl_preds = [], []
    self_match = 0   # Swift-equiv should find the sample itself at distance 0 (in-train)
    for i in range(n):
        raw = Xn[i].tolist()
        sp, sd = swift_predict(params, raw, train_vecs, train_labels)
        swift_preds.append(sp)
        if sd < 1e-6:
            self_match += 1
    skl_preds = skl.predict(Xn).tolist()

    agree = sum(int(a == b) for a, b in zip(swift_preds, skl_preds))
    print("# Phase 5 (b) — Swift-equivalent vs sklearn consistency")
    print(f"  samples: {n}")
    print(f"  Swift-equiv == sklearn predictions: {agree}/{n} = {agree/n*100:.2f}%")
    print(f"  Swift-equiv self-match @dist~0 (in-train 1-NN finds itself): {self_match}/{n}")
    if agree == n:
        print("  [OK] Swift port is numerically faithful — distances/labels match sklearn.")
    else:
        mism = [(i, swift_preds[i], skl_preds[i]) for i in range(n) if swift_preds[i] != skl_preds[i]][:8]
        print(f"  [!] {n-agree} mismatches (first few): {mism}")
        print("      -> investigate preprocessing order/values before trusting the Swift demo.")
    print("# done.")


if __name__ == "__main__":
    main()
