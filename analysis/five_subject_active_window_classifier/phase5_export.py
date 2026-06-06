#!/usr/bin/env python3
"""Phase 5 (a) — export the deployable KNN model to iOS knn_params.json format.

Deploys the FULL KNN (no feature-selection — H1 decision: calib m=1 full 0.887 vs
sel 0.902, ~equal in real use, and the existing Swift pipeline has no selection
stage). Refit on the whole SHOE-4 cohort (JEONG excluded), then dump the exact
schema MotionRuleClassifier.swift decodes:

  feature_names : [str]      full feature list BEFORE VarianceThreshold
  variance_mask : [bool]     which of those survive VarianceThreshold (len = feature_names)
  scaler_center : [float]    RobustScaler center_ over surviving features
  scaler_scale  : [float]    RobustScaler scale_  over surviving features
  train_vectors : [[float]]  every training sample AFTER variance->scaler->normalizer(l2)
  train_labels  : [str]

Swift then, at inference, builds the same feature dict, applies variance_mask,
RobustScaler, Normalizer(l2), and 1-NN Manhattan against train_vectors — identical
to our Python pipeline. (b) verifies this numerically.

Run: python3 analysis/five_subject_active_window_classifier/phase5_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, Normalizer

THIS = Path(__file__).resolve()
OUT = THIS.parent.parents[1] / "iOS" / "Resources" / "knn_params.json"
META_OUT = THIS.parent / "model_metadata.json"
sys.path.insert(0, str(THIS.parent))

from evaluation import load_dataset, split_cohort
import phase3_models as p3


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    # Shoe-4 cohort only (JEONG excluded) — same as all our reported metrics.
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    # Exclude meta features (same as best model). These names become feature_names.
    keep = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    feature_names = [feats[i] for i in keep]
    X = X[:, keep]
    n_samples, n_feat = X.shape

    # Fit preprocessing on the FULL cohort (deployment refit), mirroring knn(sel=False):
    #   imputer(const 0) -> VarianceThreshold -> RobustScaler -> Normalizer(l2)
    imputer = SimpleImputer(strategy="constant", fill_value=0).fit(X)
    Xi = imputer.transform(X)
    variance = VarianceThreshold().fit(Xi)
    variance_mask = variance.get_support()              # bool over feature_names
    Xv = variance.transform(Xi)
    scaler = RobustScaler().fit(Xv)
    Xs = scaler.transform(Xv)
    normalizer = Normalizer("l2")
    Xn = normalizer.transform(Xs)                        # train_vectors (scaled+normalized)

    params = {
        "feature_names": feature_names,                  # before variance
        "variance_mask": [bool(b) for b in variance_mask],
        "scaler_center": [float(c) for c in scaler.center_],
        "scaler_scale":  [float(s) for s in scaler.scale_],
        "train_vectors": [[round(float(v), 6) for v in row] for row in Xn],
        "train_labels":  y.tolist(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(params))

    # Sanity: every train_vector should be unit L2 norm (Normalizer applied).
    norms = np.sqrt((Xn ** 2).sum(axis=1))

    print("# Phase 5 (a) — KNN export (FULL, shoe-5)")
    print(f"  cohort: {sorted(set(subj))}  samples={n_samples}  classes={len(labels)}")
    print(f"  feature_names: {n_feat}  -> variance kept: {int(variance_mask.sum())}")
    print(f"  train_vectors: {Xn.shape}  (L2 norm min={norms.min():.4f} max={norms.max():.4f})")
    print(f"  saved -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")

    # model_metadata.json — honesty note (reported metric is LOSO, this is a refit).
    meta = {
        "model": "KNN(n=1, weights=distance, metric=manhattan), FULL (no feature-selection)",
        "cohort": sorted({str(s) for s in subj.tolist()}),
        "n_train_samples": int(n_samples),
        "n_classes": len(labels),
        "labels": labels,
        "reported_metric_LOSO_macro_f1_full": 0.581,
        "reported_metric_LOSO_macro_f1_sel_best": 0.605,
        "calibration_m1_macro_f1_full": 0.887,
        "honesty_note": (
            "Reported accuracy is the LEAVE-ONE-SUBJECT-OUT estimate (unseen person). "
            "This deployed model is REFIT on all shoe-5 subjects, so it is a different "
            "model; its training accuracy is an overfit indicator, not generalization. "
            "Real-world use applies on-device few-shot calibration (m=1 -> ~0.89)."
        ),
    }
    META_OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  saved -> {META_OUT}")
    print("# done.")


if __name__ == "__main__":
    main()
