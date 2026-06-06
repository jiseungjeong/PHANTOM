#!/usr/bin/env python3
"""B1 — export the combo-augmented KNN reference to knn_params_combo.json.

Phase 7 (e2s) showed: single + SELECTED combo windows (dominant-sensor conf >= 1.0)
improves fast-transition (combo) recognition while barely touching single accuracy
(single -0.013, combo +0.116, overall +0.035). This script bakes that reference into
the iOS knn_params schema, as a SEPARATE file so the single-only demo file is left
intact (knn_params.json). Same frozen preprocessing as phase5_export.

Output: iOS/Resources/knn_params_combo.json  (single 598 + selected combo 264)

Run: python3 analysis/five_subject_active_window_classifier/phase5_export_combo.py
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
OUT = THIS.parent.parents[1] / "iOS" / "Resources" / "knn_params_combo.json"
COMBO_NPZ = THIS.parent / "reports" / "combo_windows.npz"
sys.path.insert(0, str(THIS.parent))
from evaluation import load_dataset, split_cohort
import phase3_models as p3

CONF_THR = 1.0   # selected combo windows (matches Phase 7 e2s)


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (Xs, ys, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    combo = np.load(COMBO_NPZ, allow_pickle=True)
    Xc_all = combo["X"]; yc = combo["y"].astype(str)
    cfeats = [str(f) for f in combo["feature_names"]]
    cconf = combo["conf"].astype(float) if "conf" in combo.files else np.ones(len(yc))

    # common features (single column order), then drop meta features (deploy parity)
    common = [f for f in feats if f in set(cfeats)]
    common = [f for f in common if f not in p3.META_FEATURES]
    s_idx = [feats.index(f) for f in common]
    c_idx = [cfeats.index(f) for f in common]
    Xs = Xs[:, s_idx]
    Xc = Xc_all[:, c_idx]

    # selection: high-confidence combo windows only
    sel = cconf >= CONF_THR
    Xc_sel, yc_sel = Xc[sel], yc[sel]

    # combine single + selected combo
    X = np.vstack([Xs, Xc_sel])
    y = np.concatenate([ys, yc_sel])
    feature_names = common

    # frozen preprocessing (== knn(sel=False)): imputer->variance->scaler->normalizer
    imputer = SimpleImputer(strategy="constant", fill_value=0).fit(X)
    Xi = imputer.transform(X)
    variance = VarianceThreshold().fit(Xi)
    variance_mask = variance.get_support()
    Xv = variance.transform(Xi)
    scaler = RobustScaler().fit(Xv)
    Xsc = scaler.transform(Xv)
    Xn = Normalizer("l2").transform(Xsc)

    params = {
        "feature_names": feature_names,
        "variance_mask": [bool(b) for b in variance_mask],
        "scaler_center": [float(c) for c in scaler.center_],
        "scaler_scale":  [float(s) for s in scaler.scale_],
        "train_vectors": [[round(float(v), 6) for v in row] for row in Xn],
        "train_labels":  y.tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(params))

    norms = np.sqrt((Xn ** 2).sum(axis=1))
    print("# B1 — combo-augmented KNN export")
    print(f"  single: {len(ys)}  + selected combo (conf>={CONF_THR}): {int(sel.sum())}/{len(yc)}"
          f"  = total {len(y)}")
    print(f"  feature_names: {len(feature_names)}  -> variance kept: {int(variance_mask.sum())}")
    print(f"  train_vectors: {Xn.shape}  (L2 min={norms.min():.4f} max={norms.max():.4f})")

    # Swift-equiv self-check: re-derive the Swift inference path in numpy and confirm
    # 1-NN manhattan on a few held vectors reproduces sklearn's KNN on the same data.
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan").fit(Xn, y)
    # leave-one-out style spot check on 30 random rows
    rng = list(range(0, len(Xn), max(1, len(Xn) // 30)))
    agree = 0
    for i in rng:
        # numpy 1-NN manhattan over all OTHER rows (mimics Swift loop)
        d = np.abs(Xn - Xn[i]).sum(axis=1); d[i] = np.inf
        swift_lbl = y[int(np.argmin(d))]
        # sklearn on data without row i
        mask = np.ones(len(Xn), bool); mask[i] = False
        sk = KNeighborsClassifier(1, weights="distance", metric="manhattan").fit(Xn[mask], y[mask])
        if swift_lbl == sk.predict(Xn[i:i+1])[0]:
            agree += 1
    print(f"  Swift-equiv spot check: {agree}/{len(rng)} match (numpy 1-NN == sklearn)")
    print(f"  saved -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    from collections import Counter as _C
    print(f"  per-label: {dict(sorted(_C(y.tolist()).items()))}")


if __name__ == "__main__":
    main()
