#!/usr/bin/env python3
"""Fair feature comparison: is our edge from FEATURES or from the whole pipeline?

The coverage_calib chart compared PHANTOM (full pipeline: 489 feats -> tree-select,
RobustScaler+L2, 1-NN) vs BoxerSense (126 feats, MinMax, SVC). That changes features
AND selection AND scaler AND model at once — not a clean feature test.

Here we hold MODEL + PREPROCESSING fixed and swap ONLY the feature set:
  feat sets:  OURS_FULL (489) | OURS_SEL (tree-select) | BOXER (126)
  controls :  C1 = 1-NN  + RobustScaler+L2  (our model/scaler, swap features)
              C2 = SVC   + MinMax           (boxer model/scaler, swap features)
Each at calib m=0 and m=1 (file-separated LOSO, macro-F1 over all 10 classes).

If OURS beats BOXER under BOTH controls -> the features genuinely carry more info.
If the gap vanishes when model/scaler are equalized -> the edge was pipeline, not features.

Run: python3 analysis/five_subject_active_window_classifier/phase35_fair_feature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler, Normalizer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluation import load_dataset, split_cohort
from phase45_calibration_extra import run_curve, build_index

STATS = ["mean", "p50", "std", "min", "max", "p25", "p75"]
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
DEVICES = ["a", "i", "w"]
SEED = 42


def boxer_cols(names):
    idx = {n: i for i, n in enumerate(names)}
    return [idx[f"{d}_{ax}_{st}"] for d in DEVICES for ax in AXES for st in STATS
            if f"{d}_{ax}_{st}" in idx]


# ── feature-set transformers (applied as the FIRST step, before shared preproc) ──
class ColumnPick:
    """Pick fixed columns (BoxerSense 126). sklearn-compatible."""
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.cols]
    def get_params(self, deep=True): return {"cols": self.cols}
    def set_params(self, **p):
        self.cols = p.get("cols", self.cols); return self


def _shared_pre_ours():
    return [("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ("var", VarianceThreshold()),
            ("scale", RobustScaler()), ("norm", Normalizer("l2"))]


def _shared_pre_boxer():
    return [("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ("var", VarianceThreshold()), ("scale", MinMaxScaler())]


def _sel_step():
    et = ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    return ("select", SelectFromModel(et, threshold="median"))


def make_pipe(feat_set, control, cols):
    """feat_set in {ours_full, ours_sel, boxer}; control in {C1_knn_ours, C2_svc_boxer}."""
    steps = []
    if feat_set == "boxer":
        steps.append(("pick", ColumnPick(cols)))      # restrict to BoxerSense 126
    # shared preprocessing depends on the control (model/scaler family)
    if control == "C1_knn_ours":
        steps += _shared_pre_ours()
        if feat_set == "ours_sel":
            steps.insert(2, _sel_step())              # selection after imp/var
        model = ("clf", KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan"))
    else:  # C2_svc_boxer
        steps += _shared_pre_boxer()
        if feat_set == "ours_sel":
            steps.insert(2, _sel_step())
        model = ("clf", SVC())
    steps.append(model)
    return Pipeline(steps)


def curve_macro(make, X, y, subj, files, labels):
    subjects = sorted(set(subj.tolist()))
    by = build_index(y, subj, files)
    curve = run_curve(make, X, y, subj, files, labels, subjects, by)
    out = {}
    for m in (0, 1):
        vals = [np.mean(curve[s][m]) for s in subjects if curve[s].get(m)]
        out[m] = float(np.mean(vals)) if vals else None
    return out


def main():
    Xa, ya, subja, filesa, feats, _ = load_dataset()
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)
    labels = sorted(set(y.tolist()))
    cols = boxer_cols(feats)

    print(f"# Fair feature comparison (shoe-5, {len(y)} samp, 10 classes, file-sep LOSO)\n")
    print(f"  feature counts: ours_full={X.shape[1]}, boxer={len(cols)}, "
          f"ours_sel=~half of full (tree-selection)\n")

    controls = ["C1_knn_ours", "C2_svc_boxer"]
    feat_sets = ["ours_full", "ours_sel", "boxer"]
    results = {}
    for ctrl in controls:
        print(f"## Control {ctrl}  (model+scaler fixed, features swapped)")
        results[ctrl] = {}
        for fs in feat_sets:
            r = curve_macro(lambda fs=fs, ctrl=ctrl: make_pipe(fs, ctrl, cols),
                            X, y, subj, files, labels)
            results[ctrl][fs] = r
            print(f"    {fs:12s}  m0 {r[0]:.3f}  m1 {r[1]:.3f}")
        # verdict per control (m1)
        ob = results[ctrl]["ours_sel"][1] - results[ctrl]["boxer"][1]
        print(f"    -> ours_sel vs boxer @ m1: {ob:+.3f} "
              f"({'ours better' if ob>0.02 else 'boxer better' if ob<-0.02 else 'tie'})\n")

    import json
    (HERE / "reports" / "fair_feature.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved -> {HERE}/reports/fair_feature.json")


if __name__ == "__main__":
    main()
