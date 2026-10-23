#!/usr/bin/env python3
"""Phase 2 — evaluation protocol (reusable evaluation engine).

Model-agnostic: Phase 3 plugs multiple
models into this engine for fair comparison.

Provided protocols:
  - LOSO            : leave one whole subject out -> test on that person. per-subject acc/F1 + min~max
  - file_grouped_cv : a recording (file) stays in one fold only (auxiliary metric, within-cohort upper bound)
  - nested_loso     : outer=LOSO evaluation, inner=model/hyperparameter selection on the remaining subjects
                      A2: inner is also split by subject/file (no leakage). No random k-fold.
  - sample-level random k-fold is intentionally NOT provided (leaks the 5 repetition samples).

metric: accuracy + macro-F1 + per-class F1 + confusion (fixed label order).

This file is a collection of functions; running it directly verifies the engine with the baseline (KNN n=1).
Run: python3 analysis/five_subject_active_window_classifier/evaluation.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

THIS = Path(__file__).resolve()
DATASET = THIS.parent / "single_dataset.json"

# Environment policy (auto-extends when subjects are added):
#   - JEONG and SEUNG are the same person measured in two environments
#     (JEONG=indoor/barefoot, SEUNG=outdoor/shoes).
#   - Only JEONG (barefoot) is excluded from train/eval and kept for the
#     'environment robustness test'.
#   - All other subjects (SEUNG, JIHO, JUHO, SHIM + new additions) auto-join the shoe cohort.
#     -> Just add a new subject folder (Data/<name>_data) and it joins the cohort (no hardcoding).
#   - Note: new subjects must also be measured in the 'shoes' environment for consistency
#     (see the measurement guide).
HOLDOUT_ENV_SUBJECTS = ("JEONG",)                     # different environment (barefoot etc.) — excluded from cohort


def split_cohort(X, y, subjects, files):
    """Return the train/eval cohort (everyone except HOLDOUT) and the environment holdout (JEONG).
    As subjects grow, everyone except HOLDOUT auto-joins the cohort."""
    env_mask = np.isin(subjects, HOLDOUT_ENV_SUBJECTS)
    train_mask = ~env_mask
    cohort = (X[train_mask], y[train_mask], subjects[train_mask], files[train_mask])
    env = (X[env_mask], y[env_mask], subjects[env_mask], files[env_mask])
    return cohort, env


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET):
    """single_dataset.json -> (X, y, subjects, files, feature_names, labels)."""
    d = json.loads(path.read_text())
    feats = d["feature_names_motion_all"]
    samples = d["samples"]
    X = np.array([[s["features"].get(n, 0.0) for n in feats] for s in samples], dtype=float)
    y = np.array([s["label"] for s in samples])
    subjects = np.array([s["subject"] for s in samples])
    files = np.array([s["file"] for s in samples])
    labels = sorted(set(y.tolist()))
    return X, y, subjects, files, feats, labels


# ── metric helper ─────────────────────────────────────────────────────────────

def score_block(y_true, y_pred, labels) -> dict:
    """Standard bundle of metrics for the given predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            lab: float(f1_score(y_true, y_pred, labels=[lab], average="macro", zero_division=0))
            for lab in labels
        },
    }


# PipelineFactory: a no-arg callable returning a fresh (unfitted) sklearn-compatible estimator.
PipelineFactory = Callable[[], object]


# ── LOSO (primary metric) ───────────────────────────────────────────────────

def loso(make_pipe: PipelineFactory, X, y, subjects, labels) -> dict:
    """Leave-One-Subject-Out. Collect out-of-fold predictions over all samples and evaluate."""
    subs = sorted(set(subjects.tolist()))
    preds = np.empty(len(y), dtype=object)
    per_subject = {}
    for s in subs:
        te = subjects == s
        tr = ~te
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
        per_subject[s] = float(accuracy_score(y[te], preds[te]))
    overall = score_block(y, preds, labels)
    accs = list(per_subject.values())
    overall["per_subject_acc"] = per_subject
    overall["per_subject_mean"] = float(np.mean(accs))
    overall["per_subject_min"] = float(np.min(accs))
    overall["per_subject_max"] = float(np.max(accs))
    overall["confusion"] = confusion_matrix(y, preds, labels=labels).tolist()
    return overall


# ── file-grouped 5-fold (auxiliary) ───────────────────────────────────────────

def file_grouped_cv(make_pipe: PipelineFactory, X, y, files, labels, k=5, seed=7) -> dict:
    """A file stays in one fold only. Distribute files per label to keep label balance."""
    rng = np.random.RandomState(seed)
    file_to_idx = defaultdict(list)
    for i, f in enumerate(files):
        file_to_idx[f].append(i)
    files_by_label = defaultdict(list)
    for f, idxs in file_to_idx.items():
        files_by_label[y[idxs[0]]].append(f)

    folds = [[] for _ in range(k)]
    for fs in files_by_label.values():
        fs = list(fs); rng.shuffle(fs)
        for off, f in enumerate(fs):
            folds[off % k].extend(file_to_idx[f])

    preds = np.empty(len(y), dtype=object)
    for fi in range(k):
        te = np.array(folds[fi], dtype=int)
        tr = np.array([i for j in range(k) if j != fi for i in folds[j]], dtype=int)
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
    out = score_block(y, preds, labels)
    out["confusion"] = confusion_matrix(y, preds, labels=labels).tolist()
    return out


# ── Nested LOSO (removes model-selection bias) ─────────────────────────────────

def nested_loso(
    candidates: dict[str, PipelineFactory],
    X, y, subjects, labels,
) -> dict:
    """Outer=LOSO. In each outer fold, run an inner LOSO over the remaining subjects,
    pick the best candidate (inner macro-F1), then apply that model to the outer test subject.
    A2: the inner split is also per-subject (LOSO) -> no repetition/subject leakage.

    candidates: {name: factory}. The return also records the model selected per outer fold.
    """
    subs = sorted(set(subjects.tolist()))
    preds = np.empty(len(y), dtype=object)
    chosen_per_outer = {}
    per_subject = {}

    for test_s in subs:
        te = subjects == test_s
        dev_mask = ~te
        dev_subs = [s for s in subs if s != test_s]

        # inner LOSO: evaluate each candidate, leaving out one dev subject at a time
        best_name, best_score = None, -1.0
        inner_scores = {}
        for name, make in candidates.items():
            inner_pred = np.empty(dev_mask.sum(), dtype=object)
            dev_idx = np.where(dev_mask)[0]
            dev_y = y[dev_idx]
            dev_subj = subjects[dev_idx]
            for val_s in dev_subs:
                val = dev_subj == val_s
                tr = ~val
                pipe = make()
                pipe.fit(X[dev_idx[tr]], dev_y[tr])
                inner_pred[val] = pipe.predict(X[dev_idx[val]])
            sc = f1_score(dev_y, inner_pred, labels=labels, average="macro", zero_division=0)
            inner_scores[name] = float(sc)
            if sc > best_score:
                best_score, best_name = sc, name

        # refit the selected model on all dev subjects -> outer test
        pipe = candidates[best_name]()
        pipe.fit(X[dev_mask], y[dev_mask])
        preds[te] = pipe.predict(X[te])
        per_subject[test_s] = float(accuracy_score(y[te], preds[te]))
        chosen_per_outer[test_s] = {"chosen": best_name, "inner_macro_f1": inner_scores}

    out = score_block(y, preds, labels)
    accs = list(per_subject.values())
    out["per_subject_acc"] = per_subject
    out["per_subject_mean"] = float(np.mean(accs))
    out["per_subject_min"] = float(np.min(accs))
    out["per_subject_max"] = float(np.max(accs))
    out["confusion"] = confusion_matrix(y, preds, labels=labels).tolist()
    out["chosen_per_outer"] = chosen_per_outer
    return out


# ── baseline verification (when run directly) ──────────────────────────────────

def _baseline_pipe():
    """Existing model: SimpleImputer -> VarianceThreshold -> RobustScaler -> Normalizer -> KNN(1)."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import RobustScaler, Normalizer
    from sklearn.neighbors import KNeighborsClassifier
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold()),
        ("scaler", RobustScaler()),
        ("normalizer", Normalizer(norm="l2")),
        ("model", KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan")),
    ])


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (X, y, subjects, files), (Xe, ye, sube, _) = split_cohort(Xa, ya, subja, filesa)

    print(f"# Phase 2 — evaluation protocol verification (baseline KNN n=1)")
    print(f"  total samples {len(ya)} / features {len(feats)} / {len(labels)} classes")
    print(f"  train/eval cohort (4 shoe subjects {sorted(set(subjects))}): {len(y)} samples")
    print(f"  environment holdout (JEONG barefoot): {len(ye)} samples — excluded from training, robustness only\n")

    print("## file-grouped 5-fold (auxiliary, within-cohort upper bound)")
    fg = file_grouped_cv(_baseline_pipe, X, y, files, labels)
    print(f"  acc={fg['accuracy']:.4f}  macro-F1={fg['macro_f1']:.4f}")

    print("\n## LOSO — 4 shoe subjects (* honest primary metric)")
    lo = loso(_baseline_pipe, X, y, subjects, labels)
    print(f"  sample-mean acc={lo['accuracy']:.4f}  macro-F1={lo['macro_f1']:.4f}")
    print(f"  * group-mean acc (one vote per person)={lo['per_subject_mean']:.4f}  "
          f"min={lo['per_subject_min']:.4f} max={lo['per_subject_max']:.4f}")
    print(f"    per-subject: { {s: round(a,3) for s,a in lo['per_subject_acc'].items()} }")

    print("\n## Nested LOSO — 4 shoe subjects (single candidate -> sanity check the engine)")
    nl = nested_loso({"knn1": _baseline_pipe}, X, y, subjects, labels)
    print(f"  group-mean acc={nl['per_subject_mean']:.4f}  macro-F1={nl['macro_f1']:.4f}")

    # Environment robustness: test on JEONG (barefoot). SEUNG (=same person as JEONG) must be
    # left out of training so we measure the pure environment gap of 'unseen person + barefoot'.
    # Keeping SEUNG would leak the same person (0.99).
    print("\n## Environment robustness test (SEUNG = same person, excluded from training)")
    no_seung = subjects != "SEUNG"
    pipe = _baseline_pipe(); pipe.fit(X[no_seung], y[no_seung])   # JIHO+JUHO+SHIM only
    env_acc = float(accuracy_score(ye, pipe.predict(Xe)))
    print(f"  train=JIHO+JUHO+SHIM(shoes) -> JEONG(barefoot) acc={env_acc:.4f}")
    # Control: same 3 -> SEUNG (shoes, same person's shoe version) — only environment differs
    seung_mask = subja == "SEUNG"
    seung_acc = float(accuracy_score(ya[seung_mask], pipe.predict(Xa[seung_mask])))
    print(f"  control) same train -> SEUNG(shoes, same person) acc={seung_acc:.4f}")
    print(f"  -> barefoot({env_acc:.2f}) vs shoes({seung_acc:.2f}) gap = environment effect for the same person")

    print("\n## Lowest 5 per-class F1 (shoe-5 LOSO — most-confused classes)")
    for lab, f in sorted(lo["per_class_f1"].items(), key=lambda x: x[1])[:5]:
        print(f"    {lab:16s} F1={f:.3f}")
    print("\n# Engine verification done. (Phase 3 feeds multiple candidates into nested_loso.)")


if __name__ == "__main__":
    main()
