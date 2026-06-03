#!/usr/bin/env python3
"""Phase 3 — 모델 비교 (신발 4명 코호트).

plan.md §5 / todolist Phase 3. evaluation.py 의 LOSO/nested_loso 엔진에
여러 후보를 넣어 LOSO macro-F1 기준으로 공정 비교 → best 선정.

후보:
  - KNN(n=1)        : 기존 baseline (과적합 비교 기준)
  - ExtraTrees      : 기존 10클래스 best 계열
  - SVC(RBF)        : 기존 8클래스 best 계열
  - HistGBT         : Gradient-Boosted Trees (LightGBM 미설치 → sklearn HistGBT 대체)

평가:
  - LOSO (신발 4명, 그룹평균 주지표) — 후보별 acc/macro-F1
  - Nested LOSO — 후보들을 함께 넣어 outer fold 마다 best 자동선택
  - class imbalance 대응(class_weight='balanced') 가능 후보엔 적용

실행: python3 analysis/five_subject_active_window_classifier/phase3_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import load_dataset, split_cohort, loso, nested_loso, file_grouped_cv

SEED = 20260604


# ── 후보 파이프라인 팩토리 ────────────────────────────────────────────────────

def knn1():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold()),
        ("scaler", RobustScaler()),
        ("normalizer", Normalizer(norm="l2")),
        ("model", KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan")),
    ])


def extratrees():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold()),
        ("model", ExtraTreesClassifier(
            n_estimators=300, max_features=0.5, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])


def svc_rbf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("model", SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")),
    ])


def histgbt():
    # LightGBM 미설치 → sklearn 히스토그램 GBT (동일 계열)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold()),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, class_weight="balanced", random_state=SEED)),
    ])


CANDIDATES = {
    "KNN(n=1)": knn1,
    "ExtraTrees": extratrees,
    "SVC(RBF)": svc_rbf,
    "HistGBT": histgbt,
}


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (X, y, subjects, files), _ = split_cohort(Xa, ya, subja, filesa)
    print(f"# Phase 3 — 모델 비교 (신발 4명, {len(y)} 샘플, {len(labels)}클래스)")
    print(f"  피험자: {sorted(set(subjects))}\n")

    print("## 후보별 LOSO (그룹평균 acc / macro-F1) — LOSO 기준 best 선정")
    print(f"  {'모델':12s} {'LOSO acc':>9s} {'macro-F1':>9s} {'min~max':>14s}  {'file-grp':>9s}")
    results = {}
    for name, make in CANDIDATES.items():
        lo = loso(make, X, y, subjects, labels)
        fg = file_grouped_cv(make, X, y, files, labels)
        results[name] = lo
        rng = f"{lo['per_subject_min']:.2f}~{lo['per_subject_max']:.2f}"
        print(f"  {name:12s} {lo['per_subject_mean']:9.4f} {lo['macro_f1']:9.4f} "
              f"{rng:>14s}  {fg['macro_f1']:9.4f}")

    # LOSO macro-F1 기준 best
    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\n  ★ LOSO macro-F1 기준 best: {best} ({results[best]['macro_f1']:.4f})")

    print("\n## Nested LOSO (4 후보 동시 투입 → outer fold 마다 자동 best 선택)")
    nl = nested_loso(CANDIDATES, X, y, subjects, labels)
    print(f"  그룹평균 acc={nl['per_subject_mean']:.4f}  macro-F1={nl['macro_f1']:.4f}  "
          f"min~max={nl['per_subject_min']:.2f}~{nl['per_subject_max']:.2f}")
    print("  outer fold 별 선택된 모델:")
    for s, info in nl["chosen_per_outer"].items():
        print(f"    {s:8s} → {info['chosen']}")

    print(f"\n## best 모델({best}) per-class F1 (낮은 순)")
    for lab, f in sorted(results[best]["per_class_f1"].items(), key=lambda x: x[1]):
        print(f"    {lab:16s} F1={f:.3f}")

    print(f"\n# 끝. baseline KNN macro-F1 {results['KNN(n=1)']['macro_f1']:.3f} "
          f"→ best {best} {results[best]['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
