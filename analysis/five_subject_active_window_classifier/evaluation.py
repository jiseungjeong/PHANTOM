#!/usr/bin/env python3
"""Phase 2 — 평가 프로토콜 (재사용 가능한 평가 엔진).

plan.md §6 / todolist Phase 2 구현. 모델-agnostic: Phase 3 가 여러 모델을
이 엔진에 넣어 공정 비교한다.

제공 프로토콜:
  - LOSO            : 한 피험자 통째 제외 → 그 사람으로 테스트. per-subject acc/F1 + min~max
  - file_grouped_cv : 같은 녹화(파일)는 한 fold 에만 (보조 지표, within-cohort 상한)
  - nested_loso     : outer=LOSO 평가, inner=남은 피험자로 모델/하이퍼파라미터 선택
                      ⭐ A2: inner 도 피험자/파일 단위 분할(누설 차단). random k-fold 금지.
  - sample-level random k-fold 는 의도적으로 제공하지 않음 (반복 5샘플 누설).

metric: accuracy + macro-F1 + per-class F1 + confusion(라벨순서 고정).

이 파일은 함수 모음이며, 직접 실행하면 baseline(KNN n=1)으로 엔진 동작을 검증한다.
실행: python3 analysis/five_subject_active_window_classifier/evaluation.py
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

# ⭐ 환경 정책:
#   - JEONG·SEUNG 은 동일인의 두 환경 측정 (JEONG=실내·맨발 / SEUNG=실외·신발).
#   - 나머지(JIHO·JUHO·SHIM)는 모두 신발. 앱 실사용 환경(신발)과 일관성을 위해
#     학습/평가 코호트는 신발 4명(SEUNG·JIHO·JUHO·SHIM)으로 한다.
#   - JEONG(맨발)은 학습/평가에서 제외하고 '환경 robustness 테스트' 전용으로 보관.
#     → 이로써 (a) 동일인 누설 차단, (b) 환경 일관, (c) fold 균등(각 25%) 모두 달성.
TRAIN_SUBJECTS = ("SEUNG", "JIHO", "JUHO", "SHIM")   # 신발, 일관 환경
HOLDOUT_ENV_SUBJECT = "JEONG"                         # 맨발, robustness 분석 전용


def split_cohort(X, y, subjects, files):
    """신발 4명(학습/평가) 과 JEONG(환경 holdout)을 분리해 반환."""
    train_mask = np.isin(subjects, TRAIN_SUBJECTS)
    env_mask = subjects == HOLDOUT_ENV_SUBJECT
    cohort = (X[train_mask], y[train_mask], subjects[train_mask], files[train_mask])
    env = (X[env_mask], y[env_mask], subjects[env_mask], files[env_mask])
    return cohort, env


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET):
    """single_dataset.json → (X, y, subjects, files, feature_names, labels)."""
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
    """주어진 예측에 대한 표준 metric 묶음."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            lab: float(f1_score(y_true, y_pred, labels=[lab], average="macro", zero_division=0))
            for lab in labels
        },
    }


# PipelineFactory: 인자 없이 새 (미학습) sklearn-호환 추정기를 반환하는 콜러블.
PipelineFactory = Callable[[], object]


# ── LOSO (주 지표) ────────────────────────────────────────────────────────────

def loso(make_pipe: PipelineFactory, X, y, subjects, labels) -> dict:
    """Leave-One-Subject-Out. 모든 샘플에 대해 out-of-fold 예측을 모아 평가."""
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


# ── file-grouped 5-fold (보조) ────────────────────────────────────────────────

def file_grouped_cv(make_pipe: PipelineFactory, X, y, files, labels, k=5, seed=7) -> dict:
    """같은 파일은 한 fold 에만. 라벨별로 파일을 분배해 라벨 균형 유지."""
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


# ── Nested LOSO (모델 선택 편향 제거) ─────────────────────────────────────────

def nested_loso(
    candidates: dict[str, PipelineFactory],
    X, y, subjects, labels,
) -> dict:
    """Outer=LOSO. 각 outer fold 에서, 남은 피험자들로 inner LOSO 를 돌려
    후보 중 best(inner macro-F1)를 고른 뒤 그 모델을 outer 테스트 피험자에 적용.
    ⭐ A2: inner 분할도 피험자 단위(LOSO) → 반복/피험자 누설 없음.

    candidates: {이름: 팩토리}. 반환에 outer fold 별 선택된 모델도 기록.
    """
    subs = sorted(set(subjects.tolist()))
    preds = np.empty(len(y), dtype=object)
    chosen_per_outer = {}
    per_subject = {}

    for test_s in subs:
        te = subjects == test_s
        dev_mask = ~te
        dev_subs = [s for s in subs if s != test_s]

        # inner LOSO: dev_subs 안에서 한 명씩 빼며 각 후보 평가
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

        # 선택된 모델을 dev 전체로 재학습 → outer 테스트
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


# ── baseline 검증 (직접 실행 시) ──────────────────────────────────────────────

def _baseline_pipe():
    """기존 모델: SimpleImputer→VarianceThreshold→RobustScaler→Normalizer→KNN(1)."""
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

    print(f"# Phase 2 — 평가 프로토콜 검증 (baseline KNN n=1)")
    print(f"  전체 샘플 {len(ya)} / feature {len(feats)} / {len(labels)}클래스")
    print(f"  학습·평가 코호트(신발 4명 {sorted(set(subjects))}): {len(y)} 샘플")
    print(f"  환경 holdout(JEONG 맨발): {len(ye)} 샘플 — 학습 제외, robustness 전용\n")

    print("## file-grouped 5-fold (보조, within-cohort 상한)")
    fg = file_grouped_cv(_baseline_pipe, X, y, files, labels)
    print(f"  acc={fg['accuracy']:.4f}  macro-F1={fg['macro_f1']:.4f}")

    print("\n## LOSO — 신발 4명 (★ 정직한 주 지표)")
    lo = loso(_baseline_pipe, X, y, subjects, labels)
    print(f"  샘플평균 acc={lo['accuracy']:.4f}  macro-F1={lo['macro_f1']:.4f}")
    print(f"  ★ 그룹평균 acc(사람당 1표)={lo['per_subject_mean']:.4f}  "
          f"min={lo['per_subject_min']:.4f} max={lo['per_subject_max']:.4f}")
    print(f"    per-subject: { {s: round(a,3) for s,a in lo['per_subject_acc'].items()} }")

    print("\n## Nested LOSO — 신발 4명 (단일 후보 → 엔진 동작 확인)")
    nl = nested_loso({"knn1": _baseline_pipe}, X, y, subjects, labels)
    print(f"  그룹평균 acc={nl['per_subject_mean']:.4f}  macro-F1={nl['macro_f1']:.4f}")

    # 환경 robustness: JEONG(맨발) 테스트. 단 SEUNG(=JEONG과 동일인)을 학습에서 빼야
    # '처음 보는 사람 + 맨발'의 순수 환경차가 측정된다. SEUNG을 두면 동일인 누설(0.99).
    print("\n## 환경 robustness 테스트 (SEUNG=동일인 학습 제외)")
    no_seung = subjects != "SEUNG"
    pipe = _baseline_pipe(); pipe.fit(X[no_seung], y[no_seung])   # JIHO+JUHO+SHIM 만
    env_acc = float(accuracy_score(ye, pipe.predict(Xe)))
    print(f"  학습=JIHO+JUHO+SHIM(신발) → JEONG(맨발) acc={env_acc:.4f}")
    # 대조군: 같은 3명 → SEUNG(신발, 동일인의 신발버전) — 환경만 다른 비교
    seung_mask = subja == "SEUNG"
    seung_acc = float(accuracy_score(ya[seung_mask], pipe.predict(Xa[seung_mask])))
    print(f"  대조) 같은 학습 → SEUNG(신발, 동일인) acc={seung_acc:.4f}")
    print(f"  → 맨발({env_acc:.2f}) vs 신발({seung_acc:.2f}) 차이 = 같은 사람에서 환경차 영향")

    print("\n## 최저 per-class F1 5개 (신발4 LOSO 기준 — 혼동 큰 클래스)")
    for lab, f in sorted(lo["per_class_f1"].items(), key=lambda x: x[1])[:5]:
        print(f"    {lab:16s} F1={f:.3f}")
    print("\n# 엔진 검증 끝. (Phase 3 에서 후보 여러 개를 nested_loso 에 투입)")


if __name__ == "__main__":
    main()
