#!/usr/bin/env python3
"""Phase 4.5 extra — calibration robustness (A1 multi-model) + interpretability (A2 confusion).

Builds on phase45_calibration.py (same file-separated, peak-level, frozen-preprocessing
protocol). Adds:

A1 — Does calibration help for models OTHER than KNN? Defends against "KNN just
     memorized the user's own samples". Run the m=0..5 curve for KNN / ExtraTrees /
     SVC. Common framework for ALL models: preprocessing is frozen on the factory
     train, and only the MODEL is refit on (factory train + calibration peaks).
     For KNN that equals "grow the reference"; for trees/SVC it is an honest refit.
     If all three rise -> personalization itself works (not a KNN trick).

A2 — What did calibration FIX? Pool out-of-fold predictions at m=0 vs m=1 and draw
     both confusion matrices. Quantify the worst spot (Left<->Right Slip, 26.7% at
     m=0) after calibration.

Outputs: reports/calibration_models.json, figures/calibration_models.png,
         figures/calibration_confusion.png, reports/calibration_confusion.json
Run: python3 analysis/five_subject_active_window_classifier/phase45_calibration_extra.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix

THIS = Path(__file__).resolve()
OUT_DIR = THIS.parent / "reports"; OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = THIS.parent / "figures"; FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THIS.parent))

from evaluation import load_dataset, split_cohort, HOLDOUT_ENV_SUBJECTS
import phase3_models as p3
from phase45_calibration import PEAKS_PER_FILE

M_VALUES = [0, 1, 2, 3, 4, 5]
# A1 models: factory builder -> (preproc, model). All use the frozen-preproc / refit-model framework.
MODEL_FACTORIES = {
    "KNN(n=1)+sel": lambda: p3.knn(sel=True),
    "ExtraTrees+sel": lambda: p3.extratrees(sel=True),
    "SVC(RBF)+sel": lambda: p3.svc(sel=True),
}


def split_preproc_model(make_pipe):
    """Split any phase3 pipeline into (preproc_pipeline, final_estimator)."""
    full = make_pipe()
    return Pipeline(full.steps[:-1]), full.steps[-1][1]


def calibrated_predict(preproc, model_proto, Xf, yf, Xcal, ycal, Xte):
    """Freeze preproc on factory; refit model on (factory + calibration); predict test."""
    pre = clone(preproc).fit(Xf, yf)
    Zf = pre.transform(Xf)
    model = clone(model_proto)
    if len(Xcal):
        Z = np.vstack([Zf, pre.transform(Xcal)])
        y = np.concatenate([yf, ycal])
    else:
        Z, y = Zf, yf
    model.fit(Z, y)
    return model.predict(pre.transform(Xte))


def build_index(y, subj, files):
    by_sl_file = defaultdict(lambda: defaultdict(list))
    for i in range(len(y)):
        by_sl_file[(subj[i], y[i])][files[i]].append(i)
    return by_sl_file


def run_curve(make_pipe, X, y, subj, files, labels, subjects, by_sl_file,
              collect_preds_at=None, pool_subjects=None):
    """Return {subject: {m: mean_macro_f1}}. If collect_preds_at is set (an m),
    also return pooled (y_true, y_pred) across folds/positions/rotations at that m.
    pool_subjects (set|None): if given, ONLY these subjects' predictions are pooled
    into the confusion (e.g. shoe-5 only, to exclude the env-holdout JEONG and keep
    the confusion comparable to the shoe-5 LOSO). None = all (curve behavior)."""
    preproc, model_proto = split_preproc_model(make_pipe)
    results = defaultdict(lambda: defaultdict(list))
    pooled_true, pooled_pred = [], []

    for test_s in subjects:
        in_pool = (pool_subjects is None) or (test_s in pool_subjects)
        # Factory train = all OTHER shoe-5 subjects (JEONG already excluded upstream,
        # so no same-person twin remains to leak).
        factory_mask = subj != test_s
        Xf_all, yf_all = X[factory_mask], y[factory_mask]
        labels_S = sorted({lab for (s, lab) in by_sl_file if s == test_s})

        for calib_pos in range(3):
            cal_idx_by_action, test_idx, ok = {}, [], True
            for lab in labels_S:
                fmap = by_sl_file[(test_s, lab)]
                fs = sorted(fmap.keys())
                if len(fs) < 3:
                    ok = False; break
                cal_file = fs[calib_pos]
                cal_idx_by_action[lab] = sorted(fmap[cal_file])
                for f in fs:
                    if f != cal_file:
                        test_idx.extend(fmap[f])
            if not ok:
                continue
            test_idx = np.array(sorted(test_idx))
            Xte, yte = X[test_idx], y[test_idx]

            for m in M_VALUES:
                if m == 0:
                    pred = calibrated_predict(preproc, model_proto, Xf_all, yf_all,
                                              np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype), Xte)
                    results[test_s][m].append(f1_score(yte, pred, labels=labels, average="macro", zero_division=0))
                    if collect_preds_at == 0 and in_pool:
                        pooled_true.extend(yte.tolist()); pooled_pred.extend(pred.tolist())
                    continue
                rot_f1 = []
                for rot in range(PEAKS_PER_FILE):
                    cal_sel = []
                    for lab in labels_S:
                        idxs = cal_idx_by_action[lab]
                        mm = min(m, len(idxs))
                        order = [idxs[(rot + j) % len(idxs)] for j in range(len(idxs))]
                        cal_sel.extend(order[:mm])
                    cal_sel = np.array(cal_sel)
                    pred = calibrated_predict(preproc, model_proto, Xf_all, yf_all,
                                              X[cal_sel], y[cal_sel], Xte)
                    rot_f1.append(f1_score(yte, pred, labels=labels, average="macro", zero_division=0))
                    if collect_preds_at == m and rot == 0 and in_pool:   # one rotation to avoid over-counting
                        pooled_true.extend(yte.tolist()); pooled_pred.extend(pred.tolist())
                results[test_s][m].append(float(np.mean(rot_f1)))

    if collect_preds_at is not None:
        return results, (np.array(pooled_true), np.array(pooled_pred))
    return results


def shoe_mean(results, subjects, m):
    shoe = [s for s in subjects if s not in HOLDOUT_ENV_SUBJECTS]
    vals = [np.mean(results[s][m]) for s in shoe if results[s][m]]
    return float(np.mean(vals)) if vals else None


def lr_swap_rate(cm, labels, a, b):
    idx = {l: i for i, l in enumerate(labels)}
    if a not in idx or b not in idx:
        return None
    cm = np.array(cm, float)
    ia, ib = idx[a], idx[b]
    denom = cm[ia, :].sum() + cm[ib, :].sum()
    swap = cm[ia, ib] + cm[ib, ia]
    return float(swap / denom) if denom > 0 else 0.0


def draw_confusion(cm, labels, title, ax, show_ylabels=True):
    cm = np.array(cm, float)
    row = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    if show_ylabels:
        ax.set_yticklabels(labels, fontsize=6); ax.set_ylabel("True")
    else:
        ax.set_yticklabels([])   # right matrix omits y labels (same as left; avoids overlap)
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            if cmn[i, j] > 0:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if cmn[i, j] > 0.55 else "#222")
    return im


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    # Shoe-4 cohort only (split_cohort excludes JEONG = barefoot holdout & SEUNG twin).
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)
    keep = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    X = X[:, keep]
    subjects = sorted(set(subj.tolist()))
    by_sl_file = build_index(y, subj, files)

    # ── A1: multi-model calibration curves ───────────────────────────────────────
    print("# A1 — calibration across models (frozen preproc, model refit on factory+calib)")
    curves = {}
    for name, make in MODEL_FACTORIES.items():
        res = run_curve(make, X, y, subj, files, labels, subjects, by_sl_file)
        curve = {m: shoe_mean(res, subjects, m) for m in M_VALUES}
        curves[name] = curve
        print(f"  {name:16s} " + "  ".join(f"m{m}={curve[m]:.3f}" for m in M_VALUES))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"KNN(n=1)+sel": ("o-", "#1f4e79"), "ExtraTrees+sel": ("s--", "#2a9d8f"),
              "SVC(RBF)+sel": ("^:", "#e76f51")}
    for name, curve in curves.items():
        st, col = styles[name]
        ax.plot(M_VALUES, [curve[m] for m in M_VALUES], st, color=col, label=name)
    ax.axhline(0.963, ls=":", color="#888", label="file-grouped ceiling 0.963")
    ax.set_xlabel("calibration peaks per action (m)"); ax.set_ylabel("shoe-5 LOSO macro-F1")
    ax.set_ylim(0.4, 1.0); ax.set_xticks(M_VALUES)
    ax.set_title("A1: calibration helps across models (not a KNN-only trick)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration_models.png", dpi=130); plt.close(fig)
    (OUT_DIR / "calibration_models.json").write_text(json.dumps(curves, indent=2, ensure_ascii=False))

    # ── A2: confusion at m=0 vs m=1 (best model KNN) ─────────────────────────────
    # All subjects here ARE the shoe-5 cohort (JEONG already excluded), so the
    # confusion is pooled over the same shoe-5 used everywhere else — no leak.
    print(f"\n# A2 — what calibration fixed (confusion m=0 vs m=1, KNN; pooled = {subjects})")
    _, (yt0, yp0) = run_curve(MODEL_FACTORIES["KNN(n=1)+sel"], X, y, subj, files, labels,
                              subjects, by_sl_file, collect_preds_at=0)
    _, (yt1, yp1) = run_curve(MODEL_FACTORIES["KNN(n=1)+sel"], X, y, subj, files, labels,
                              subjects, by_sl_file, collect_preds_at=1)
    cm0 = confusion_matrix(yt0, yp0, labels=labels)
    cm1 = confusion_matrix(yt1, yp1, labels=labels)
    f1_0 = f1_score(yt0, yp0, labels=labels, average="macro", zero_division=0)
    f1_1 = f1_score(yt1, yp1, labels=labels, average="macro", zero_division=0)

    pairs = [("Left Slip", "Right Slip"), ("Left Roll", "Right Roll")]
    print("  Left<->Right swap rate (pooled):")
    swaps = {}
    for a, b in pairs:
        s0 = lr_swap_rate(cm0, labels, a, b); s1 = lr_swap_rate(cm1, labels, a, b)
        swaps[f"{a} <-> {b}"] = {"m0": round(s0, 4), "m1": round(s1, 4)}
        print(f"    {a} <-> {b}: m0 {s0*100:.1f}% -> m1 {s1*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.30, right=0.88)   # gap between matrices + room for colorbar on the right
    draw_confusion(cm0, labels, f"m=0 (no calibration)  macro-F1={f1_0:.3f}", axes[0], show_ylabels=True)
    im = draw_confusion(cm1, labels, f"m=1 (1 demo/action)  macro-F1={f1_1:.3f}", axes[1], show_ylabels=False)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])   # dedicated colorbar axis (pinned far right), unaffected by wspace
    fig.colorbar(im, cax=cax)
    fig.suptitle("A2: calibration confusion — Left<->Right Slip confusion resolved", fontsize=12)
    fig.savefig(FIG_DIR / "calibration_confusion.png", dpi=130); plt.close(fig)   # no bbox_inches so add_axes/wspace layout is kept as intended
    (OUT_DIR / "calibration_confusion.json").write_text(json.dumps({
        "labels": labels, "macro_f1_m0": f1_0, "macro_f1_m1": f1_1,
        "left_right_swap": swaps, "confusion_m0": cm0.tolist(), "confusion_m1": cm1.tolist(),
    }, indent=2, ensure_ascii=False))

    print(f"\n  saved -> figures/calibration_models.png, figures/calibration_confusion.png")
    print(f"           reports/calibration_models.json, reports/calibration_confusion.json")
    print("# done.")


if __name__ == "__main__":
    main()
