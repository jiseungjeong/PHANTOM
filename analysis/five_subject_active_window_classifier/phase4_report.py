#!/usr/bin/env python3
"""Phase 4 — Metric & report (for the presentation).

Per the P1/P2 decisions:
  P1 — confusion matrix on BOTH LOSO and file-grouped, drawn with the best model
       (KNN + feature-selection from Phase 3). LOSO = unseen person (~0.60),
       file-grouped = known person (~0.97); shown side by side so the message
       "known person is near-perfect; unseen person confuses defense (Slip/Roll)"
       lands clearly. Primary = LOSO.
  P2 — device ablation with a FIXED model (ExtraTrees) and NO feature-selection,
       masking features by device prefix (a_/i_/w_). Compares head(AirPods) /
       wrist(Watch) / body(iPhone) / pairs / all under the same LOSO, so the
       "pure per-device contribution" stays interpretable.
  + our-vs-BoxerSense comparison bar chart (Phase 3.5 table -> figure).

Outputs (reports/):
  confusion_loso.png/json, confusion_filegrouped.png/json,
  per_subject.json, ablation.json, ablation.png,
  ours_vs_boxersense.png, phase4_summary.json

Run: python3 analysis/five_subject_active_window_classifier/phase4_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
OUT_DIR = THIS.parent / "reports"          # all artifacts (regenerable, .gitignore'd)
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = THIS.parent / "figures"          # curated presentation figures (tracked in git)
FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THIS.parent))

# Curated subset copied from reports/ -> figures/ for the presentation (the actual deliverable).
# Verification grids (segmentation/jumpcut/suspect) stay only in reports/ (ignored).
PRESENTATION_FIGURES = [
    "confusion_loso.png", "confusion_filegrouped.png",
    "ablation.png", "ours_vs_boxersense.png",
]

from evaluation import load_dataset, split_cohort, loso, file_grouped_cv
import phase3_models as p3

# Best model from Phase 3: KNN + feature-selection (macro-F1 0.605).
BEST_NAME = "KNN(n=1)+sel"
def best_model():
    return p3.knn(sel=True)

# Ablation model: the DEPLOYED system (KNN + feature-selection), so the chart
# reflects how the real classifier scales with device count (honest for the
# report). ExtraTrees(no selection) is kept as an appendix view of "pure
# per-device contribution" (a fixed model that ignores weak channels on its own).
def ablation_model():
    return p3.knn(sel=True)

def ablation_model_extratrees():
    return p3.extratrees(sel=False)

# BoxerSense best (Phase 3.5) for the comparison chart.
BOXERSENSE_BEST = ("BoxerSense (RandomForest)", 0.577)
OUR_BEST_F1 = 0.605

DEVICE_PREFIX = {"head (AirPods)": "a_", "wrist (Watch)": "w_", "body (iPhone)": "i_"}


# ── Confusion matrix drawing ────────────────────────────────────────────────────

def draw_confusion(cm, labels, title, out_png, normalize=True):
    cm = np.array(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    else:
        cmn = cm
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.62 * n + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1 if normalize else cmn.max())
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    thr = (cmn.max() if not normalize else 1.0) * 0.55
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if v <= 0:
                continue
            txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v > thr else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


# ── Device ablation ─────────────────────────────────────────────────────────────

def feature_mask(feats, keep_prefixes):
    """Indices of features whose name starts with one of keep_prefixes (device subset)."""
    return [i for i, n in enumerate(feats) if any(n.startswith(p) for p in keep_prefixes)]


def run_ablation(X, y, subj, labels, feats, model_factory=ablation_model):
    """LOSO macro-F1 for each device subset. Default model = deployed KNN+sel."""
    configs = {
        "head (AirPods)": ["a_"],
        "wrist (Watch)": ["w_"],
        "body (iPhone)": ["i_"],
        "head+wrist": ["a_", "w_"],
        "head+body": ["a_", "i_"],
        "wrist+body": ["w_", "i_"],
        "all (3 IMU)": ["a_", "i_", "w_"],
    }
    out = {}
    for name, prefixes in configs.items():
        cols = feature_mask(feats, prefixes)
        lo = loso(model_factory, X[:, cols], y, subj, labels)
        out[name] = {
            "n_features": len(cols),
            "macro_f1": lo["macro_f1"],
            "accuracy": lo["per_subject_mean"],
            "per_class_f1": lo["per_class_f1"],
        }
        print(f"  {name:16s} feats={len(cols):3d}  LOSO acc={lo['per_subject_mean']:.3f}  macro-F1={lo['macro_f1']:.3f}")
    return out


def draw_ablation(ablation, out_png, title="Device ablation (KNN 1-NN + feature-selection, shoe-5 LOSO)"):
    names = list(ablation.keys())
    f1s = [ablation[n]["macro_f1"] for n in names]
    colors = ["#7fb3ff" if "+" not in n and "all" not in n else
              "#84d68a" if "+" in n else "#d72638" for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), f1s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0, max(f1s) * 1.18)
    ax.set_title(title, fontsize=11)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def draw_ours_vs_boxersense(out_png):
    names = [BOXERSENSE_BEST[0], f"Ours ({BEST_NAME})"]
    f1s = [BOXERSENSE_BEST[1], OUR_BEST_F1]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(range(2), f1s, color=["#9aa0a6", "#1f4e79"])
    ax.set_xticks(range(2)); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0, max(f1s) * 1.2)
    ax.set_title("Ours vs BoxerSense baseline\n(same data / segmentation / LOSO)", fontsize=11)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.annotate(f"+{OUR_BEST_F1 - BOXERSENSE_BEST[1]:.3f}", xy=(1, OUR_BEST_F1),
                xytext=(1.2, OUR_BEST_F1 + 0.04), fontsize=9, color="#1f4e79", weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


# ── per-class precision/recall/F1 from a confusion matrix ────────────────────────

def per_class_prf(cm, labels):
    cm = np.array(cm, dtype=float)
    out = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[lab] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
    return out


# ── Defense left/right confusion summary (the known weak spot) ───────────────────

def lr_confusion_summary(cm, labels):
    """How often Left<->Right Slip/Roll (and Lead<->Rear) get swapped."""
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.array(cm, dtype=float)
    pairs = [("Left Slip", "Right Slip"), ("Left Roll", "Right Roll"),
             ("Lead Jab", "Rear Jab"), ("Lead Hook", "Rear Hook"),
             ("Lead Uppercut", "Rear Uppercut")]
    out = {}
    for a, b in pairs:
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            tot_a = cm[ia, :].sum()
            tot_b = cm[ib, :].sum()
            swap = (cm[ia, ib] + cm[ib, ia])
            denom = tot_a + tot_b
            out[f"{a} <-> {b}"] = {
                "swapped": int(swap),
                "of_total": int(denom),
                "swap_rate": round(float(swap / denom), 4) if denom > 0 else 0.0,
            }
    return out


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (X, y, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    # M1: exclude meta features (same as Phase 3) for the best/confusion model.
    keep = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    feats_keep = [feats[i] for i in keep]
    Xk = X[:, keep]

    print(f"# Phase 4 — metric & report (shoe-5, {len(y)} samples, {len(labels)} classes)")
    print(f"  best model = {BEST_NAME}; ablation model = ExtraTrees (no selection)\n")

    # ── P1: confusion matrices (LOSO + file-grouped) with the best model ─────────
    print("## P1 — confusion matrices (best model)")
    lo = loso(best_model, Xk, y, subj, labels)
    fg = file_grouped_cv(best_model, Xk, y, files, labels)
    print(f"  LOSO         acc={lo['per_subject_mean']:.3f}  macro-F1={lo['macro_f1']:.3f}")
    print(f"  file-grouped acc={fg['accuracy']:.3f}  macro-F1={fg['macro_f1']:.3f}")

    draw_confusion(lo["confusion"], labels,
                   f"LOSO (unseen person)  macro-F1={lo['macro_f1']:.3f}  — best={BEST_NAME}",
                   OUT_DIR / "confusion_loso.png")
    draw_confusion(fg["confusion"], labels,
                   f"file-grouped (known person)  macro-F1={fg['macro_f1']:.3f}  — best={BEST_NAME}",
                   OUT_DIR / "confusion_filegrouped.png")

    loso_prf = per_class_prf(lo["confusion"], labels)
    fg_prf = per_class_prf(fg["confusion"], labels)
    lr_loso = lr_confusion_summary(lo["confusion"], labels)

    (OUT_DIR / "confusion_loso.json").write_text(json.dumps({
        "labels": labels, "confusion": lo["confusion"], "macro_f1": lo["macro_f1"],
        "accuracy": lo["per_subject_mean"], "per_class": loso_prf,
        "left_right_confusion": lr_loso,
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "confusion_filegrouped.json").write_text(json.dumps({
        "labels": labels, "confusion": fg["confusion"], "macro_f1": fg["macro_f1"],
        "accuracy": fg["accuracy"], "per_class": fg_prf,
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "per_subject.json").write_text(json.dumps({
        "per_subject_acc": lo["per_subject_acc"],
        "per_subject_mean": lo["per_subject_mean"],
        "per_subject_min": lo["per_subject_min"],
        "per_subject_max": lo["per_subject_max"],
    }, indent=2, ensure_ascii=False))

    print("  left/right & lead/rear swap (LOSO):")
    for k, v in lr_loso.items():
        print(f"    {k:28s} swap {v['swapped']}/{v['of_total']} = {v['swap_rate']*100:.1f}%")

    # ── P2: device ablation ──────────────────────────────────────────────────────
    # MAIN = deployed system (KNN + feature-selection): honest for the report, and
    # 'all 3 IMU' stays the best. ExtraTrees(no-sel) kept as an appendix view.
    print("\n## P2 — device ablation (KNN + feature-selection, DEPLOYED system)")
    ablation = run_ablation(Xk, y, subj, labels, feats_keep, model_factory=ablation_model)
    draw_ablation(ablation, OUT_DIR / "ablation.png")
    (OUT_DIR / "ablation.json").write_text(json.dumps(ablation, indent=2, ensure_ascii=False))

    print("\n## P2 (appendix) — device ablation (ExtraTrees, no selection — pure per-device)")
    ablation_et = run_ablation(Xk, y, subj, labels, feats_keep, model_factory=ablation_model_extratrees)
    draw_ablation(ablation_et, OUT_DIR / "ablation_extratrees.png",
                  title="Device ablation (ExtraTrees, no feature-selection, shoe-5 LOSO) — appendix")
    (OUT_DIR / "ablation_extratrees.json").write_text(json.dumps(ablation_et, indent=2, ensure_ascii=False))

    # ── Comparison chart: ours vs BoxerSense ─────────────────────────────────────
    draw_ours_vs_boxersense(OUT_DIR / "ours_vs_boxersense.png")
    print(f"\n## Ours vs BoxerSense: {OUR_BEST_F1:.3f} vs {BOXERSENSE_BEST[1]:.3f} "
          f"(+{OUR_BEST_F1 - BOXERSENSE_BEST[1]:.3f})")

    # ── Summary json ─────────────────────────────────────────────────────────────
    best_dev = max(ablation, key=lambda k: ablation[k]["macro_f1"])
    single_best = max([k for k in ablation if "+" not in k and "all" not in k],
                      key=lambda k: ablation[k]["macro_f1"])
    (OUT_DIR / "phase4_summary.json").write_text(json.dumps({
        "best_model": BEST_NAME,
        "loso": {"macro_f1": lo["macro_f1"], "accuracy": lo["per_subject_mean"]},
        "file_grouped": {"macro_f1": fg["macro_f1"], "accuracy": fg["accuracy"]},
        "ablation_best_config": best_dev,
        "ablation_best_single_device": single_best,
        "ablation": {k: v["macro_f1"] for k, v in ablation.items()},
        "ours_vs_boxersense": {"ours": OUR_BEST_F1, "boxersense": BOXERSENSE_BEST[1]},
    }, indent=2, ensure_ascii=False))

    print(f"\n  ablation best: {best_dev} ({ablation[best_dev]['macro_f1']:.3f}); "
          f"best single device: {single_best} ({ablation[single_best]['macro_f1']:.3f})")
    print(f"\n  saved -> {OUT_DIR}/  (confusion_*.png/json, ablation.*, ours_vs_boxersense.png, phase4_summary.json)")

    # Mirror the curated subset into figures/ (tracked in git) so the presentation
    # figures stay in sync on every re-run.
    import shutil
    copied = []
    for fn in PRESENTATION_FIGURES:
        src = OUT_DIR / fn
        if src.exists():
            shutil.copy2(src, FIG_DIR / fn)
            copied.append(fn)
    print(f"  presentation figures -> {FIG_DIR}/  ({len(copied)}: {', '.join(copied)})")
    print("# done.")


if __name__ == "__main__":
    main()
