"""Phase 7 (a)/(b) — transfer to combo (fast action transitions).

Purpose chain (plan §C2): the combo segmentation (anchor-first) was a means to an
end — turn combos into per-action windows and use them to test/strengthen the
classifier on FAST action transitions. This script measures:

  (a) single-trained -> combo-tested: how much the single-only KNN drops on combo
      windows (C2 quantified).
  (b) single + combo-trained -> combo-tested: does adding combo windows to the KNN
      reference improve combo classification (= the goal, "fast-transition aware").

LOSO by SUBJECT: the test subject's data (single AND combo) is held out of training.
Common feature set between the single dataset and combo_windows.npz. Same frozen
preprocessing as the deployed KNN (SimpleImputer->VarianceThreshold->RobustScaler->
Normalizer->1-NN manhattan, distance-weighted).

Run: python3 analysis/five_subject_active_window_classifier/phase7_transfer.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

import phase3_models as p3
from evaluation import load_dataset, split_cohort

THIS = Path(__file__).resolve()
COMBO_NPZ = THIS.parent / "reports" / "combo_windows.npz"


def main() -> None:
    # single-action data (shoe-4 cohort, same as all reported metrics)
    Xs, ys, ss, fs, sfeats, _ = load_dataset()
    (Xs, ys, ss, fs), _ = split_cohort(Xs, ys, ss, fs)

    combo = np.load(COMBO_NPZ, allow_pickle=True)
    Xc_all = combo["X"]; yc = combo["y"].astype(str)
    sc = combo["subject"].astype(str); cfeats = [str(f) for f in combo["feature_names"]]
    cconf = combo["conf"].astype(float) if "conf" in combo.files else np.ones(len(yc))

    # align to common features (column order from the single dataset)
    common = [f for f in sfeats if f in set(cfeats)]
    s_idx = [sfeats.index(f) for f in common]
    c_idx = [cfeats.index(f) for f in common]
    Xs = Xs[:, s_idx]
    Xc = Xc_all[:, c_idx]
    print(f"# Phase 7 transfer — common features {len(common)}")
    print(f"  single (shoe-4): {len(ys)}   combo windows: {len(yc)}")

    subjects = sorted(set(sc.tolist()))
    # restrict combo labels to those the single classifier knows (all 10 are known)
    a_true, a_pred = [], []
    b_true, b_pred = [], []
    for test_subj in subjects:
        # combo test set = this subject's combo windows
        te = sc == test_subj
        if te.sum() == 0:
            continue
        Xc_te, yc_te = Xc[te], yc[te]

        # (a) train on ALL single (minus test subject's single), test on combo
        s_tr = ss != test_subj
        knn_a = p3.knn(sel=False).fit(Xs[s_tr], ys[s_tr])
        a_pred.extend(knn_a.predict(Xc_te)); a_true.extend(yc_te)

        # (b) train on single(minus test) + combo(minus test subject), test on combo
        c_tr = sc != test_subj
        Xb = np.vstack([Xs[s_tr], Xc[c_tr]])
        yb = np.concatenate([ys[s_tr], yc[c_tr]])
        knn_b = p3.knn(sel=False).fit(Xb, yb)
        b_pred.extend(knn_b.predict(Xc_te)); b_true.extend(yc_te)

    a_f1 = f1_score(a_true, a_pred, average="macro")
    b_f1 = f1_score(b_true, b_pred, average="macro")
    a_acc = np.mean(np.array(a_true) == np.array(a_pred))
    b_acc = np.mean(np.array(b_true) == np.array(b_pred))
    print("\n## LOSO (by subject), tested on combo windows")
    print(f"  (a) single-only train      : macro-F1 {a_f1:.3f}  acc {a_acc:.3f}")
    print(f"  (b) single + combo train   : macro-F1 {b_f1:.3f}  acc {b_acc:.3f}")
    print(f"  improvement (b - a)        : macro-F1 {b_f1 - a_f1:+.3f}  acc {b_acc - a_acc:+.3f}")
    print(f"\n  reference: single-only LOSO on SINGLE test = 0.605 (plan).")
    print(f"  (a) drop vs that = combo transfer gap (C2). (b)-(a) = fast-transition gain.")

    # ---- (e) MIXED test (single + combo) — real-world usage; report 3 splits ----
    # test set per subject = that subject's single AND combo. Tag each test sample as
    # 'single' or 'combo' so we can score: overall / single-part / combo-part. The
    # single-part of (e2) vs (e1) shows whether adding combo HURTS single recognition.
    def loso_mixed(use_combo_in_train, conf_thr=0.0):
        # conf_thr: only combo TRAIN windows with dominant-sensor conf >= thr are used
        # (selection — drop noisy combo labels). Test set is unchanged (all single+combo).
        true_all, pred_all, tag_all = [], [], []
        for test_subj in subjects:
            s_te = ss == test_subj
            c_te = sc == test_subj
            if s_te.sum() == 0 and c_te.sum() == 0:
                continue
            s_tr = ss != test_subj
            if use_combo_in_train:
                c_tr = (sc != test_subj) & (cconf >= conf_thr)
                Xtr = np.vstack([Xs[s_tr], Xc[c_tr]])
                ytr = np.concatenate([ys[s_tr], yc[c_tr]])
            else:
                Xtr, ytr = Xs[s_tr], ys[s_tr]
            knn = p3.knn(sel=False).fit(Xtr, ytr)
            # single part of test
            if s_te.sum():
                pred_all.extend(knn.predict(Xs[s_te])); true_all.extend(ys[s_te])
                tag_all.extend(["single"] * int(s_te.sum()))
            # combo part of test
            if c_te.sum():
                pred_all.extend(knn.predict(Xc[c_te])); true_all.extend(yc[c_te])
                tag_all.extend(["combo"] * int(c_te.sum()))
        t = np.array(true_all); p = np.array(pred_all); g = np.array(tag_all)
        def f1of(mask):
            return f1_score(t[mask], p[mask], average="macro") if mask.any() else float("nan")
        return f1of(np.ones(len(t), bool)), f1of(g == "single"), f1of(g == "combo")

    e1 = loso_mixed(False)            # single-only train
    e2 = loso_mixed(True)             # single + ALL combo train
    # selection: keep only high-confidence combo windows (median conf threshold)
    thr = float(np.median(cconf))
    kept = int((cconf >= thr).sum())
    e2s = loso_mixed(True, conf_thr=thr)
    print("\n## (e) MIXED test (single + combo) — real-world usage, LOSO, macro-F1")
    print(f"  {'':30} {'overall':>8} {'single':>8} {'combo':>8}")
    print(f"  (e1) single-only train       : {e1[0]:8.3f} {e1[1]:8.3f} {e1[2]:8.3f}")
    print(f"  (e2) single + ALL combo      : {e2[0]:8.3f} {e2[1]:8.3f} {e2[2]:8.3f}")
    print(f"  (e2s) single + SELECTED combo: {e2s[0]:8.3f} {e2s[1]:8.3f} {e2s[2]:8.3f}   "
          f"(conf>={thr:.2f}, {kept}/{len(cconf)} combo kept)")
    print(f"  delta e2  - e1               : {e2[0]-e1[0]:+8.3f} {e2[1]-e1[1]:+8.3f} {e2[2]-e1[2]:+8.3f}")
    print(f"  delta e2s - e1               : {e2s[0]-e1[0]:+8.3f} {e2s[1]-e1[1]:+8.3f} {e2s[2]-e1[2]:+8.3f}")
    print(f"  -> selection goal: single-part hit smaller than e2, combo-part still up.")

    # ---- presentation figure: (a) vs (b) macro-F1 on combo windows ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ["Single-only\ntrain", "Single + Combo\ntrain"]
    vals = [a_f1, b_f1]
    colors = ["#c0654a", "#3b7a57"]
    xb = ax.bar(bars, vals, color=colors, width=0.55)
    for r, v in zip(xb, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.annotate(f"+{b_f1 - a_f1:.3f}", xy=(1, b_f1), xytext=(0.5, max(vals) + 0.08),
                ha="center", fontsize=12, color="#3b7a57", fontweight="bold")
    ax.set_ylabel("macro-F1  (combo windows, LOSO)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Fast-transition (combo) recognition\nadding combo windows to the KNN reference",
                 fontsize=11)
    ax.text(0.5, -0.22, "Same combo test set. Combo windows are noisier (seg labels) — honest lower bound.",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout()
    out = THIS.parent / "reports" / "phase7_transfer_bar.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out.name}")


if __name__ == "__main__":
    main()
