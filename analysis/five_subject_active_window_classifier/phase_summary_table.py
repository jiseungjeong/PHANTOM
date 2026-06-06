"""Summary accuracy table: (single-test / combo-test) x (calib 0 / calib 1 single peak)
x (combo windows in train: no / yes). 8 cells, all LOSO by subject, macro-F1.

Calibration protocol (file-separated, honest — same as phase45_calibration):
  the test subject's single-action files split into 1 calib file + 2 test files;
  m=1 calibration adds ONE peak per action (from the calib file) to the KNN
  reference; the single-test set is the 2 test files (never mixed with calib).
  Combo windows for that subject form the combo-test set.

Train variants:
  - "no combo"   : factory = OTHER subjects' single actions.
  - "with combo" : factory = OTHER subjects' single actions + OTHER subjects'
                   selected combo windows (the B / Phase 7 (b) augmentation).
Calibration always adds the test subject's SINGLE peaks only (the user demos
single actions), matching the deployed calibration flow.

Run: python3 analysis/five_subject_active_window_classifier/phase_summary_table.py
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from sklearn.base import clone

import phase3_models as p3
from evaluation import load_dataset, split_cohort

THIS = Path(__file__).resolve()
COMBO_NPZ = THIS.parent / "reports" / "combo_windows.npz"
CONF_THR = 1.0     # selected combo windows (matches B export / Phase 7 e2s)
CALIB_FILE_POSITIONS = 3   # average over which of the 3 files is the calib file


def main() -> None:
    Xa, ya, subja, filesa, feats, labels = load_dataset()
    (Xs, ys, subj, files), _ = split_cohort(Xa, ya, subja, filesa)

    combo = np.load(COMBO_NPZ, allow_pickle=True)
    Xc_all = combo["X"]; yc = combo["y"].astype(str); sc = combo["subject"].astype(str)
    cfeats = [str(f) for f in combo["feature_names"]]
    cconf = combo["conf"].astype(float) if "conf" in combo.files else np.ones(len(yc))

    # common features (single column order, drop meta) — identical space for both
    common = [f for f in feats if f in set(cfeats) and f not in p3.META_FEATURES]
    s_idx = [feats.index(f) for f in common]
    c_idx = [cfeats.index(f) for f in common]
    Xs = Xs[:, s_idx]; Xc = Xc_all[:, c_idx]
    sel = cconf >= CONF_THR

    preproc, knn_proto = p3.split_preproc_knn(sel=False) if hasattr(p3, "split_preproc_knn") else (None, None)
    # build preproc/knn the same way phase45 does
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import RobustScaler, Normalizer
    from sklearn.neighbors import KNeighborsClassifier
    def make_pre():
        return Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                         ("var", VarianceThreshold()),
                         ("scl", RobustScaler()), ("nrm", Normalizer("l2"))])
    def make_knn():
        return KNeighborsClassifier(n_neighbors=1, weights="distance", metric="manhattan")

    def predict(Xf, yf, Xcal, ycal, Xte):
        pre = make_pre().fit(Xf, yf)
        Zf = pre.transform(Xf)
        if len(Xcal):
            Z = np.vstack([Zf, pre.transform(Xcal)]); yy = np.concatenate([yf, ycal])
        else:
            Z, yy = Zf, yf
        knn = make_knn().fit(Z, yy)
        return knn.predict(pre.transform(Xte))

    subjects = sorted(set(subj.tolist()))
    by_sl_file = defaultdict(lambda: defaultdict(list))
    for i in range(len(ys)):
        by_sl_file[(subj[i], ys[i])][files[i]].append(i)

    M_VALUES = [0, 1, 2, 3, 4, 5]   # calibration peaks per action
    # accumulate predictions per cell: (test=single|combo, train=base|combo, m)
    cells = {(te, tr, m): {"true": [], "pred": []}
             for te in ("single", "combo")
             for tr in ("base", "combo")
             for m in M_VALUES}

    for test_s in subjects:
        other_single = subj != test_s
        Xf_single, yf_single = Xs[other_single], ys[other_single]
        other_combo = (sc != test_s) & sel
        Xf_combo, yf_combo = Xc[other_combo], yc[other_combo]
        # factory variants
        train = {
            "base":  (Xf_single, yf_single),
            "combo": (np.vstack([Xf_single, Xf_combo]), np.concatenate([yf_single, yf_combo])),
        }
        # combo test set for this subject
        c_te = sc == test_s
        Xc_te, yc_te = Xc[c_te], yc[c_te]

        labels_S = sorted({lab for (s, lab) in by_sl_file if s == test_s})
        # average over calib-file position
        for calib_pos in range(CALIB_FILE_POSITIONS):
            cal_idx, test_idx, ok = {}, [], True
            for lab in labels_S:
                fmap = by_sl_file[(test_s, lab)]; fs = sorted(fmap.keys())
                if len(fs) < 3: ok = False; break
                cal_idx[lab] = fmap[fs[calib_pos]]
                for j in range(3):
                    if j != calib_pos: test_idx += fmap[fs[j]]
            if not ok: continue
            test_idx = np.array(test_idx)
            Xsingle_te, ysingle_te = Xs[test_idx], ys[test_idx]

            # calibration set for each m: first m peaks per action from the calib file.
            # m=0 -> empty. (calib file has up to 5 peaks/action.)
            def calib_set(m):
                rows, lab = [], []
                for L in labels_S:
                    for idx in cal_idx[L][:m]:
                        rows.append(Xs[idx]); lab.append(L)
                X = np.array(rows) if rows else np.empty((0, Xs.shape[1]))
                return X, np.array(lab)

            for tr in ("base", "combo"):
                Xf, yf = train[tr]
                for m in M_VALUES:
                    Xc_use, yc_use = calib_set(m)
                    # single test
                    p_s = predict(Xf, yf, Xc_use, yc_use, Xsingle_te)
                    cells[("single", tr, m)]["true"] += list(ysingle_te)
                    cells[("single", tr, m)]["pred"] += list(p_s)
                    # combo test
                    if len(Xc_te):
                        p_c = predict(Xf, yf, Xc_use, yc_use, Xc_te)
                        cells[("combo", tr, m)]["true"] += list(yc_te)
                        cells[("combo", tr, m)]["pred"] += list(p_c)

    def f1(cell):
        d = cells[cell]
        return f1_score(d["true"], d["pred"], average="macro") if d["true"] else float("nan")

    print("# Summary table — LOSO macro-F1  (calib learning curve m=0..5)\n")
    hdr = "  " + f"{'test / train':18}" + "".join(f"{'m'+str(m):>7}" for m in M_VALUES)
    for tr in ("base", "combo"):
        tname = "single-train" if tr == "base" else "single+combo-train"
        print(f"== {tname} ==")
        print(hdr)
        for te in ("single", "combo"):
            row = "".join(f"{f1((te,tr,m)):7.3f}" for m in M_VALUES)
            print(f"  {te+' test':18}{row}")
        print()
    print("  m0=no calib; m1..m5 = +m single-action peaks/action (test subject, file-separated)")
    print("  single test = 2 held test files;  combo test = subject's combo windows")


if __name__ == "__main__":
    main()
