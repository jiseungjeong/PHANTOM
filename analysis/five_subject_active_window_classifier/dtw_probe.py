#!/usr/bin/env python3
"""DTW probe (stage 1) — does DTW beat our KNN on DEFENSE actions?

Light, fair check inspired by Thomas et al. (UbiComp'24, earable boxing defense,
DTW+DBA). Question: on the SAME shoe-5 LOSO, same 4 defense classes, does 1-NN DTW
on raw windowed time-series beat our feature-based KNN?

Fairness: both restricted to the 4 defense classes (Left/Right Slip, Left/Right
Roll), shoe-5, LOSO. Not compared against the 10-class KNN.

Time-series: re-segment defense files with phase1 (choose_real_peaks +
uniform_windows), pull each window's raw signal. Axis configs tried:
  - "head" : AirPods gyro_y + acc_x   (paper's choice)
  - "norm3": per-sensor accel-magnitude + gyro-magnitude (Watch/iPhone/AirPods)
DTW: 1-NN over all train windows (no DBA yet — added only if 1-NN shows promise).
Manhattan pointwise cost, Sakoe-Chiba band for speed.

KNN baseline: phase3 knn(sel=False) on our features, restricted to the 4 classes.

Run: python3 analysis/five_subject_active_window_classifier/dtw_probe.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import time
import numpy as np
from sklearn.metrics import f1_score

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
LEGACY = REPO / "analysis" / "common"
sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"] = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp
sys.path.insert(0, str(THIS.parent))
import phase1_segment as p1

DEFENSE = ["Left Slip", "Right Slip", "Left Roll", "Right Roll"]
SHOE = ["seung", "jiho", "juho", "shim"]
RESAMPLE_N = 40          # resample each window to fixed length for stable DTW
BAND = 8                 # Sakoe-Chiba band radius


# ── time-series extraction ──────────────────────────────────────────────────────

def _resample(sig: np.ndarray, n: int) -> np.ndarray:
    if len(sig) < 2:
        return np.zeros((n, sig.shape[1]) if sig.ndim > 1 else n)
    xs = np.linspace(0, 1, len(sig))
    xt = np.linspace(0, 1, n)
    if sig.ndim == 1:
        return np.interp(xt, xs, sig)
    return np.stack([np.interp(xt, xs, sig[:, c]) for c in range(sig.shape[1])], axis=1)


def _zscore(series: np.ndarray) -> np.ndarray:
    """Per-channel z-score (L1 fix): remove accel-vs-gyro scale imbalance so no
    single channel dominates the DTW Manhattan cost. std=0 channels left as 0."""
    mu = series.mean(axis=0, keepdims=True)
    sd = series.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (series - mu) / sd


def _window_series(data, lo, hi, axis_cfg, jump):
    """Extract the multivariate series for [lo,hi] under an axis config, resampled,
    and per-channel z-scored.

    [lo,hi] are RELATIVE seconds (jump-cut origin), matching motion_envelope/windows.
    Raw rows carry ABSOLUTE timestamps, and motion_envelope's relative axis is
    (row.t - source_t0 - jump) PER SOURCE. So we convert each source's rows to the
    same relative axis before filtering (fixes the all-zeros time-axis bug)."""
    def src_rows(name):
        raw = data.get(name, [])
        if not raw:
            return np.empty((0, 7))
        t0 = raw[0][0]                                   # per-source first timestamp
        out = []
        for r in raw:
            rel = r[0] - t0 - jump                       # same axis as windows
            if lo <= rel <= hi:
                out.append((rel, *r[1:]))
        return np.array(out, dtype=float) if out else np.empty((0, 7))

    if axis_cfg == "head":
        a = src_rows("AirPods")            # cols: t, ax,ay,az, gx,gy,gz
        if len(a) < 2:
            return None
        gy = a[:, 5]; ax = a[:, 1]         # gyro_y, acc_x (paper)
        return _zscore(_resample(np.stack([gy, ax], axis=1), RESAMPLE_N))

    if axis_cfg == "norm3":
        chans = []
        for name in ("Watch", "iPhone", "AirPods"):
            r = src_rows(name)
            if len(r) < 2:
                chans.append(np.zeros(RESAMPLE_N)); chans.append(np.zeros(RESAMPLE_N)); continue
            amag = np.sqrt((r[:, 1:4] ** 2).sum(axis=1))
            gmag = np.sqrt((r[:, 4:7] ** 2).sum(axis=1))
            chans.append(_resample(amag, RESAMPLE_N))
            chans.append(_resample(gmag, RESAMPLE_N))
        return _zscore(np.stack(chans, axis=1))

    if axis_cfg == "all18":
        # Fairness improvement A: give DTW the full 6 axes x 3 sensors = 18 raw channels
        # (matches KNN seeing the whole 3-IMU signal). Per-channel z-score.
        chans = []
        for name in ("Watch", "iPhone", "AirPods"):
            r = src_rows(name)
            for c in range(1, 7):   # ax,ay,az,gx,gy,gz
                if len(r) < 2:
                    chans.append(np.zeros(RESAMPLE_N))
                else:
                    chans.append(_resample(r[:, c], RESAMPLE_N))
        return _zscore(np.stack(chans, axis=1))
    return None


def collect_windows(axis_cfg, classes, want_files=False):
    """Return series[], labels[], subjects[] (+files[] if want_files) over shoe-5
    single files for `classes`. files[] = source filename per window (for the
    file-separated calibration protocol)."""
    series, labels, subjects, files = [], [], [], []
    for folder in SHOE:
        subj = folder.upper()
        for path in sorted((DATA_DIR / f"{folder}_data").glob("*.csv")):
            if "_Watch_" in path.name:
                continue
            label = p1.label_from_path(path)
            if label not in classes:
                continue
            whole, meta = rbc.extract_features(path)
            if whole is None:
                continue
            jump = float(meta.get("jump_cut_s", 0.0))
            data = rbc.read_csv_sources(path)
            times, env, dur, _ = bp.motion_envelope(data, jump)
            if len(env) == 0:
                continue
            peaks, _ = p1.choose_real_peaks(times, env, dur)
            if len(peaks) < p1.MIN_REPS:
                continue
            wins = p1.uniform_windows(bp.windows_from_peaks(times, env, peaks, dur), dur)
            for w in wins:
                s = _window_series(data, w["start_s"], w["end_s"], axis_cfg, jump)
                if s is None:
                    continue
                series.append(s); labels.append(label); subjects.append(subj); files.append(path.name)
    if want_files:
        return series, np.array(labels), np.array(subjects), np.array(files)
    return series, np.array(labels), np.array(subjects)


# ── DTW (Sakoe-Chiba banded, Manhattan pointwise) ───────────────────────────────

def dtw_dist(a: np.ndarray, b: np.ndarray, band: int = BAND) -> float:
    n, m = len(a), len(b)
    inf = float("inf")
    prev = np.full(m + 1, inf); prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, inf)
        jlo = max(1, i - band); jhi = min(m, i + band)
        for j in range(jlo, jhi + 1):
            cost = float(np.abs(a[i - 1] - b[j - 1]).sum())
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m]


def loso_dtw(series, labels, subjects, classes, k: int = 1):
    """k-NN DTW LOSO over `classes`. k=1 nearest; k>1 majority (tie -> nearest)."""
    subs = sorted(set(subjects.tolist()))
    preds = np.empty(len(labels), dtype=object)
    for s in subs:
        te = np.where(subjects == s)[0]
        tr = np.where(subjects != s)[0]
        for i in te:
            dists = [(dtw_dist(series[i], series[j]), labels[j]) for j in tr]
            dists.sort(key=lambda x: x[0])
            top = dists[:k]
            votes = {}
            for _, lab in top:
                votes[lab] = votes.get(lab, 0) + 1
            best = max(votes.items(), key=lambda kv: (kv[1], -min(d for d, l in top if l == kv[0])))
            preds[i] = best[0]
    return f1_score(labels, preds, labels=classes, average="macro", zero_division=0), preds


# ── DBA (DTW Barycenter Averaging) — paper's template method ─────────────────────

def dtw_path(a, b, band=BAND):
    """DTW with backtrace → list of (i,j) aligned index pairs."""
    n, m = len(a), len(b)
    inf = float("inf")
    D = np.full((n + 1, m + 1), inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost = float(np.abs(a[i - 1] - b[j - 1]).sum())
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        if step == D[i - 1, j - 1]: i, j = i - 1, j - 1
        elif step == D[i - 1, j]:   i -= 1
        else:                       j -= 1
    return path[::-1]


def dba(samples, n_iter=5):
    """Average a set of (fixed-length) series via DTW Barycenter Averaging.
    Start from the medoid, then iteratively DTW-align all to the running average."""
    if len(samples) == 1:
        return samples[0]
    avg = samples[0].copy()
    L = len(avg)
    for _ in range(n_iter):
        acc = np.zeros_like(avg); cnt = np.zeros(L)
        for s in samples:
            for (ia, js) in dtw_path(avg, s):
                acc[ia] += s[js]; cnt[ia] += 1
        cnt[cnt == 0] = 1
        avg = acc / cnt[:, None]
    return avg


def loso_dba(series, labels, subjects, classes):
    """LOSO with ONE DBA template per class (built from train subjects)."""
    subs = sorted(set(subjects.tolist()))
    preds = np.empty(len(labels), dtype=object)
    series = list(series)
    for s in subs:
        te = np.where(subjects == s)[0]
        tr = np.where(subjects != s)[0]
        templates = {}
        for c in classes:
            members = [series[j] for j in tr if labels[j] == c]
            if members:
                templates[c] = dba(members)
        for i in te:
            best_d, best_l = float("inf"), classes[0]
            for c, tmpl in templates.items():
                d = dtw_dist(series[i], tmpl)
                if d < best_d:
                    best_d, best_l = d, c
            preds[i] = best_l
    return f1_score(labels, preds, labels=classes, average="macro", zero_division=0), preds


def knn_baseline(classes):
    """Our feature KNN restricted to `classes`, shoe-5 LOSO."""
    from evaluation import load_dataset, split_cohort, loso
    import phase3_models as p3
    X, y, subj, files, feats, labels = load_dataset()
    (X, y, subj, files), _ = split_cohort(X, y, subj, files)
    keep_f = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    X = X[:, keep_f]
    mask = np.isin(y, classes)
    lo = loso(lambda: p3.knn(sel=False), X[mask], y[mask], subj[mask], classes)
    return lo["macro_f1"]


ALL10 = ["Lead Jab", "Lead Hook", "Lead Uppercut", "Rear Jab", "Rear Hook",
         "Rear Uppercut", "Left Slip", "Right Slip", "Left Roll", "Right Roll"]


def run_stage(name, classes, cfgs=("head", "norm3"), do_dba=True):
    print(f"\n## {name} — {len(classes)} classes, shoe-5 LOSO")
    knn_f1 = knn_baseline(classes)
    print(f"  KNN (our features) macro-F1 = {knn_f1:.3f}")
    best = ("KNN", knn_f1)
    for cfg in cfgs:
        series, labels, subjects = collect_windows(cfg, classes)
        if not series:
            print(f"  [{cfg}] no windows"); continue
        for k in (1, 3):
            f1, _ = loso_dtw(series, labels, subjects, classes, k=k)
            if f1 > best[1]: best = (f"DTW {k}-NN/{cfg}", f1)
            print(f"  DTW {k}-NN [{cfg:6s}]  macro-F1 = {f1:.3f}")
        if do_dba:
            f1, _ = loso_dba(series, labels, subjects, classes)
            if f1 > best[1]: best = (f"DBA/{cfg}", f1)
            print(f"  DBA      [{cfg:6s}]  macro-F1 = {f1:.3f}")
    print(f"  -> best: {best[0]} {best[1]:.3f}  (vs KNN {knn_f1:.3f}, diff {best[1]-knn_f1:+.3f})")
    return knn_f1, best


def run_timing(classes=None):
    """N1-N4 speed reality check: per-prediction DTW time vs KNN, for
    {1-NN large-ref / DBA small-ref} x {18ch, 6ch}. Python pure-loop baseline;
    report RELATIVE multiples vs KNN (Swift would be much faster)."""
    classes = classes or ALL10
    print("# DTW timing probe — per-prediction speed (Python pure-loop; relative to KNN)")
    print(f"  classes={len(classes)} (full reference, no LOSO split — worst-case deployed ref)\n")

    # KNN per-prediction baseline (sklearn, our features) — for the relative multiple.
    from evaluation import load_dataset, split_cohort
    import phase3_models as p3
    X, y, subj, files, feats, labs = load_dataset()
    (X, y, subj, files), _ = split_cohort(X, y, subj, files)
    keep = [i for i, n in enumerate(feats) if n not in p3.META_FEATURES]
    Xk = X[:, keep]; mask = np.isin(y, classes)
    knn = p3.knn(sel=False).fit(Xk[mask], y[mask])
    xq = Xk[mask][:1]
    t0 = time.perf_counter()
    for _ in range(50):
        knn.predict(xq)
    knn_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  KNN predict/sample = {knn_ms:.3f} ms  (reference {int(mask.sum())} feature-vectors)")

    for cfg in ("norm3", "all18"):
        series, lbl, _ = collect_windows(cfg, classes)
        if not series:
            print(f"  [{cfg}] no windows"); continue
        nref = len(series)
        ch = series[0].shape[1]
        query = series[0]

        # 1-NN: query vs ENTIRE reference (large ref)
        t0 = time.perf_counter()
        for r in series:
            dtw_dist(query, r)
        nn_ms = (time.perf_counter() - t0) * 1000
        print(f"  [{cfg} {ch}ch] 1-NN: 1 query x {nref} ref = {nn_ms:.1f} ms/pred  "
              f"(~{nn_ms/knn_ms:.0f}x KNN)")

        # DBA: query vs ONE template per class (small ref)
        templates = []
        for c in classes:
            mem = [series[i] for i in range(len(series)) if lbl[i] == c]
            if mem:
                templates.append(dba(mem))
        t0 = time.perf_counter()
        for tmpl in templates:
            dtw_dist(query, tmpl)
        dba_ms = (time.perf_counter() - t0) * 1000
        print(f"  [{cfg} {ch}ch] DBA : 1 query x {len(templates)} templates = {dba_ms:.1f} ms/pred  "
              f"(~{dba_ms/knn_ms:.0f}x KNN)")

    print("\n  N1 Python pure-loop — Swift/numba would be MUCH faster; use RELATIVE multiples.")
    print("  N4 demo (one classify per action, user waits) tolerates hundreds of ms;")
    print("     streaming deployment is stricter -> DBA (small ref) or 6ch trade-off.")
    print("# done.")


def run_dtw_calib(classes=None):
    """DTW + calibration: defense, 1-NN(18ch), m=0->2.
    File-separated protocol like phase45: test subject's 3 files per action split
    into 1 calibration file vs 2 test files; calibration windows (m peaks) are
    APPENDED to the factory DTW reference (other subjects). Double-averaged over
    the 3 calib-file choices. m=0 = plain LOSO (no calibration)."""
    from collections import defaultdict
    classes = classes or DEFENSE
    print("# DTW + calibration — defense 4-class, 1-NN(18ch), m=0->2 (file-separated LOSO)")
    series, labels, subjects, files = collect_windows("all18", classes, want_files=True)
    series = list(series)
    subs = sorted(set(subjects.tolist()))
    # index per (subject,label) -> {file: [window idx]}
    by = defaultdict(lambda: defaultdict(list))
    for i in range(len(labels)):
        by[(subjects[i], labels[i])][files[i]].append(i)

    M = [0, 1, 2]
    fold_scores = {m: [] for m in M}
    for ts in subs:
        factory = [j for j in range(len(labels)) if subjects[j] != ts]   # other subjects
        labs_S = sorted({l for (s, l) in by if s == ts})
        for calib_pos in range(3):
            cal_idx = {}; test_idx = []; ok = True
            for lab in labs_S:
                fmap = by[(ts, lab)]; fs = sorted(fmap.keys())
                if len(fs) < 3: ok = False; break
                cal_idx[lab] = sorted(fmap[fs[calib_pos]])
                for f in fs:
                    if f != fs[calib_pos]:
                        test_idx.extend(fmap[f])
            if not ok: continue
            test_idx = np.array(sorted(test_idx))
            for m in M:
                ref = list(factory)
                if m > 0:
                    for lab in labs_S:
                        ref.extend(cal_idx[lab][:m])     # append m calibration peaks/action
                preds = []
                for i in test_idx:
                    best_d, best_l = float("inf"), classes[0]
                    for j in ref:
                        d = dtw_dist(series[i], series[j])
                        if d < best_d: best_d, best_l = d, labels[j]
                    preds.append(best_l)
                fold_scores[m].append(
                    f1_score(labels[test_idx], preds, labels=classes, average="macro", zero_division=0))
        done = {m: round(float(np.mean(fold_scores[m])), 3) for m in M if fold_scores[m]}
        print(f"  {ts:8s}: {done}")

    print("\n## DTW(18ch,1-NN) + calibration curve (defense, shoe-5)")
    for m in M:
        if fold_scores[m]:
            print(f"  m={m}  macro-F1 = {np.mean(fold_scores[m]):.3f}")
    print("  compare: KNN+calib defense m=0->1 (~0.34->0.62 region) & overall calib 0.90.")
    print("  note: gain is partly C2 (own demo = nearest), same caveat as KNN calibration.")
    print("# done.")


def main() -> None:
    import sys as _sys
    if "--dtw-calib" in _sys.argv:
        run_dtw_calib(DEFENSE)
        return
    if "--timing" in _sys.argv:
        run_timing(DEFENSE if "--defense" in _sys.argv else ALL10)
        return
    if "--fairness" in _sys.argv:
        # Fairness improvement A: defense-only, DTW gets full 18 channels (3 sensors x 6 axes),
        # so it sees the same whole-3-IMU signal as KNN. Light (defense, no DBA).
        print("# DTW fairness probe (A) — defense 4-class, DTW with 18 raw channels")
        run_stage("Fairness: DEFENSE (18ch)", DEFENSE, cfgs=("norm3", "all18"), do_dba=False)
        print("\n  note: even with full 3-sensor signal, compare vs KNN 0.383 & calibration 0.89.")
        print("# done."); return
    if "--all18-10" in _sys.argv:
        # 18-channel re-check on ALL 10 classes (norm3 had KNN winning; does 18ch flip it?).
        print("# DTW 18-channel re-check — ALL 10 classes")
        run_stage("ALL 10 (18ch)", ALL10, cfgs=("all18",), do_dba=False)
        print("\n  compare vs KNN 0.581 & calibration 0.90.")
        print("# done."); return
    print("# DTW/DBA probe — shoe-5 LOSO (defense 4-class + all 10-class)")
    run_stage("Stage 1: DEFENSE", DEFENSE)
    run_stage("Stage 2: ALL 10", ALL10)
    print("\n  verdict: DTW/DBA best vs KNN — +0.05↑ => adopt/hybrid; else cite-only.")
    print("# done.")


if __name__ == "__main__":
    main()
