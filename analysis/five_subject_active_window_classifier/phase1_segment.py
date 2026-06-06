#!/usr/bin/env python3
"""Phase 1 — single-action segmentation + feature extraction.

Implements plan.md §2/§4, todolist Phase 1.
- Single-action files only (no hyphen). 5 subjects x 30 = 150 files.
- Sync-jump detection -> 5 active peak windows -> 1 window = 1 sample (non-overlap).
- Extract motion_all features (~490). Feature names match the existing
  build_single_repetition_dataset (iOS consistency).
- Save to single_dataset.json (used by the training stage). Print a QC summary.

Reuses the existing verified functions (extract_features / motion_envelope /
choose_five_peaks / windows_from_peaks / features_for_window) but does NOT go
through build_samples. (Reason: build_samples lacks subfolder glob support, and
rbc.subject_from_name does not know the new 5 subjects.)

Relative paths (Q4). Run: python3 analysis/five_subject_active_window_classifier/phase1_segment.py
"""

from __future__ import annotations

import json
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

# ── Repo root & reuse of existing scripts (Q4: avoid hardcoded paths) ───────────
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
OUT_DIR = THIS.parent
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
# Work around build_peak_repetition_dataset's old package-path import (no file edit)
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"] = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp

SINGLE_REPETITIONS = bsr.SINGLE_REPETITIONS          # 5
FULL_SENSOR = bp.FULL_SENSOR_SIGNATURE               # "AirPods+Watch+iPhone"
BIN_S = bp.BIN_S


def _gap_cv(idxs, times):
    """Coefficient of variation of the gaps between selected peaks (smaller = more even spacing)."""
    ts = sorted(float(times[i]) for i in idxs)
    gaps = np.diff(ts)
    m = float(np.mean(gaps))
    return float(np.std(gaps) / m) if m > 0 else 9.99


def _height_cv(idxs, envelope):
    """CV of the heights of selected peaks (smaller = more uniform height).
    A real 5-rep set has even intensity; fakes (jump residue = too big, noise = too small) spike in height."""
    hs = np.array([float(envelope[i]) for i in idxs])
    m = float(hs.mean())
    return float(hs.std() / m) if m > 0 else 9.99


def uniform_windows(windows, duration):
    """Improvement 5: uniform crop width. Compute a representative window width for the
    file and reset every window to the same width, centered on its peak_time_s.

    Width is 'generous': between cropping too little vs too much, too much is better
    (cutting off the action is more harmful) -> use the **75th-percentile width**, not
    the median, to crop a bit longer. Clamp to the neighbor-peak midpoint and the file
    range to prevent overlap/overrun."""
    if not windows:
        return windows
    widths = [w["end_s"] - w["start_s"] for w in windows]
    half = float(np.percentile(widths, 75)) / 2.0     # more generous than median (crop wider)
    ws = sorted(windows, key=lambda w: w["peak_time_s"])
    out = []
    for idx, w in enumerate(ws):
        pt = w["peak_time_s"]
        lo = pt - half
        hi = pt + half
        # prevent overlap with neighbor peaks: clamp to the midpoint
        if idx > 0:
            lo = max(lo, (ws[idx - 1]["peak_time_s"] + pt) / 2.0)
        if idx < len(ws) - 1:
            hi = min(hi, (pt + ws[idx + 1]["peak_time_s"]) / 2.0)
        lo = max(0.0, lo)
        hi = min(duration, hi)
        nw = dict(w)
        nw["start_s"], nw["end_s"] = lo, hi
        out.append(nw)
    return out


MAX_REPS = SINGLE_REPETITIONS      # 5 (normal repetition count)
MIN_REPS = 4                        # if there are 4 real peaks, keep as 4 samples. <=3 is invalid.


def choose_real_peaks(times, envelope, duration, max_reps=MAX_REPS):
    """Strengthened floor (improvement 1) + gap regularity (improvement 3) + flexible rep count.

    Improvement 1: peaks below the floor (jump residue / noise) are excluded outright.
    Improvement 3: with 6+ candidates, instead of 'tallest first', pick the **most
           evenly-spaced combination** to reject jump residue and rhythm-breaking fake peaks.
    Flexible rep count: the real problem is 'fake peaks among 6+', not 'genuinely did it 4 times'.
           -> if real (floor-passing) peaks are 6+, select 5 by regularity; exactly 5 -> use as is;
             4 -> keep as 4 samples; **only <=3 is invalid**.
    """
    peaks = bp.local_peak_indices(envelope)
    if not peaks:
        return [], {"status": "no_peaks", "real_peaks": 0}
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median))) or 1e-6
    floor = max(float(np.percentile(envelope, 62)), median + 0.45 * mad)
    real = sorted(i for i in peaks if envelope[i] >= floor)
    n = len(real)
    if n < MIN_REPS:
        return [], {"status": "too_few_real_peaks", "real_peaks": n, "floor": floor}

    # target count: match real count within 4~5 (6+ -> reduce to 5)
    target = min(max_reps, n) if n <= max_reps else max_reps
    # min separation is only at the 'noise twin-peak' removal level (0.30s). A real action
    # gap can be as fast as ~0.6s, so a large min_sep would drop normal files (no_valid_combo).
    # Tight fakes are rejected by gap regularity (gapCV), so keep this loose here.
    min_sep = max(1, int(round(0.30 / BIN_S)))

    if n == target:
        sel, gcv = real, _gap_cv(real, times)
    else:
        # n > target (6+ -> target): pick the best combination. key = smaller is better.
        #   gapCV (main)        : even spacing
        #   0.3*heightCV (aux)  : uniform height (reject jump residue=big, noise=small)
        #   0.1*edge (weak aux) : avoid edge peaks (improvement 6) — penalize jump residue
        #                         (front) and tail wobble (back) slightly. Weak, so it does not
        #                         push out genuine early reps.
        from itertools import combinations
        n_bins = len(envelope)
        best, best_key, gcv_sel = None, None, 9.99
        for combo in combinations(real, target):
            if any(combo[k + 1] - combo[k] < min_sep for k in range(target - 1)):
                continue
            gcv = _gap_cv(combo, times)
            hcv = _height_cv(combo, envelope)
            # edge_penalty: average edge-proximity of each peak (0=dead center, 1=both ends).
            # |pos-0.5|*2 -> center 0, ends 1. The closer to an edge, the bigger the penalty.
            edge_penalty = float(np.mean([2.0 * abs(i / max(1, n_bins - 1) - 0.5) for i in combo]))
            key = gcv + 0.3 * hcv + 0.1 * edge_penalty
            if best_key is None or key < best_key:
                best_key, best, gcv_sel = key, combo, gcv
        if best is None:
            return [], {"status": "no_valid_combo", "real_peaks": n, "floor": floor}
        sel, gcv = list(best), gcv_sel

    return sorted(sel), {"status": "ok", "real_peaks": n, "n_selected": len(sel),
                         "floor": floor, "gap_cv": round(gcv, 3),
                         "height_cv": round(_height_cv(sel, envelope), 3)}


def subject_from_path(path: Path) -> str:
    """Based on the folder name (jeong_data -> JEONG). Done directly since rbc.SUBJECTS does not know the new 5."""
    return path.parent.name.replace("_data", "").upper()


# ★ Left/Right Slip label correction (2026-06-06). The AirPods yaw (gyroZ) sign at
# the slip peak splits subjects into two opposite groups: {JIHO, SHIM} (treated as
# the correct reference) vs {SEUNG, JO, JUHO}, who demonstrated/labeled Left/Right
# Slip mirrored. Swapping {SEUNG,JO,JUHO}'s Left<->Right Slip cut the L/R-Slip
# confusion ~33%->~5% and raised LOSO macro-F1 0.599->0.659 (probe_swap2). Applied
# at label-parse time so BOTH single and combo (sequence tokens) are corrected, with
# the raw files untouched. (See plan §9.7.)
SLIP_SWAP_SUBJECTS = {"SEUNG", "JO", "JUHO"}


def _swap_slip_tokens(label: str) -> str:
    # swap Left Slip <-> Right Slip within a label (single OR combo "a-b-c")
    out = []
    for tok in label.split("-"):
        if tok == "Left Slip":
            out.append("Right Slip")
        elif tok == "Right Slip":
            out.append("Left Slip")
        else:
            out.append(tok)
    return "-".join(out)


def label_from_path(path: Path) -> str | None:
    import re
    m = re.match(r"^(.*)_(\d{8})_(\d{6})$", path.stem)
    if not m:
        return None
    parts = m.group(1).split("_")
    if parts and parts[0].isupper() and parts[0].isalpha():
        parts = parts[1:]
    if parts and parts[-1] == "Watch":
        parts = parts[:-1]
    label = " ".join(parts).replace("-", " - ")
    label = re.sub(r"\s+", " ", label).strip().replace(" - ", "-")
    subject = path.parent.name.replace("_data", "").upper()
    if subject in SLIP_SWAP_SUBJECTS and "Slip" in label:
        label = _swap_slip_tokens(label)
    return label


def main() -> None:
    csvs = sorted(p for p in DATA_DIR.glob("*/*.csv") if "_Watch_" not in p.name)
    print(f"# Phase 1 — single-action segmentation")
    print(f"DATA_DIR = {DATA_DIR}")

    samples: list[dict] = []
    invalid: list[dict] = []
    qc_rows: list[dict] = []
    skipped_combo = 0
    skipped_signature = 0

    for path in csvs:
        label = label_from_path(path)
        if label is None:
            invalid.append({"file": path.name, "reason": "label_parse_fail"}); continue
        if "-" in label:                                  # combo -> excluded from stage 1
            skipped_combo += 1; continue
        subject = subject_from_path(path)

        # whole-file feature + jump detection (no P00 fallback needed: the 5 subjects have no P00)
        whole_features, whole_meta = rbc.extract_features(path)
        if whole_features is None:
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": "extract_features_none", "meta": whole_meta}); continue
        signature = rbc.source_signature(whole_meta)
        if signature != FULL_SENSOR:                      # full-sensor only
            skipped_signature += 1
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": f"signature={signature}"}); continue

        data = rbc.read_csv_sources(path)                 # per-source time-axis data (D3: timestamp-based)
        jump_cut_s = float(whole_meta.get("jump_cut_s", 0.0))
        times, envelope, duration, env_meta = bp.motion_envelope(data, jump_cut_s)
        if len(envelope) == 0:
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": env_meta.get("status", "no_envelope")}); continue

        peaks, peak_meta = choose_real_peaks(times, envelope, duration)   # improvements 1+3+flexible reps
        if len(peaks) < MIN_REPS:                          # allow 4 or 5, only <=3 is invalid
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": f"{peak_meta.get('status')}(real={peak_meta.get('real_peaks')})"}); continue
        windows = bp.windows_from_peaks(times, envelope, peaks, duration)
        windows = uniform_windows(windows, duration)       # improvement 5: uniform width (generous)

        coverage_s = sum(max(0.0, w["end_s"] - w["start_s"]) for w in windows)
        qc_rows.append({
            "file": path.name, "label": label, "subject": subject,
            "duration_s": round(duration, 3), "jump_cut_s": round(jump_cut_s, 3),
            "coverage_pct": round(coverage_s / duration, 3) if duration > 0 else 0.0,
            "n_peaks": len(windows),
            "peak_times_s": [round(w["peak_time_s"], 3) for w in windows],
        })

        for rep_idx, win in enumerate(windows, start=1):
            rep_features, rep_meta = bsr.features_for_window(
                data, jump_cut_s, whole_features.get("jump_found", 0.0) > 0,
                win["start_s"], win["end_s"],
                base={
                    "single_repetition_sample": 1.0,
                    "repetition_index": float(rep_idx),
                    "repetition_count": float(len(windows)),   # actual selected count (4 or 5)
                    "whole_file_duration_s": duration,
                    "active_peak_time_s": win["peak_time_s"],
                    "active_peak_score": win["peak_score"],
                    "active_window_duration_s": win["end_s"] - win["start_s"],
                },
            )
            if rep_features is None:
                invalid.append({"file": path.name, "label": label, "subject": subject,
                                "reason": f"window_features_none(rep{rep_idx})", "meta": rep_meta}); continue
            samples.append({
                "sample_id": f"{path.stem}__rep{rep_idx}",
                "file": path.name, "label": label, "subject": subject,
                "repetition_index": rep_idx,
                "segmentation_method": "active_peak_window",
                "features": rep_features,
            })

    # ── Feature name set (motion_all) ─────────────────────────────────────────
    all_names = sorted({k for s in samples for k in s["features"].keys()})
    motion_all = [n for n in all_names if "present" not in n and "raw_n" not in n and n != "total_rows"]

    # ── Save ────────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "single_dataset.json"
    out_path.write_text(json.dumps({
        "feature_names_all": all_names,
        "feature_names_motion_all": motion_all,
        "samples": samples,
    }, ensure_ascii=False))

    # ── QC output ─────────────────────────────────────────────────────────────
    print(f"\n## Result")
    print(f"  single files processed -> samples: {len(samples)}  (expected 150 files x 5 = 750)")
    print(f"  combo excluded: {skipped_combo}  / signature excluded: {skipped_signature}  / invalid: {len(invalid)}")
    print(f"  samples per subject: {dict(Counter(s['subject'] for s in samples))}")
    print(f"  samples per label:   {dict(sorted(Counter(s['label'] for s in samples).items()))}")
    files_used = len({s['file'] for s in samples})
    n5 = sum(1 for q in qc_rows if q["n_peaks"] == 5)
    n4 = sum(1 for q in qc_rows if q["n_peaks"] == 4)
    print(f"  files used: {files_used}  (5-peak {n5} + 4-peak {n4})")
    print(f"  feature count: all {len(all_names)} / motion_all {len(motion_all)}")
    if invalid:
        print(f"  (!) invalid reasons: {dict(Counter(i['reason'].split('(')[0] for i in invalid))}")

    # window-coverage distribution (segmentation health)
    if qc_rows:
        cov = np.array([q["coverage_pct"] for q in qc_rows])
        dur = np.array([q["duration_s"] for q in qc_rows])
        print(f"  coverage%: median={np.median(cov):.2f} min={cov.min():.2f} max={cov.max():.2f}")
        print(f"  file duration(s): median={np.median(dur):.1f} min={dur.min():.1f} max={dur.max():.1f}")

    print(f"\n  saved -> {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
    print("# done.")


if __name__ == "__main__":
    main()
