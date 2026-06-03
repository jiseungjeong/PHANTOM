#!/usr/bin/env python3
"""Phase 0 — data inspection (read-only).

Measures the Phase 0 items in plan.md / todolist.md. No model training, no edits
to existing files.
Inspection items:
  - inventory: file counts per subject/label (single vs combo separated)
  - C0: per-session AirPods validity (row count, signal magnitude) — verify the full-sensor assumption
  - Q5: how many sessions are dropped for not being full-sensor (by reason / by subject, no silent drop)
  - D2: peak count detected per single file (vs expected 5)
  - D4: label-string consistency (is the set identical across subjects?)
  - class distribution (imbalance)
  - D1: combo-file peak hit rate (detected peak count == combo length?), defense-miss tendency

Relative paths (Q4). Run from repo root: python3 analysis/five_subject_active_window_classifier/phase0_inspect.py
"""

from __future__ import annotations

import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Repo root & reuse of existing scripts (avoid hardcoded paths = Q4) ──────────
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]                       # .../PHANTOM
DATA_DIR = REPO / "Data"
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc            # standalone import OK

# build_peak_repetition_dataset hardcodes imports of the old package path.
# Pre-register the current modules under those names to work around it (no file edits).
pkg_a = types.ModuleType("analysis")
pkg_b = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis"] = pkg_a
sys.modules["analysis.active_window_classifier_minimal"] = pkg_b
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp

SOURCES = rbc.SOURCES                         # ('Watch','iPhone','AirPods')
MIN_VALID_ROWS = 5                            # min rows for a device to count as 'valid'


def subject_from_path(path: Path) -> str:
    """Based on the folder name (jeong_data) — parse directly since rbc.SUBJECTS does not know the new 5."""
    return path.parent.name.replace("_data", "").upper()


def label_from_path(path: Path) -> str | None:
    """Strip the subject prefix and timestamp from the filename, leaving only the label. Avoids rbc.SUBJECTS."""
    import re
    m = re.match(r"^(.*)_(\d{8})_(\d{6})$", path.stem)
    if not m:
        return None
    parts = m.group(1).split("_")
    # Drop the first token if it is an uppercase subject name (includes JEONG_, SEUNG_, etc.)
    if parts and parts[0].isupper() and parts[0].isalpha():
        parts = parts[1:]
    if parts and parts[-1] == "Watch":
        parts = parts[:-1]
    label = " ".join(parts).replace("-", " - ")
    return re.sub(r"\s+", " ", label).strip().replace(" - ", "-")


def is_combo(label: str) -> bool:
    return "-" in label


def combo_len(label: str) -> int:
    return len(label.split("-"))


def detect_peaks(data: dict, expected: int) -> tuple[int, dict]:
    """jump_cut detection -> envelope -> peak detection. Returns (count, meta)."""
    try:
        jump_cut_s, _ = rbc.detect_jump_cut(data)
        times, envelope, duration, env_meta = bp.motion_envelope(data, jump_cut_s)
        if envelope is None or len(envelope) == 0 or duration <= 0:
            return 0, {"status": env_meta.get("status", "no_envelope")}
        sel, meta = bp.choose_five_peaks(times, envelope, duration, repetitions=expected)
        return len(sel), meta
    except Exception as e:                     # inspection only — just tally exceptions
        return -1, {"status": f"error:{type(e).__name__}:{e}"}


def main() -> None:
    full_csvs = sorted(
        p for p in DATA_DIR.glob("*/*.csv") if "_Watch_" not in p.name
    )
    print(f"# Phase 0 data inspection")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"full-sensor candidate files (watch-derived excluded): {len(full_csvs)}\n")

    # tally buckets
    by_subject = Counter()
    by_label = Counter()
    single_files, combo_files = [], []
    label_set_by_subject = defaultdict(set)

    excluded = []                # (file, reason)
    airpods_report = []          # (subject, label, n_air, n_watch, n_iphone, signature)
    single_peak_rows = []        # (subject, label, n_peaks, status)
    combo_peak_rows = []         # (subject, label, expected, n_peaks, hit, status)

    for path in full_csvs:
        subject = subject_from_path(path)
        label = label_from_path(path)
        if label is None:
            excluded.append((path.name, "label_parse_fail")); continue

        by_subject[subject] += 1
        by_label[label] += 1
        label_set_by_subject[subject].add(label)

        data = rbc.read_csv_sources(path)
        counts = {s: len(data.get(s, [])) for s in SOURCES}
        present = [s for s in SOURCES if counts.get(s, 0) >= MIN_VALID_ROWS]
        signature = "+".join(sorted(present))

        # C0 / Q5: whether full-sensor (3 devices) is satisfied
        is_full = set(present) == set(SOURCES)
        airpods_report.append((subject, label, counts["AirPods"], counts["Watch"], counts["iPhone"], signature))
        if not is_full:
            excluded.append((path.name, f"not_full_sensor({signature or 'empty'})"))
            continue

        if is_combo(label):
            combo_files.append(path)
            exp = combo_len(label)
            n, meta = detect_peaks(data, exp)
            combo_peak_rows.append((subject, label, exp, n, n == exp, meta.get("status", "ok")))
        else:
            single_files.append(path)
            n, meta = detect_peaks(data, bp.SINGLE_REPETITIONS)
            single_peak_rows.append((subject, label, n, meta.get("status", "ok")))

    # ── Output ─────────────────────────────────────────────────────────────────
    print("## 1. Inventory (full-sensor candidates per subject)")
    for s in sorted(by_subject):
        print(f"  {s:8s}: {by_subject[s]:3d}  (single+combo total)")
    print(f"  single files: {len(single_files)} / combo files: {len(combo_files)}\n")

    print("## 2. C0/Q5 — drop count (not full-sensor, etc.)")
    print(f"  total dropped: {len(excluded)}")
    reasons = Counter(r for _, r in excluded)
    for r, c in reasons.most_common():
        print(f"    {c:3d}  {r}")
    # dropped files by subject
    exc_subj = Counter(f.split("_")[0] for f, _ in excluded)
    if exc_subj:
        print("  dropped by subject:", dict(exc_subj))
    print()

    print("## 3. C0 — AirPods row-count distribution (all candidates)")
    air = np.array([r[2] for r in airpods_report], dtype=float)
    wat = np.array([r[3] for r in airpods_report], dtype=float)
    iph = np.array([r[4] for r in airpods_report], dtype=float)
    def stat(name, arr):
        print(f"  {name:8s} min={arr.min():.0f} p25={np.percentile(arr,25):.0f} "
              f"median={np.median(arr):.0f} p75={np.percentile(arr,75):.0f} max={arr.max():.0f} "
              f"  (<{MIN_VALID_ROWS} rows: {(arr<MIN_VALID_ROWS).sum()}, ==0: {(arr==0).sum()})")
    stat("AirPods", air); stat("Watch", wat); stat("iPhone", iph)
    # sessions with weak (low) AirPods, by subject
    weak_air = Counter(r[0] for r in airpods_report if r[2] < 50)
    print(f"  AirPods<50-row sessions by subject: {dict(weak_air) if weak_air else 'none'}\n")

    print("## 4. D4 — label-string consistency (across subjects)")
    all_labels = set().union(*label_set_by_subject.values()) if label_set_by_subject else set()
    print(f"  total unique labels: {len(all_labels)} (single {sum(not is_combo(l) for l in all_labels)} + combo {sum(is_combo(l) for l in all_labels)})")
    for s in sorted(label_set_by_subject):
        missing = all_labels - label_set_by_subject[s]
        if missing:
            print(f"  (!) {s} missing labels ({len(missing)}): {sorted(missing)}")
    if all(label_set_by_subject[s] == all_labels for s in label_set_by_subject):
        print("  OK: every subject has the identical label set\n")
    else:
        print()

    print("## 5. Class distribution (single actions only)")
    single_labels = Counter(label_from_path(p) for p in single_files)
    for l in sorted(single_labels):
        print(f"  {l:16s}: {single_labels[l]}")
    if single_labels:
        vals = list(single_labels.values())
        print(f"  -> min={min(vals)} max={max(vals)} (imbalance ratio {max(vals)/min(vals):.2f})\n")

    print("## 6. D2 — peaks detected per single file (expected 5)")
    npk = Counter(r[2] for r in single_peak_rows)
    for k in sorted(npk):
        flag = "" if k == bp.SINGLE_REPETITIONS else "  (!)"
        print(f"  {k} peaks: {npk[k]} files{flag}")
    bad = [r for r in single_peak_rows if r[2] != bp.SINGLE_REPETITIONS]
    print(f"  -> mismatch vs expected (5): {len(bad)}/{len(single_peak_rows)}")
    if bad[:8]:
        for r in bad[:8]:
            print(f"     {r[0]}/{r[1]}: {r[2]} ({r[3]})")
    print()

    print("## 7. D1 — combo peak hit rate (detected count == combo length?)")
    if combo_peak_rows:
        hit = sum(1 for r in combo_peak_rows if r[4])
        print(f"  hit (detected==length): {hit}/{len(combo_peak_rows)} = {hit/len(combo_peak_rows)*100:.1f}%")
        # by combo length
        by_len = defaultdict(lambda: [0, 0])
        for r in combo_peak_rows:
            by_len[r[2]][0] += 1
            by_len[r[2]][1] += int(r[4])
        for L in sorted(by_len):
            tot, h = by_len[L]
            print(f"    {L}-action combo: hit {h}/{tot} = {h/tot*100:.1f}%")
        # combos with defense vs punch-only (defense-miss tendency)
        def has_def(lbl): return any(k in lbl for k in ("Slip", "Roll"))
        defc = [r for r in combo_peak_rows if has_def(r[1])]
        punc = [r for r in combo_peak_rows if not has_def(r[1])]
        if defc:
            print(f"    combos with defense hit: {sum(r[4] for r in defc)}/{len(defc)} = {sum(r[4] for r in defc)/len(defc)*100:.1f}%")
        if punc:
            print(f"    punch-only combos hit:   {sum(r[4] for r in punc)}/{len(punc)} = {sum(r[4] for r in punc)/len(punc)*100:.1f}%")
    print("\n# done.")


if __name__ == "__main__":
    main()
