#!/usr/bin/env python3
"""Phase 0 — 데이터 점검 (read-only).

plan.md / todolist.md 의 Phase 0 항목을 실측한다. 모델 학습 없음, 기존 파일 수정 없음.
점검 항목:
  - 인벤토리: 피험자별·라벨별 파일 수 (단일/연속 분리)
  - C0: 세션별 AirPods 유효성 (행 개수·신호 크기) — full-sensor 가정 검증
  - Q5: full-sensor 아닌 세션 제외 규모 (사유·피험자별, silent drop 금지)
  - D2: 단일 파일 피크 검출 수 (기대 5 대비)
  - D4: 라벨 문자열 일관성 (피험자 간 동일 집합인가)
  - 클래스 분포 (불균형)
  - D1: 콤보 파일 피크 적중률 (검출 피크 수 == 콤보 길이?), 방어 누락 경향

경로는 상대경로 기준(Q4). 레포 루트에서 실행: python3 analysis/five_subject_active_window_classifier/phase0_inspect.py
"""

from __future__ import annotations

import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── 레포 루트 & 기존 스크립트 재사용 (경로 하드코딩 회피 = Q4) ──────────────────
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]                       # .../PHANTOM
DATA_DIR = REPO / "Data"
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc            # 단독 import OK

# build_peak_repetition_dataset 는 옛 패키지경로를 하드코딩 import 한다.
# 그 이름으로 현재 모듈들을 미리 등록해 우회(파일 수정 없이).
pkg_a = types.ModuleType("analysis")
pkg_b = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis"] = pkg_a
sys.modules["analysis.active_window_classifier_minimal"] = pkg_b
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp

SOURCES = rbc.SOURCES                         # ('Watch','iPhone','AirPods')
MIN_VALID_ROWS = 5                            # 한 기기가 '유효'로 간주되는 최소 행수


def subject_from_path(path: Path) -> str:
    """폴더명(jeong_data) 기준 — 기존 rbc.SUBJECTS 상수가 새 5명을 모르므로 직접 파싱."""
    return path.parent.name.replace("_data", "").upper()


def label_from_path(path: Path) -> str | None:
    """파일명에서 피험자 접두어·타임스탬프 제거 후 라벨만. rbc.SUBJECTS 의존 회피."""
    import re
    m = re.match(r"^(.*)_(\d{8})_(\d{6})$", path.stem)
    if not m:
        return None
    parts = m.group(1).split("_")
    # 첫 토큰이 대문자 피험자명이면 제거 (JEONG_, SEUNG_ 등 포함)
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
    """jump_cut 검출 → envelope → 피크 검출. (검출수, meta) 반환."""
    try:
        jump_cut_s, _ = rbc.detect_jump_cut(data)
        times, envelope, duration, env_meta = bp.motion_envelope(data, jump_cut_s)
        if envelope is None or len(envelope) == 0 or duration <= 0:
            return 0, {"status": env_meta.get("status", "no_envelope")}
        sel, meta = bp.choose_five_peaks(times, envelope, duration, repetitions=expected)
        return len(sel), meta
    except Exception as e:                     # 점검이므로 예외는 집계만
        return -1, {"status": f"error:{type(e).__name__}:{e}"}


def main() -> None:
    full_csvs = sorted(
        p for p in DATA_DIR.glob("*/*.csv") if "_Watch_" not in p.name
    )
    print(f"# Phase 0 데이터 점검")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"full-sensor 후보 파일(워치파생 제외): {len(full_csvs)}\n")

    # 집계 버킷
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

        # C0 / Q5: full-sensor(3기기) 충족 여부
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

    # ── 출력 ─────────────────────────────────────────────────────────────────
    print("## 1. 인벤토리 (피험자별 full-sensor 후보)")
    for s in sorted(by_subject):
        print(f"  {s:8s}: {by_subject[s]:3d}  (단일+연속 합)")
    print(f"  단일 파일: {len(single_files)} / 연속 파일: {len(combo_files)}\n")

    print("## 2. C0/Q5 — 제외 규모 (full-sensor 미충족 등)")
    print(f"  제외 총: {len(excluded)}")
    reasons = Counter(r for _, r in excluded)
    for r, c in reasons.most_common():
        print(f"    {c:3d}  {r}")
    # 제외 파일 피험자별
    exc_subj = Counter(f.split("_")[0] for f, _ in excluded)
    if exc_subj:
        print("  제외 피험자별:", dict(exc_subj))
    print()

    print("## 3. C0 — AirPods 행 개수 분포 (전체 후보)")
    air = np.array([r[2] for r in airpods_report], dtype=float)
    wat = np.array([r[3] for r in airpods_report], dtype=float)
    iph = np.array([r[4] for r in airpods_report], dtype=float)
    def stat(name, arr):
        print(f"  {name:8s} min={arr.min():.0f} p25={np.percentile(arr,25):.0f} "
              f"median={np.median(arr):.0f} p75={np.percentile(arr,75):.0f} max={arr.max():.0f} "
              f"  (<{MIN_VALID_ROWS}행: {(arr<MIN_VALID_ROWS).sum()}개, ==0: {(arr==0).sum()}개)")
    stat("AirPods", air); stat("Watch", wat); stat("iPhone", iph)
    # AirPods가 약한(하위) 세션 피험자별
    weak_air = Counter(r[0] for r in airpods_report if r[2] < 50)
    print(f"  AirPods<50행 세션 피험자별: {dict(weak_air) if weak_air else '없음'}\n")

    print("## 4. D4 — 라벨 문자열 일관성 (피험자 간)")
    all_labels = set().union(*label_set_by_subject.values()) if label_set_by_subject else set()
    print(f"  전체 고유 라벨: {len(all_labels)} (단일 {sum(not is_combo(l) for l in all_labels)} + 연속 {sum(is_combo(l) for l in all_labels)})")
    for s in sorted(label_set_by_subject):
        missing = all_labels - label_set_by_subject[s]
        if missing:
            print(f"  ⚠️ {s} 누락 라벨({len(missing)}): {sorted(missing)}")
    if all(label_set_by_subject[s] == all_labels for s in label_set_by_subject):
        print("  ✅ 모든 피험자가 동일 라벨 집합 보유\n")
    else:
        print()

    print("## 5. 클래스 분포 (단일 동작만)")
    single_labels = Counter(label_from_path(p) for p in single_files)
    for l in sorted(single_labels):
        print(f"  {l:16s}: {single_labels[l]}")
    if single_labels:
        vals = list(single_labels.values())
        print(f"  → min={min(vals)} max={max(vals)} (불균형비 {max(vals)/min(vals):.2f})\n")

    print("## 6. D2 — 단일 파일 피크 검출 수 (기대 5)")
    npk = Counter(r[2] for r in single_peak_rows)
    for k in sorted(npk):
        flag = "" if k == bp.SINGLE_REPETITIONS else "  ⚠️"
        print(f"  피크 {k}개: {npk[k]} 파일{flag}")
    bad = [r for r in single_peak_rows if r[2] != bp.SINGLE_REPETITIONS]
    print(f"  → 기대(5) 불일치: {len(bad)}/{len(single_peak_rows)}")
    if bad[:8]:
        for r in bad[:8]:
            print(f"     {r[0]}/{r[1]}: {r[2]}개 ({r[3]})")
    print()

    print("## 7. D1 — 콤보 피크 적중률 (검출수 == 콤보길이?)")
    if combo_peak_rows:
        hit = sum(1 for r in combo_peak_rows if r[4])
        print(f"  적중(검출==길이): {hit}/{len(combo_peak_rows)} = {hit/len(combo_peak_rows)*100:.1f}%")
        # 콤보 길이별
        by_len = defaultdict(lambda: [0, 0])
        for r in combo_peak_rows:
            by_len[r[2]][0] += 1
            by_len[r[2]][1] += int(r[4])
        for L in sorted(by_len):
            tot, h = by_len[L]
            print(f"    {L}동작 콤보: 적중 {h}/{tot} = {h/tot*100:.1f}%")
        # 방어 포함 콤보 vs 펀치만 콤보 적중률 (방어 누락 경향)
        def has_def(lbl): return any(k in lbl for k in ("Slip", "Roll"))
        defc = [r for r in combo_peak_rows if has_def(r[1])]
        punc = [r for r in combo_peak_rows if not has_def(r[1])]
        if defc:
            print(f"    방어 포함 콤보 적중: {sum(r[4] for r in defc)}/{len(defc)} = {sum(r[4] for r in defc)/len(defc)*100:.1f}%")
        if punc:
            print(f"    펀치만 콤보 적중:   {sum(r[4] for r in punc)}/{len(punc)} = {sum(r[4] for r in punc)/len(punc)*100:.1f}%")
    print("\n# 끝.")


if __name__ == "__main__":
    main()
