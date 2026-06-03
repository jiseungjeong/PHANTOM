#!/usr/bin/env python3
"""Phase 1 — 단일 동작 세그멘테이션 + feature 추출.

plan.md §2/§4, todolist Phase 1 구현.
- 단일 동작 파일만(하이픈 없음) 대상. 5명 × 30 = 150 파일.
- sync 점프 검출 → active peak window 5개 → 1 window = 1 sample (non-overlap).
- motion_all feature 추출 (~490). feature 이름은 기존 build_single_repetition_dataset와 일치(iOS 정합).
- 결과를 single_dataset.json 으로 저장(학습 단계에서 사용). QC 요약 출력.

기존 검증된 함수(extract_features / motion_envelope / choose_five_peaks /
windows_from_peaks / features_for_window)를 재사용하되, build_samples 는 거치지 않는다.
(이유: build_samples 는 하위폴더 glob 미지원 + rbc.subject_from_name 이 새 5명을 모름.)

상대경로 기준(Q4). 실행: python3 analysis/five_subject_active_window_classifier/phase1_segment.py
"""

from __future__ import annotations

import json
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

# ── 레포 루트 & 기존 스크립트 재사용 (Q4: 경로 하드코딩 회피) ───────────────────
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
OUT_DIR = THIS.parent
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
# build_peak_repetition_dataset 의 옛 패키지경로 import 우회 (파일 수정 없이)
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
    """선택된 피크들의 간격 변동계수(작을수록 등간격)."""
    ts = sorted(float(times[i]) for i in idxs)
    gaps = np.diff(ts)
    m = float(np.mean(gaps))
    return float(np.std(gaps) / m) if m > 0 else 9.99


def _height_cv(idxs, envelope):
    """선택된 피크들의 높이 변동계수(작을수록 높이 균일).
    진짜 5반복은 세기가 고르고, 가짜(점프잔재=너무 큼, 잡음=너무 작음)는 높이가 튄다."""
    hs = np.array([float(envelope[i]) for i in idxs])
    m = float(hs.mean())
    return float(hs.std() / m) if m > 0 else 9.99


def uniform_windows(windows, duration):
    """개선5: 크롭 길이 균일화. 파일 내 윈도우 폭의 대표값을 구해 모든 윈도우를
    각자의 peak_time_s 중심으로 같은 폭으로 재설정.

    폭은 '넉넉하게': 적게 vs 많이 중 많이가 나음(동작 잘림이 더 해로움) → 중앙값이
    아니라 **75퍼센타일 폭**을 대표값으로 써 약간 길게 자른다. 단 이웃 피크 중간점·
    파일 범위로 clamp 해 겹침/이탈은 막는다."""
    if not windows:
        return windows
    widths = [w["end_s"] - w["start_s"] for w in windows]
    half = float(np.percentile(widths, 75)) / 2.0     # 중앙값보다 넉넉 (많이 잡기)
    ws = sorted(windows, key=lambda w: w["peak_time_s"])
    out = []
    for idx, w in enumerate(ws):
        pt = w["peak_time_s"]
        lo = pt - half
        hi = pt + half
        # 이웃 피크와 겹침 방지: 중간점으로 제한
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


MAX_REPS = SINGLE_REPETITIONS      # 5 (정상 반복 수)
MIN_REPS = 4                        # 진짜 봉우리가 4개면 4샘플로 살림. 3개 이하 invalid.


def choose_real_peaks(times, envelope, duration, max_reps=MAX_REPS):
    """floor 강화(개선1) + 간격 규칙성(개선3) + 유연 반복수.

    개선1: floor 미만 봉우리(점프 잔재·잡음)는 끝까지 제외.
    개선3: 6개+ 후보면 '높이 우선'이 아니라 **간격이 가장 일정한 조합**을 골라
           점프 잔재·리듬 깨는 가짜 피크를 배제.
    유연 반복수: 진짜 문제는 '가짜 피크가 끼는 6개+'이지 '진짜 4번 한 것'이 아니다.
           → floor 통과 봉우리(real)가 6개+면 규칙성으로 5개 선택, 정확히 5면 그대로,
             4개면 4샘플로 살림, **3개 이하만 invalid**.
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

    # 목표 개수: 4~5 사이에서 real 수에 맞춤 (6개+ 면 5로 줄임)
    target = min(max_reps, n) if n <= max_reps else max_reps
    # 최소간격은 '잡음 쌍둥이 피크' 제거 수준(0.30s)으로만 둔다. 진짜 동작 간격이
    # 0.6s 수준으로 빠를 수 있어 큰 min_sep 은 정상 파일을 떨군다(no_valid_combo).
    # 촘촘한 가짜는 간격 규칙성(gapCV) 점수가 알아서 배제하므로 여기선 느슨하게.
    min_sep = max(1, int(round(0.30 / BIN_S)))

    if n == target:
        sel, gcv = real, _gap_cv(real, times)
    else:
        # n > target (6개+ → target개로): best 조합 선택. key = 작을수록 좋음.
        #   gapCV(주)         : 간격 등간격
        #   0.3*heightCV(보조): 높이 균일 (점프잔재 큼·잡음 작음 배제)
        #   0.1*edge(약보조)  : 가장자리 피크 회피(개선6) — 점프잔재(맨앞)·꼬리흔들림(맨뒤)
        #                       을 살짝 불리하게. 진짜 앞쪽 반복은 안 밀어내도록 약하게.
        from itertools import combinations
        n_bins = len(envelope)
        best, best_key, gcv_sel = None, None, 9.99
        for combo in combinations(real, target):
            if any(combo[k + 1] - combo[k] < min_sep for k in range(target - 1)):
                continue
            gcv = _gap_cv(combo, times)
            hcv = _height_cv(combo, envelope)
            # edge_penalty: 각 피크의 가장자리 근접도 평균 (0=완전중앙, 1=양끝).
            # |pos-0.5|*2 → 중앙0, 양끝1. 가장자리일수록 페널티 커짐.
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
    """폴더명 기준(jeong_data → JEONG). 기존 rbc.SUBJECTS 가 새 5명을 모르므로 직접."""
    return path.parent.name.replace("_data", "").upper()


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
    return re.sub(r"\s+", " ", label).strip().replace(" - ", "-")


def main() -> None:
    csvs = sorted(p for p in DATA_DIR.glob("*/*.csv") if "_Watch_" not in p.name)
    print(f"# Phase 1 — 단일 동작 세그멘테이션")
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
        if "-" in label:                                  # 콤보 → 1단계 제외
            skipped_combo += 1; continue
        subject = subject_from_path(path)

        # whole-file feature + jump 검출 (P00 fallback 불필요: 5명에 P00 없음)
        whole_features, whole_meta = rbc.extract_features(path)
        if whole_features is None:
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": "extract_features_none", "meta": whole_meta}); continue
        signature = rbc.source_signature(whole_meta)
        if signature != FULL_SENSOR:                      # full-sensor만
            skipped_signature += 1
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": f"signature={signature}"}); continue

        data = rbc.read_csv_sources(path)                 # source별 시간축 데이터(D3: timestamp 기반)
        jump_cut_s = float(whole_meta.get("jump_cut_s", 0.0))
        times, envelope, duration, env_meta = bp.motion_envelope(data, jump_cut_s)
        if len(envelope) == 0:
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": env_meta.get("status", "no_envelope")}); continue

        peaks, peak_meta = choose_real_peaks(times, envelope, duration)   # 개선1+3+유연반복
        if len(peaks) < MIN_REPS:                          # 4·5개 허용, 3개 이하만 invalid
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": f"{peak_meta.get('status')}(real={peak_meta.get('real_peaks')})"}); continue
        windows = bp.windows_from_peaks(times, envelope, peaks, duration)
        windows = uniform_windows(windows, duration)       # 개선5: 길이 균일(넉넉)

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
                    "repetition_count": float(len(windows)),   # 실제 선택 수(4 or 5)
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

    # ── feature 이름 집합 (motion_all) ────────────────────────────────────────
    all_names = sorted({k for s in samples for k in s["features"].keys()})
    motion_all = [n for n in all_names if "present" not in n and "raw_n" not in n and n != "total_rows"]

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "single_dataset.json"
    out_path.write_text(json.dumps({
        "feature_names_all": all_names,
        "feature_names_motion_all": motion_all,
        "samples": samples,
    }, ensure_ascii=False))

    # ── QC 출력 ───────────────────────────────────────────────────────────────
    print(f"\n## 결과")
    print(f"  단일 파일 처리 → 샘플: {len(samples)}  (기대 150파일 × 5 = 750)")
    print(f"  콤보 제외: {skipped_combo}  / signature 제외: {skipped_signature}  / invalid: {len(invalid)}")
    print(f"  피험자별 샘플: {dict(Counter(s['subject'] for s in samples))}")
    print(f"  라벨별 샘플:   {dict(sorted(Counter(s['label'] for s in samples).items()))}")
    files_used = len({s['file'] for s in samples})
    n5 = sum(1 for q in qc_rows if q["n_peaks"] == 5)
    n4 = sum(1 for q in qc_rows if q["n_peaks"] == 4)
    print(f"  사용 파일 수: {files_used}  (5피크 {n5} + 4피크 {n4})")
    print(f"  feature 수: 전체 {len(all_names)} / motion_all {len(motion_all)}")
    if invalid:
        print(f"  ⚠️ invalid 사유: {dict(Counter(i['reason'].split('(')[0] for i in invalid))}")

    # 윈도우 커버리지 분포 (세그멘테이션 건강성)
    if qc_rows:
        cov = np.array([q["coverage_pct"] for q in qc_rows])
        dur = np.array([q["duration_s"] for q in qc_rows])
        print(f"  커버리지%: median={np.median(cov):.2f} min={cov.min():.2f} max={cov.max():.2f}")
        print(f"  파일 duration(s): median={np.median(dur):.1f} min={dur.min():.1f} max={dur.max():.1f}")

    print(f"\n  저장 → {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
    print("# 끝.")


if __name__ == "__main__":
    main()
