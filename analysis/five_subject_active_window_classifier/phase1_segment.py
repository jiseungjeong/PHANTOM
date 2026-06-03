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

        peaks, peak_meta = bp.choose_five_peaks(times, envelope, duration)
        if len(peaks) != SINGLE_REPETITIONS:
            invalid.append({"file": path.name, "label": label, "subject": subject,
                            "reason": f"peaks={len(peaks)}({peak_meta.get('status')})"}); continue
        windows = bp.windows_from_peaks(times, envelope, peaks, duration)

        coverage_s = sum(max(0.0, w["end_s"] - w["start_s"]) for w in windows)
        qc_rows.append({
            "file": path.name, "label": label, "subject": subject,
            "duration_s": round(duration, 3), "jump_cut_s": round(jump_cut_s, 3),
            "coverage_pct": round(coverage_s / duration, 3) if duration > 0 else 0.0,
            "peak_times_s": [round(w["peak_time_s"], 3) for w in windows],
        })

        for rep_idx, win in enumerate(windows, start=1):
            rep_features, rep_meta = bsr.features_for_window(
                data, jump_cut_s, whole_features.get("jump_found", 0.0) > 0,
                win["start_s"], win["end_s"],
                base={
                    "single_repetition_sample": 1.0,
                    "repetition_index": float(rep_idx),
                    "repetition_count": float(SINGLE_REPETITIONS),
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
    print(f"  사용 파일 수: {files_used}  (기대 150)")
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
