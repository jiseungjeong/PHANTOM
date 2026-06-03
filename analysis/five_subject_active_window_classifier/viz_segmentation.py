#!/usr/bin/env python3
"""피크 기반 크롭 시각화 — 세그멘테이션 품질 점검 (read-only).

각 subplot = 한 (피험자, 동작) 파일의:
  - envelope (모션 세기 곡선, 파랑)
  - 검출된 피크 (빨간 세로선)
  - 5개 active window 크롭 구간 (색칠 밴드 rep1~5)
이를 통해 "5반복을 동작 위치에서 제대로 5조각 잘랐나"를 눈으로 확인.

대표 샘플 그리드: 10동작(행) × 선택 피험자(열). 첫 번째 측정 파일 사용.
출력: reports/segmentation_grid.png  (.gitignore로 추적 제외)

실행: python3 analysis/five_subject_active_window_classifier/viz_segmentation.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
OUT_DIR = THIS.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"

sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"] = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp
sys.path.insert(0, str(THIS.parent))
import phase1_segment as p1      # 최신 choose_real_peaks (floor+gapCV+heightCV+유연반복)

SINGLE_LABELS = ["Lead Jab", "Lead Hook", "Lead Uppercut",
                 "Rear Jab", "Rear Hook", "Rear Uppercut",
                 "Left Slip", "Right Slip", "Left Roll", "Right Roll"]
# 신발 코호트 중심으로 보되, 환경 비교 위해 JEONG(맨발)도 포함
COLS = ["SEUNG", "JIHO", "JUHO", "SHIM"]
WIN_COLORS = ["#7fb3ff", "#84d68a", "#ffd166", "#c792ea", "#ff8fa3"]


def find_file(subject: str, label: str, take: int = 0) -> Path | None:
    """그 라벨(콤보 아님)의 take 번째 측정 파일 (0,1,2 = 3회 측정 중)."""
    folder = DATA_DIR / f"{subject.lower()}_data"
    cands = sorted(p for p in folder.glob(f"{subject}_{label}_*.csv") if "_Watch_" not in p.name)
    return cands[take] if take < len(cands) else None


def segment(path: Path):
    data = rbc.read_csv_sources(path)
    whole, meta = rbc.extract_features(path)
    if whole is None:
        return None
    jump = float(meta.get("jump_cut_s", 0.0))
    times, env, dur, _ = bp.motion_envelope(data, jump)
    if len(env) == 0:
        return None
    peaks, pmeta = p1.choose_real_peaks(times, env, dur)   # 최신 알고리즘
    windows = bp.windows_from_peaks(times, env, peaks, dur) if len(peaks) >= p1.MIN_REPS else []
    windows = p1.uniform_windows(windows, dur)            # 개선5: 길이 균일(넉넉)
    return times, env, windows, pmeta


def draw_grid(take: int) -> None:
    """take 번째 측정 파일로 그리드 1장 생성."""
    nrows, ncols = len(SINGLE_LABELS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 1.7 * nrows), squeeze=False)
    drawn = 0
    for r, label in enumerate(SINGLE_LABELS):
        for c, subj in enumerate(COLS):
            ax = axes[r][c]
            path = find_file(subj, label, take)
            res = segment(path) if path else None
            if res is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            times, env, windows, pmeta = res
            ax.plot(times, env, color="#1f4e79", lw=1.2)
            ymax = max(env.max(), 1e-6)
            for i, w in enumerate(windows):
                ax.add_patch(Rectangle((w["start_s"], 0), w["end_s"] - w["start_s"], ymax,
                                       color=WIN_COLORS[i % 5], alpha=0.45, zorder=0))
                ax.axvline(w["peak_time_s"], color="#d72638", lw=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            n = len(windows)
            if n >= 4:
                ttl = f"{subj} | {label} ({n}pk g{pmeta.get('gap_cv')} h{pmeta.get('height_cv')})"
                col = "black"
            else:
                ttl = f"{subj} | {label} INVALID"; col = "red"
            ax.set_title(ttl, fontsize=6, color=col)
            drawn += 1
    for c, subj in enumerate(COLS):
        axes[0][c].annotate(subj, xy=(0.5, 1.4), xycoords="axes fraction",
                            ha="center", fontsize=10, weight="bold")
    fig.suptitle(f"Peak-based active-window cropping — measurement #{take + 1}/3  "
                 f"(blue=envelope, red=peak, bands=crop)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT_DIR / f"segmentation_grid_{take + 1}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  측정#{take + 1}: subplot {drawn}/{nrows * ncols} → {out.name}")


def main() -> None:
    print("# 피크 크롭 시각화 (3회 측정 각각)")
    for take in (0, 1, 2):
        draw_grid(take)


if __name__ == "__main__":
    main()
