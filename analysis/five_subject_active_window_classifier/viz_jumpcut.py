#!/usr/bin/env python3
"""점프 컷 검증 시각화 (read-only).

세그멘테이션은 'sync 점프 이후'부터 동작을 자른다. 이 점프 컷 시점이
올바른 경계(맨 처음 동기화 흔들림의 끝)에 놓였는지 눈으로 검증한다.

⭐ 실제 점프컷·크롭이 일어나는 신호와 동일하게 **융합 motion_envelope**으로 그린다
   (자이로62%+가속도38%, 3센서 72%max+28%mean 합침). 단 점프 전 구간도 보여야 하므로
   jump_cut_s=0 으로 envelope 을 구해 전체를 그린 뒤, 실제 jump_cut 위치를 선으로 표시.
   → segmentation_grid.png 와 같은 곡선이라 두 그림이 한눈에 연결됨.

각 subplot = 한 (피험자, 동작):
  - 융합 envelope (점프 포함 전체, 파랑)
  - jump_cut_s 위치 = 빨간 점선 (왼쪽 음영=버려지는 점프 구간, 오른쪽=동작 5반복)
점프 선이 초반 큰 점프 봉우리 '뒤', 규칙적 동작 봉우리들 '앞'이면 OK.

출력: reports/jumpcut_grid.png  (.gitignore 제외)
실행: python3 analysis/five_subject_active_window_classifier/viz_jumpcut.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

SINGLE_LABELS = ["Lead Jab", "Lead Hook", "Lead Uppercut",
                 "Rear Jab", "Rear Hook", "Rear Uppercut",
                 "Left Slip", "Right Slip", "Left Roll", "Right Roll"]
COLS = ["SEUNG", "JIHO", "JUHO", "SHIM"]


def find_file(subject: str, label: str) -> Path | None:
    folder = DATA_DIR / f"{subject.lower()}_data"
    cands = sorted(p for p in folder.glob(f"{subject}_{label}_*.csv") if "_Watch_" not in p.name)
    return cands[0] if cands else None


def full_envelope(data):
    """점프 포함 전체 구간의 융합 envelope. jump_cut=0 으로 구함."""
    times, env, dur, _ = bp.motion_envelope(data, 0.0)
    return (times, env) if len(env) else (None, None)


def main() -> None:
    nrows, ncols = len(SINGLE_LABELS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 1.8 * nrows), squeeze=False)
    print("# 점프 컷 검증 시각화")
    bad = []
    for r, label in enumerate(SINGLE_LABELS):
        for c, subj in enumerate(COLS):
            ax = axes[r][c]
            path = find_file(subj, label)
            if path is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_xticks([]); ax.set_yticks([]); continue
            data = rbc.read_csv_sources(path)
            _, meta = rbc.extract_features(path)
            jump = float(meta.get("jump_cut_s", 0.0))
            t, env = full_envelope(data)
            if t is None:
                ax.text(0.5, 0.5, "no env", ha="center", va="center"); ax.set_xticks([]); ax.set_yticks([]); continue
            ax.plot(t, env, color="#1f4e79", lw=1.0)                 # 융합 envelope (점프 포함 전체)
            ax.axvline(jump, color="red", lw=1.6, ls="--")           # 점프 컷 경계
            ax.axvspan(0, jump, color="red", alpha=0.08)             # 버려지는 구간
            ax.set_xticks([]); ax.set_yticks([])
            tot = t[-1] if len(t) else 1.0
            ok = not (jump <= 0.05 or jump > 0.6 * tot)
            ax.set_title(f"{subj} | {label}  cut={jump:.1f}s" + ("" if ok else " ⚠️"),
                         fontsize=7, color="black" if ok else "red")
            if not ok:
                bad.append((subj, label, jump, tot))
    fig.suptitle("Jump-cut verification — FUSED envelope (gyro62%+acc38%, 3-sensor); "
                 "red dashed = jump_cut (left shaded = discarded)",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT_DIR / "jumpcut_grid.png"
    fig.savefig(out, dpi=110)
    print(f"  저장 → {out}")
    if bad:
        print(f"  ⚠️ 의심 컷({len(bad)}): " + ", ".join(f"{s}/{l}(cut={j:.1f}/{t:.1f}s)" for s,l,j,t in bad))
    else:
        print("  ✅ 모든 컷이 합리적 범위(0<cut<60% 길이)")


if __name__ == "__main__":
    main()
