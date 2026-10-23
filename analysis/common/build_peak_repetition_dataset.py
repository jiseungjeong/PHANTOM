#!/usr/bin/env python3
"""Build single-repetition samples by detecting active motion bursts.

Unlike the older equal-time split, this script finds five motion peaks after the
initial sync jump and extracts only the active window around each peak.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np

import analysis.active_window_classifier_minimal.rule_base_classifier as rbc
from analysis.active_window_classifier_minimal.build_single_repetition_dataset import (
    SINGLE_REPETITIONS,
    extract_features_with_p00_fallback,
    features_for_window,
    is_single_action,
    relative_duration,
)


FULL_SENSOR_SIGNATURE = "AirPods+Watch+iPhone"
BIN_S = 0.04
SMOOTH_BINS = 5
MIN_WINDOW_S = 0.32
MAX_WINDOW_S = 1.65
WINDOW_PADDING_S = 0.10


def robust_positive_z(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, 1.4826 * mad)
    return np.maximum(0.0, (values - median) / scale)


def smooth(values: np.ndarray, width: int = SMOOTH_BINS) -> np.ndarray:
    if len(values) == 0 or width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def motion_envelope(
    data: dict[str, list[tuple[float, float, float, float, float, float, float]]],
    jump_cut_s: float,
    bin_s: float = BIN_S,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    duration = relative_duration(data, jump_cut_s)
    if duration <= 0:
        return np.array([]), np.array([]), 0.0, {"status": "no_duration"}

    n_bins = max(1, int(math.ceil(duration / bin_s)))
    per_source: list[np.ndarray] = []
    source_meta: dict[str, Any] = {}

    for source in rbc.SOURCES:
        rows = data.get(source, [])
        if len(rows) < 5:
            continue
        t0 = rows[0][0]
        segment = [row for row in rows if 0 <= row[0] - t0 - jump_cut_s < duration]
        if len(segment) < 5:
            continue

        arr = np.array([[row[0] - t0 - jump_cut_s, *row[1:]] for row in segment], dtype=float)
        t = arr[:, 0]
        ax, ay, az, gx, gy, gz = (arr[:, i] for i in range(1, 7))
        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        acc_dynamic = np.abs(acc_mag - np.median(acc_mag))

        # Gyro captures punches/rolls strongly; dynamic acceleration helps when
        # gyro is quiet. Robust z-scoring makes the three sensors comparable.
        score = 0.62 * robust_positive_z(gyro_mag) + 0.38 * robust_positive_z(acc_dynamic)
        score = np.clip(score, 0.0, 14.0)

        binned = np.zeros(n_bins, dtype=float)
        counts = np.zeros(n_bins, dtype=int)
        for tt, value in zip(t, score):
            idx = int(min(n_bins - 1, max(0, math.floor(tt / bin_s))))
            binned[idx] = max(binned[idx], float(value))
            counts[idx] += 1
        per_source.append(smooth(binned))
        source_meta[source] = {
            "rows": len(segment),
            "nonempty_bins": int(np.sum(counts > 0)),
            "score_p95": float(np.percentile(score, 95)),
            "score_max": float(np.max(score)),
        }

    if not per_source:
        return np.array([]), np.array([]), duration, {"status": "no_source_scores"}

    stacked = np.vstack(per_source)
    # Max keeps source-specific actions visible; mean damps single-sensor noise.
    envelope = 0.72 * np.max(stacked, axis=0) + 0.28 * np.mean(stacked, axis=0)
    envelope = smooth(envelope, width=SMOOTH_BINS)
    times = (np.arange(n_bins, dtype=float) + 0.5) * bin_s
    return times, envelope, duration, {"status": "ok", "sources": source_meta}


def local_peak_indices(values: np.ndarray) -> list[int]:
    if len(values) < 3:
        return []
    peaks = []
    for idx in range(1, len(values) - 1):
        if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1] and values[idx] > 0:
            peaks.append(idx)
    return peaks


def choose_five_peaks(times: np.ndarray, envelope: np.ndarray, duration: float, repetitions: int = SINGLE_REPETITIONS) -> tuple[list[int], dict[str, Any]]:
    peaks = local_peak_indices(envelope)
    if not peaks:
        return [], {"status": "no_peaks"}

    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median))) or 1e-6
    amplitude_floor = max(float(np.percentile(envelope, 62)), median + 0.45 * mad)
    peaks = [idx for idx in peaks if envelope[idx] >= amplitude_floor]
    if len(peaks) < repetitions:
        peaks = local_peak_indices(envelope)

    min_sep_s_options = [
        max(0.40, min(0.95, duration / (repetitions * 2.15))),
        max(0.32, min(0.75, duration / (repetitions * 2.80))),
        0.24,
    ]

    selected: list[int] = []
    used_sep_s = min_sep_s_options[-1]
    for min_sep_s in min_sep_s_options:
        min_sep_bins = max(1, int(round(min_sep_s / BIN_S)))
        selected = []
        for idx in sorted(peaks, key=lambda item: float(envelope[item]), reverse=True):
            if all(abs(idx - chosen) >= min_sep_bins for chosen in selected):
                selected.append(idx)
            if len(selected) == repetitions:
                break
        if len(selected) == repetitions:
            used_sep_s = min_sep_s
            break

    if len(selected) < repetitions:
        return [], {
            "status": "too_few_peaks",
            "candidate_peaks": len(peaks),
            "amplitude_floor": amplitude_floor,
        }

    selected = sorted(selected[:repetitions])
    return selected, {
        "status": "ok",
        "candidate_peaks": len(peaks),
        "amplitude_floor": amplitude_floor,
        "min_peak_separation_s": used_sep_s,
        "peak_times_s": [float(times[idx]) for idx in selected],
        "peak_scores": [float(envelope[idx]) for idx in selected],
    }


def low_run_boundary(envelope: np.ndarray, start_idx: int, stop_idx: int, direction: int, threshold: float) -> int | None:
    """Search from start toward stop for two low bins."""

    idx = start_idx
    low_run = 0
    while 0 <= idx < len(envelope) and (idx >= stop_idx if direction < 0 else idx <= stop_idx):
        if envelope[idx] <= threshold:
            low_run += 1
            if low_run >= 2:
                return idx
        else:
            low_run = 0
        idx += direction
    return None


def windows_from_peaks(times: np.ndarray, envelope: np.ndarray, peak_indices: list[int], duration: float) -> list[dict[str, float]]:
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median))) or 1e-6
    base_threshold = max(float(np.percentile(envelope, 38)), median + 0.15 * mad)

    windows: list[dict[str, float]] = []
    peak_times = [float(times[idx]) for idx in peak_indices]
    for rep_idx, peak_idx in enumerate(peak_indices):
        peak_t = float(times[peak_idx])
        peak_score = float(envelope[peak_idx])
        left_hard_t = 0.0 if rep_idx == 0 else (peak_times[rep_idx - 1] + peak_t) / 2.0
        right_hard_t = duration if rep_idx == len(peak_indices) - 1 else (peak_t + peak_times[rep_idx + 1]) / 2.0
        left_hard_idx = max(0, int(math.floor(left_hard_t / BIN_S)))
        right_hard_idx = min(len(envelope) - 1, int(math.ceil(right_hard_t / BIN_S)))

        threshold = max(base_threshold, min(peak_score * 0.28, peak_score - 0.35))
        left_idx = low_run_boundary(envelope, peak_idx, left_hard_idx, -1, threshold)
        right_idx = low_run_boundary(envelope, peak_idx, right_hard_idx, 1, threshold)

        start_s = left_hard_t if left_idx is None else max(left_hard_t, float(times[left_idx]) + BIN_S)
        end_s = right_hard_t if right_idx is None else min(right_hard_t, float(times[right_idx]) - BIN_S)
        start_s = max(left_hard_t, start_s - WINDOW_PADDING_S)
        end_s = min(right_hard_t, end_s + WINDOW_PADDING_S)

        if end_s - start_s < MIN_WINDOW_S:
            half = MIN_WINDOW_S / 2.0
            start_s = max(left_hard_t, peak_t - half)
            end_s = min(right_hard_t, peak_t + half)
        if end_s - start_s > MAX_WINDOW_S:
            half = MAX_WINDOW_S / 2.0
            start_s = max(left_hard_t, peak_t - half)
            end_s = min(right_hard_t, peak_t + half)

        windows.append(
            {
                "repetition_index": float(rep_idx + 1),
                "start_s": float(max(0.0, start_s)),
                "end_s": float(min(duration, end_s)),
                "peak_time_s": peak_t,
                "peak_score": peak_score,
                "threshold": float(threshold),
            }
        )
    return windows


def equal_fallback_windows(duration: float, repetitions: int = SINGLE_REPETITIONS) -> list[dict[str, float]]:
    return [
        {
            "repetition_index": float(idx + 1),
            "start_s": float(duration * idx / repetitions),
            "end_s": float(duration * (idx + 1) / repetitions),
            "peak_time_s": float(duration * (idx + 0.5) / repetitions),
            "peak_score": 0.0,
            "threshold": 0.0,
        }
        for idx in range(repetitions)
    ]


def svg_plot(path: Path, times: np.ndarray, envelope: np.ndarray, windows: list[dict[str, float]], title: str) -> None:
    width, height = 980, 260
    margin_l, margin_r, margin_t, margin_b = 48, 16, 30, 28
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_t = float(times[-1]) if len(times) else 1.0
    max_y = max(float(np.max(envelope)) if len(envelope) else 1.0, 1e-6)

    def x(t: float) -> float:
        return margin_l + plot_w * t / max_t

    def y(v: float) -> float:
        return margin_t + plot_h * (1.0 - v / max_y)

    points = " ".join(f"{x(float(t)):.1f},{y(float(v)):.1f}" for t, v in zip(times, envelope))
    rects = []
    colors = ["#d8ecff", "#dff5dc", "#fff0c9", "#f5ddff", "#ffdede"]
    for idx, win in enumerate(windows):
        rx = x(win["start_s"])
        rw = max(1.0, x(win["end_s"]) - rx)
        rects.append(f'<rect x="{rx:.1f}" y="{margin_t}" width="{rw:.1f}" height="{plot_h}" fill="{colors[idx % len(colors)]}" opacity="0.65"/>')
        rects.append(f'<line x1="{x(win["peak_time_s"]):.1f}" x2="{x(win["peak_time_s"]):.1f}" y1="{margin_t}" y2="{margin_t + plot_h}" stroke="#d72638" stroke-width="1.5"/>')
        rects.append(f'<text x="{rx + 4:.1f}" y="{margin_t + 16}" font-size="12" fill="#111">rep{idx + 1}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_l}" y="20" font-size="14" font-family="Arial" fill="#111">{title}</text>
{''.join(rects)}
<polyline points="{points}" fill="none" stroke="#1f4e79" stroke-width="2"/>
<line x1="{margin_l}" x2="{margin_l + plot_w}" y1="{margin_t + plot_h}" y2="{margin_t + plot_h}" stroke="#444"/>
<line x1="{margin_l}" x2="{margin_l}" y1="{margin_t}" y2="{margin_t + plot_h}" stroke="#444"/>
<text x="{margin_l}" y="{height - 8}" font-size="12" font-family="Arial" fill="#333">time after jump cut (s)</text>
</svg>
'''
    path.write_text(svg)


def build_samples(
    data_roots: list[Path],
    excluded_subjects: set[str],
    require_source_signature: str | None,
    p00_jumpcut_fallback: bool,
    write_qc_svg: bool,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    svg_dir = out_dir / "segmentation_qc_svg"
    if write_qc_svg:
        svg_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    seen: set[Path] = set()
    for data_root in data_roots:
        for path in sorted(data_root.glob("*.csv")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)

    for path in sorted(paths):
        label = rbc.label_from_name(path)
        if label is None or not is_single_action(label):
            continue
        subject = rbc.subject_from_name(path)
        if subject in excluded_subjects:
            continue

        whole_features, whole_meta = extract_features_with_p00_fallback(path, subject, p00_jumpcut_fallback)
        if whole_features is None:
            invalid.append({"file": path.name, "label": label, "subject": subject, "meta": whole_meta})
            continue
        signature = rbc.source_signature(whole_meta)
        if require_source_signature and signature != require_source_signature:
            continue

        data = rbc.read_csv_sources(path)
        jump_cut_s = float(whole_meta.get("jump_cut_s", 0.0))
        times, envelope, duration, env_meta = motion_envelope(data, jump_cut_s)
        peaks, peak_meta = choose_five_peaks(times, envelope, duration) if len(envelope) else ([], {"status": "no_envelope"})
        if len(peaks) == SINGLE_REPETITIONS:
            windows = windows_from_peaks(times, envelope, peaks, duration)
            segmentation_method = "active_peak_window"
        else:
            windows = equal_fallback_windows(duration)
            segmentation_method = "fallback_equal_time"

        coverage_s = sum(max(0.0, win["end_s"] - win["start_s"]) for win in windows)
        qc = {
            "file": path.name,
            "label": label,
            "subject": subject,
            "source_signature": signature,
            "jump_cut_s": jump_cut_s,
            "duration_s": duration,
            "segmentation_method": segmentation_method,
            "peak_status": peak_meta.get("status"),
            "candidate_peaks": peak_meta.get("candidate_peaks", 0),
            "coverage_s": coverage_s,
            "coverage_pct": coverage_s / duration if duration > 0 else 0.0,
            "window_durations_s": ";".join(f"{win['end_s'] - win['start_s']:.3f}" for win in windows),
            "peak_times_s": ";".join(f"{win['peak_time_s']:.3f}" for win in windows),
            "peak_scores": ";".join(f"{win['peak_score']:.3f}" for win in windows),
        }
        qc_rows.append(qc)

        if write_qc_svg and len(envelope):
            safe_name = path.stem.replace("/", "_")
            svg_plot(svg_dir / f"{safe_name}.svg", times, envelope, windows, f"{path.name} | {label} | {segmentation_method}")

        for rep_idx, win in enumerate(windows, start=1):
            rep_features, rep_meta = features_for_window(
                data,
                jump_cut_s,
                whole_features.get("jump_found", 0.0) > 0,
                win["start_s"],
                win["end_s"],
                base={
                    "single_repetition_sample": 1.0,
                    "repetition_index": float(rep_idx),
                    "repetition_count": float(SINGLE_REPETITIONS),
                    "whole_file_duration_s": duration,
                    "active_peak_time_s": win["peak_time_s"],
                    "active_peak_score": win["peak_score"],
                    "active_window_duration_s": win["end_s"] - win["start_s"],
                    "active_window_coverage_pct": (win["end_s"] - win["start_s"]) / duration if duration > 0 else 0.0,
                    "active_segmentation_method_peak": 1.0 if segmentation_method == "active_peak_window" else 0.0,
                },
            )
            if rep_features is None:
                invalid.append({"file": path.name, "label": label, "subject": subject, "meta": rep_meta, "repetition_index": rep_idx})
                continue
            samples.append(
                {
                    "sample_id": f"{path.stem}__active_rep{rep_idx}",
                    "file": path.name,
                    "label": label,
                    "subject": subject,
                    "sample_type": "single_repetition",
                    "repetition_index": rep_idx,
                    "repetition_count": SINGLE_REPETITIONS,
                    "measurement_group": rbc.measurement_group({"file": path.name, "label": label}),
                    "source_signature": signature,
                    "features": rep_features,
                    "meta": rep_meta,
                    "window": win,
                    "segmentation_method": segmentation_method,
                }
            )

    return samples, invalid, qc_rows


def write_outputs(out_dir: Path, samples: list[dict[str, Any]], invalid: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_cols = [
        "sample_id",
        "file",
        "label",
        "subject",
        "sample_type",
        "repetition_index",
        "repetition_count",
        "measurement_group",
        "source_signature",
        "jump_cut_s",
        "jump_found",
        "segment_start_s",
        "segment_end_s",
        "segment_duration_s",
        "segmentation_method",
        "active_peak_time_s",
        "active_peak_score",
    ]
    feature_names = sorted({name for sample in samples for name in sample["features"].keys() if name not in set(sample_cols)})

    with (out_dir / "segmented_feature_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(sample_cols + feature_names)
        for sample in samples:
            features = sample["features"]
            win = sample["window"]
            base = [
                sample["sample_id"],
                sample["file"],
                sample["label"],
                sample["subject"],
                sample["sample_type"],
                sample["repetition_index"],
                sample["repetition_count"],
                sample["measurement_group"],
                sample["source_signature"],
                features.get("jump_cut_s", 0.0),
                int(features.get("jump_found", 0.0) > 0),
                features.get("segment_start_s", 0.0),
                features.get("segment_end_s", 0.0),
                features.get("segment_duration_s", 0.0),
                sample["segmentation_method"],
                win["peak_time_s"],
                win["peak_score"],
            ]
            writer.writerow(base + [features.get(name, 0.0) for name in feature_names])

    with (out_dir / "active_window_samples.csv").open("w", newline="") as handle:
        fieldnames = sample_cols
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            features = sample["features"]
            win = sample["window"]
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "file": sample["file"],
                    "label": sample["label"],
                    "subject": sample["subject"],
                    "sample_type": sample["sample_type"],
                    "repetition_index": sample["repetition_index"],
                    "repetition_count": sample["repetition_count"],
                    "measurement_group": sample["measurement_group"],
                    "source_signature": sample["source_signature"],
                    "jump_cut_s": features.get("jump_cut_s", 0.0),
                    "jump_found": int(features.get("jump_found", 0.0) > 0),
                    "segment_start_s": features.get("segment_start_s", 0.0),
                    "segment_end_s": features.get("segment_end_s", 0.0),
                    "segment_duration_s": features.get("segment_duration_s", 0.0),
                    "segmentation_method": sample["segmentation_method"],
                    "active_peak_time_s": win["peak_time_s"],
                    "active_peak_score": win["peak_score"],
                }
            )

    if qc_rows:
        with (out_dir / "active_window_qc.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qc_rows)

    label_counts = Counter(sample["label"] for sample in samples)
    subject_counts = Counter(sample["subject"] for sample in samples)
    method_counts = Counter(sample["segmentation_method"] for sample in samples)
    file_method_counts = Counter(row["segmentation_method"] for row in qc_rows)
    coverage_values = [float(row["coverage_pct"]) for row in qc_rows]
    metrics = {
        "sample_count": len(samples),
        "file_count": len(qc_rows),
        "invalid_count": len(invalid),
        "labels": dict(label_counts),
        "subjects": dict(subject_counts),
        "sample_segmentation_methods": dict(method_counts),
        "file_segmentation_methods": dict(file_method_counts),
        "coverage_pct_median": float(statistics.median(coverage_values)) if coverage_values else 0.0,
        "coverage_pct_min": min(coverage_values) if coverage_values else 0.0,
        "coverage_pct_max": max(coverage_values) if coverage_values else 0.0,
    }
    (out_dir / "active_window_dataset_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Active-Window Single-Action Dataset Summary",
        "",
        "## Criteria",
        "",
        "- P01 excluded",
        "- SHIM excluded",
        "- full-sensor only: `AirPods+Watch+iPhone`",
        "- Combinations excluded",
        "- For each single-action file, find 5 peaks in the post-jump motion envelope and extract only the active windows",
        "- Unlike the older equal 5-way split, pauses are excluded as much as possible",
        "",
        "## Sample counts",
        "",
        f"- Files: {metrics['file_count']}",
        f"- Repetition samples: {metrics['sample_count']}",
        f"- invalid segment/file: {metrics['invalid_count']}",
        f"- segmentation method: {metrics['file_segmentation_methods']}",
        f"- active window coverage median: {metrics['coverage_pct_median']:.1%}",
        "",
        "## Samples per label",
        "",
        "| label | samples |",
        "|---|---:|",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `segmented_feature_table.csv`: active window feature table",
            "- `active_window_samples.csv`: active-window start/end per sample",
            "- `active_window_qc.csv`: per-file peak/window QC",
            "- `segmentation_qc_svg/`: per-file envelope and window plots",
            "- `active_window_dataset_metrics.json`: overall summary",
        ]
    )
    (out_dir / "ACTIVE_WINDOW_SEGMENTATION_SUMMARY_KO.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-subject", action="append", default=["P01"])
    parser.add_argument("--require-source-signature", default=FULL_SENSOR_SIGNATURE)
    parser.add_argument("--p00-jumpcut-fallback", action="store_true")
    parser.add_argument("--no-qc-svg", action="store_true")
    args = parser.parse_args()

    samples, invalid, qc_rows = build_samples(
        [data_root.resolve() for data_root in args.data_roots],
        set(args.exclude_subject),
        args.require_source_signature,
        args.p00_jumpcut_fallback,
        write_qc_svg=not args.no_qc_svg,
        out_dir=args.out.resolve(),
    )
    write_outputs(args.out.resolve(), samples, invalid, qc_rows)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "samples": len(samples),
                "files": len(qc_rows),
                "invalid": len(invalid),
                "methods": dict(Counter(row["segmentation_method"] for row in qc_rows)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
