#!/usr/bin/env python3
"""Build a sample table where single-action files are split into five repetitions."""

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

import analysis.active_window_classifier_minimal.rule_base_classifier as rbc


SINGLE_REPETITIONS = 5
P00_FALLBACK_PRIORITY = ("iPhone", "Watch", "AirPods")


def is_single_action(label: str) -> bool:
    return "-" not in label


def relative_duration(data: dict[str, list[tuple[float, float, float, float, float, float, float]]], jump_cut_s: float) -> float:
    durations = []
    for rows in data.values():
        if len(rows) < 5:
            continue
        t0 = rows[0][0]
        end = rows[-1][0] - t0 - jump_cut_s
        if end > 0:
            durations.append(float(end))
    return max(durations) if durations else 0.0


def p00_fallback_jump_cut(
    data: dict[str, list[tuple[float, float, float, float, float, float, float]]],
) -> tuple[float, dict[str, Any] | None]:
    """Looser P00-only fallback when the default sync-jump rule finds nothing.

    P00's initial synchronization jump is often below the default 2.15 g floor.
    The fallback keeps the same early-window logic but lowers the source-specific
    threshold and still requires a local acceleration-magnitude peak.
    """

    candidates: list[tuple[int, float, dict[str, Any]]] = []
    source_params = {
        "iPhone": (1.25, 1.8),
        "Watch": (1.25, 2.0),
        "AirPods": (1.35, 2.5),
    }

    for priority, source in enumerate(P00_FALLBACK_PRIORITY):
        rows = data.get(source, [])
        if len(rows) < 10:
            continue
        t0 = rows[0][0]
        duration = rows[-1][0] - t0
        if duration <= 0:
            continue

        early_limit = min(5.0, max(1.5, duration * 0.50))
        samples: list[tuple[float, float, float]] = []
        for row in rows:
            tt = row[0] - t0
            if tt > early_limit:
                break
            ax, ay, az, gx, gy, gz = row[1:]
            acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
            gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
            samples.append((tt, acc_mag, gyro_mag))
        if len(samples) < 6:
            continue

        acc = [x[1] for x in samples]
        median = statistics.median(acc)
        mad = statistics.median(abs(x - median) for x in acc) or 0.05
        floor, mad_factor = source_params[source]
        threshold = max(floor, median + mad_factor * mad)

        for i in range(1, len(samples) - 1):
            if samples[i][1] >= threshold and samples[i][1] >= samples[i - 1][1] and samples[i][1] >= samples[i + 1][1]:
                jump_t, jump_acc, jump_gyro = samples[i]
                meta = {
                    "source": source,
                    "jump_time_s": jump_t,
                    "jump_acc_mag": jump_acc,
                    "jump_gyro_mag": jump_gyro,
                    "threshold": threshold,
                    "early_window_s": early_limit,
                    "profile": "p00_loose_fallback",
                    "median": median,
                    "mad": mad,
                }
                candidates.append((priority, jump_t, meta))
                break

    if not candidates:
        return 0.0, None

    _, jump_t, meta = min(candidates, key=lambda item: (item[0], item[1]))
    return jump_t + 0.55, meta


def extract_features_with_forced_jump(
    path: Path,
    jump_cut_s: float,
    jump_meta: dict[str, Any] | None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    data = rbc.read_csv_sources(path)
    total_rows = sum(len(rows) for rows in data.values())
    meta: dict[str, Any] = {
        "file": path.name,
        "source_counts": {source: len(data.get(source, [])) for source in rbc.SOURCES if data.get(source)},
        "total_rows": total_rows,
        "jump": jump_meta,
        "jump_cut_s": jump_cut_s,
    }
    if total_rows < 5:
        meta["status"] = "invalid_too_few_rows"
        return None, meta

    features: dict[str, float] = {
        "jump_cut_s": float(jump_cut_s),
        "jump_found": 1.0 if jump_meta else 0.0,
        "total_rows": float(total_rows),
    }
    for source in rbc.SOURCES:
        rbc.add_source_features(features, source, data.get(source, []), jump_cut_s)
    if not any(features.get(f"{source[0].lower()}_n", 0.0) >= 5 for source in rbc.SOURCES):
        meta["status"] = "invalid_no_source_segment"
        return None, meta

    meta["status"] = "ok"
    return features, meta


def extract_features_with_p00_fallback(path: Path, subject: str, enabled: bool) -> tuple[dict[str, float] | None, dict[str, Any]]:
    features, meta = rbc.extract_features(path)
    if not enabled or subject != "P00" or features is None or features.get("jump_found", 0.0) > 0:
        return features, meta

    data = rbc.read_csv_sources(path)
    fallback_cut_s, fallback_meta = p00_fallback_jump_cut(data)
    if fallback_meta is None:
        return features, meta
    return extract_features_with_forced_jump(path, fallback_cut_s, fallback_meta)


def crop_source_rows(
    rows: list[tuple[float, float, float, float, float, float, float]],
    jump_cut_s: float,
    start_s: float,
    end_s: float,
) -> list[tuple[float, float, float, float, float, float, float]]:
    if len(rows) < 5:
        return []
    t0 = rows[0][0]
    return [row for row in rows if start_s <= row[0] - t0 - jump_cut_s < end_s]


def features_for_window(
    data: dict[str, list[tuple[float, float, float, float, float, float, float]]],
    jump_cut_s: float,
    jump_found: bool,
    start_s: float,
    end_s: float,
    base: dict[str, float] | None = None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    cropped = {
        source: crop_source_rows(rows, jump_cut_s, start_s, end_s)
        for source, rows in data.items()
    }
    total_rows = sum(len(rows) for rows in cropped.values())
    features: dict[str, float] = {
        "jump_cut_s": float(jump_cut_s),
        "jump_found": 1.0 if jump_found else 0.0,
        "total_rows": float(total_rows),
        "segment_start_s": float(start_s),
        "segment_end_s": float(end_s),
        "segment_duration_s": float(max(0.0, end_s - start_s)),
    }
    if base:
        features.update(base)

    for source in rbc.SOURCES:
        rbc.add_source_features(features, source, cropped.get(source, []), 0.0)

    meta = {
        "source_counts": {source: len(cropped.get(source, [])) for source in rbc.SOURCES if cropped.get(source)},
        "total_rows": total_rows,
        "jump_cut_s": jump_cut_s,
        "status": "ok" if total_rows >= 5 else "invalid_too_few_rows",
    }
    if not any(features.get(f"{source[0].lower()}_n", 0.0) >= 5 for source in rbc.SOURCES):
        meta["status"] = "invalid_no_source_segment"
        return None, meta
    return features, meta


def build_samples(
    data_roots: list[Path],
    excluded_subjects: set[str],
    require_source_signature: str | None = None,
    p00_jumpcut_fallback: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
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
        if label is None:
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

        base_row = {"file": path.name, "label": label, "features": whole_features, "meta": whole_meta, "subject": subject}
        measurement_group = rbc.measurement_group(base_row)

        if not is_single_action(label):
            sample = {
                "sample_id": f"{path.stem}__combo",
                "file": path.name,
                "label": label,
                "subject": subject,
                "sample_type": "combination_sequence",
                "repetition_index": 0,
                "repetition_count": 1,
                "measurement_group": measurement_group,
                "source_signature": signature,
                "features": dict(whole_features, single_repetition_sample=0.0, repetition_index=0.0, repetition_count=1.0),
                "meta": whole_meta,
            }
            samples.append(sample)
            continue

        data = rbc.read_csv_sources(path)
        jump_cut_s = float(whole_meta.get("jump_cut_s", 0.0))
        duration = relative_duration(data, jump_cut_s)
        if duration <= 0:
            invalid.append({"file": path.name, "label": label, "subject": subject, "meta": {"status": "invalid_no_duration"}})
            continue

        for rep_idx in range(SINGLE_REPETITIONS):
            start_s = duration * rep_idx / SINGLE_REPETITIONS
            end_s = duration * (rep_idx + 1) / SINGLE_REPETITIONS
            rep_features, rep_meta = features_for_window(
                data,
                jump_cut_s,
                whole_features.get("jump_found", 0.0) > 0,
                start_s,
                end_s,
                base={
                    "single_repetition_sample": 1.0,
                    "repetition_index": float(rep_idx + 1),
                    "repetition_count": float(SINGLE_REPETITIONS),
                    "whole_file_duration_s": duration,
                },
            )
            if rep_features is None:
                invalid.append({"file": path.name, "label": label, "subject": subject, "meta": rep_meta, "repetition_index": rep_idx + 1})
                continue
            samples.append(
                {
                    "sample_id": f"{path.stem}__rep{rep_idx + 1}",
                    "file": path.name,
                    "label": label,
                    "subject": subject,
                    "sample_type": "single_repetition",
                    "repetition_index": rep_idx + 1,
                    "repetition_count": SINGLE_REPETITIONS,
                    "measurement_group": measurement_group,
                    "source_signature": signature,
                    "features": rep_features,
                    "meta": rep_meta,
                }
            )
    return samples, invalid


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)

    rows = []
    for label in sorted(by_label):
        group = by_label[label]
        source_counts = Counter(sample["source_signature"] for sample in group)
        sample_types = Counter(sample["sample_type"] for sample in group)
        subjects = Counter(sample["subject"] for sample in group)
        features = [sample["features"] for sample in group]
        rows.append(
            {
                "label": label,
                "action_type": "single" if is_single_action(label) else "combination",
                "sample_count": len(group),
                "source_signatures": "; ".join(f"{k}:{v}" for k, v in source_counts.most_common()),
                "sample_types": "; ".join(f"{k}:{v}" for k, v in sample_types.most_common()),
                "subjects": "; ".join(f"{k}:{v}" for k, v in sorted(subjects.items())),
                "jump_found_pct": sum(1 for item in features if item.get("jump_found", 0.0) > 0) / len(features),
                "zero_cut_pct": sum(1 for item in features if abs(item.get("jump_cut_s", 0.0)) < 1e-9) / len(features),
                "no_watch_pct": sum(1 for item in features if item.get("w_raw_n", 0.0) < 5) / len(features),
                "w_dur_median": median([item.get("w_dur", 0.0) for item in features]),
                "w_am_peak_count_gt2_median": median([item.get("w_am_peak_count_gt2", 0.0) for item in features]),
                "w_am_peak_count_gt5_median": median([item.get("w_am_peak_count_gt5", 0.0) for item in features]),
                "w_gm_peak_count_gt8_median": median([item.get("w_gm_peak_count_gt8", 0.0) for item in features]),
                "w_am_absmax_median": median([item.get("w_am_absmax", 0.0) for item in features]),
                "w_gm_absmax_median": median([item.get("w_gm_absmax", 0.0) for item in features]),
            }
        )
    return rows


def write_outputs(
    out_dir: Path,
    samples: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    require_source_signature: str | None,
    p00_jumpcut_fallback: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({name for sample in samples for name in sample["features"].keys()})

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
    ]
    with (out_dir / "segmented_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_cols)
        writer.writeheader()
        for sample in samples:
            features = sample["features"]
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
                }
            )

    with (out_dir / "segmented_feature_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(sample_cols + feature_names)
        for sample in samples:
            features = sample["features"]
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
            ]
            writer.writerow(base + [features.get(name, 0.0) for name in feature_names])

    summary_rows = summarize(samples)
    with (out_dir / "segmented_label_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    metrics = {
        "excluded_subjects": ["P01"],
        "source_filter": require_source_signature or "all_valid_sources",
        "p00_jumpcut_fallback": p00_jumpcut_fallback,
        "sample_count": len(samples),
        "invalid_segment_or_file_count": len(invalid),
        "single_repetition_samples": sum(1 for sample in samples if sample["sample_type"] == "single_repetition"),
        "combination_sequence_samples": sum(1 for sample in samples if sample["sample_type"] == "combination_sequence"),
        "labels": dict(Counter(sample["label"] for sample in samples)),
        "subjects": dict(Counter(sample["subject"] for sample in samples)),
        "source_signatures": dict(Counter(sample["source_signature"] for sample in samples)),
    }
    (out_dir / "segmented_dataset_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Single-Action 5-Split Dataset Summary",
        "",
        "## Criteria",
        "",
        "- Excluded: `P01`",
        f"- source filter: `{metrics['source_filter']}`",
        f"- P00 jump cut fallback: `{p00_jumpcut_fallback}`",
        "- Single action: each file is split into 5 repetition samples",
        "- Combination: each file is kept as one sequence sample",
        "- Split method: the whole post-jump segment is divided into 5 equal time slices",
        "",
        "## Sample counts",
        "",
        f"- Total samples: {metrics['sample_count']}",
        f"- single repetition sample: {metrics['single_repetition_samples']}",
        f"- combination sequence sample: {metrics['combination_sequence_samples']}",
        f"- invalid segment/file: {metrics['invalid_segment_or_file_count']}",
        "",
        "## Main single actions",
        "",
        "| label | samples | source | no Watch | jump fail | Watch dur median | Watch acc peaks>2g median | Watch acc peaks>5g median |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["action_type"] == "single":
            lines.append(
                f"| {row['label']} | {row['sample_count']} | {row['source_signatures']} | "
                f"{row['no_watch_pct']:.0%} | {row['zero_cut_pct']:.0%} | {row['w_dur_median']:.2f} | "
                f"{row['w_am_peak_count_gt2_median']:.1f} | {row['w_am_peak_count_gt5_median']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `segmented_samples.csv`: per-sample metadata",
            "- `segmented_feature_table.csv`: feature table for model training",
            "- `segmented_label_summary.csv`: per-label sample/quality summary",
            "- `segmented_dataset_metrics.json`: overall count summary",
        ]
    )
    (out_dir / "SEGMENTED_REPETITION_SUMMARY_KO.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-subject", action="append", default=["P01"])
    parser.add_argument("--require-source-signature", default=None)
    parser.add_argument("--p00-jumpcut-fallback", action="store_true", help="use a loose P00-only fallback when default jump detection fails")
    args = parser.parse_args()

    samples, invalid = build_samples(
        [data_root.resolve() for data_root in args.data_roots],
        set(args.exclude_subject),
        args.require_source_signature,
        p00_jumpcut_fallback=args.p00_jumpcut_fallback,
    )
    write_outputs(args.out.resolve(), samples, invalid, args.require_source_signature, args.p00_jumpcut_fallback)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "samples": len(samples),
                "invalid": len(invalid),
                "source_filter": args.require_source_signature or "all_valid_sources",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
