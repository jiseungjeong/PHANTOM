#!/usr/bin/env python3
"""Build and apply a rule-based motion classifier for Phantom CSV exports.

The script intentionally avoids pandas/sklearn so it can run in this folder with
only Python + numpy. It derives a deterministic CART-style if/then tree from the
CSV labels, after removing the initial synchronization jump.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

import numpy as np


SUBJECTS = {"P00", "P01", "JUHO", "JIHO", "SHIM"}
SOURCES = ("Watch", "iPhone", "AirPods")
JUMP_SOURCE_PRIORITY = ("iPhone", "AirPods", "Watch")


def label_from_name(path: Path) -> str | None:
    match = re.match(r"^(.*)_(\d{8})_(\d{6})$", path.stem)
    if not match:
        return None

    parts = match.group(1).split("_")
    if parts and parts[0] in SUBJECTS:
        parts = parts[1:]
    if parts and parts[-1] == "Watch":
        parts = parts[:-1]

    label = " ".join(parts).replace("-", " - ")
    return re.sub(r"\s+", " ", label).strip().replace(" - ", "-")


def subject_from_name(path: Path) -> str:
    first = path.stem.split("_")[0]
    return first if first in SUBJECTS else "UNKNOWN"


def read_csv_sources(path: Path) -> dict[str, list[tuple[float, float, float, float, float, float, float]]]:
    data: dict[str, list[tuple[float, float, float, float, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                t = float(row["timestamp"])
                values = tuple(float(row[k]) for k in ("accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"))
            except (KeyError, TypeError, ValueError):
                continue
            data[row["source"]].append((t, *values))

    for rows in data.values():
        rows.sort(key=lambda row: row[0])
    return dict(data)


def local_peaks(t: np.ndarray, values: np.ndarray, threshold: float, min_separation_s: float = 0.28) -> list[tuple[float, float]]:
    peaks: list[tuple[float, float]] = []
    if len(values) < 3:
        return peaks

    for i in range(1, len(values) - 1):
        if values[i] < threshold:
            continue
        if values[i] < values[i - 1] or values[i] < values[i + 1]:
            continue

        if peaks and t[i] - peaks[-1][0] < min_separation_s:
            if values[i] > peaks[-1][1]:
                peaks[-1] = (float(t[i]), float(values[i]))
        else:
            peaks.append((float(t[i]), float(values[i])))
    return peaks


def detect_jump_cut(data: dict[str, list[tuple[float, float, float, float, float, float, float]]]) -> tuple[float, dict[str, Any] | None]:
    """Return seconds after source start to begin classification.

    The initial jump is searched in iPhone/AirPods first because body/head
    acceleration is less confounded by punch impacts. Watch is the fallback.
    """

    candidates: list[tuple[int, float, dict[str, Any]]] = []

    for priority, source in enumerate(JUMP_SOURCE_PRIORITY):
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
        threshold = max(2.15, median + 2.5 * mad)

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
                }
                candidates.append((priority, jump_t, meta))
                break

    if not candidates:
        return 0.0, None

    _, jump_t, meta = min(candidates, key=lambda item: (item[0], item[1]))
    return jump_t + 0.55, meta


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else 0.0


def add_source_features(
    features: dict[str, float],
    source: str,
    rows: list[tuple[float, float, float, float, float, float, float]],
    jump_cut_s: float,
) -> None:
    prefix = source[0].lower()
    features[f"{prefix}_present"] = 1.0 if rows else 0.0
    features[f"{prefix}_raw_n"] = float(len(rows))

    if len(rows) < 5:
        return

    t0 = rows[0][0]
    segment = [row for row in rows if row[0] - t0 >= jump_cut_s]
    if len(segment) < 5:
        segment = rows[-min(20, len(rows)) :]

    arr = np.array([[row[0] - t0 - jump_cut_s, *row[1:]] for row in segment], dtype=float)
    t = arr[:, 0]
    ax, ay, az, gx, gy, gz = (arr[:, i] for i in range(1, 7))
    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    features[f"{prefix}_n"] = float(len(segment))
    features[f"{prefix}_dur"] = float(np.max(t) - np.min(t)) if len(t) > 1 else 0.0

    series = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "am": acc_mag,
        "gm": gyro_mag,
    }
    for name, values in series.items():
        features[f"{prefix}_{name}_mean"] = float(np.mean(values))
        features[f"{prefix}_{name}_std"] = float(np.std(values))
        features[f"{prefix}_{name}_min"] = float(np.min(values))
        features[f"{prefix}_{name}_max"] = float(np.max(values))
        features[f"{prefix}_{name}_absmax"] = float(np.max(np.abs(values)))
        features[f"{prefix}_{name}_energy"] = float(np.mean(values * values))
        for q in (5, 25, 50, 75, 95, 99):
            features[f"{prefix}_{name}_p{q:02d}"] = percentile(values, q)

    for axis_name, values in (("ax", ax), ("ay", ay), ("az", az), ("gx", gx), ("gy", gy), ("gz", gz)):
        positive = np.maximum(values, 0.0)
        negative = np.maximum(-values, 0.0)
        features[f"{prefix}_{axis_name}_pos_p95"] = percentile(positive, 95)
        features[f"{prefix}_{axis_name}_neg_p95"] = percentile(negative, 95)
        features[f"{prefix}_{axis_name}_pos_energy"] = float(np.mean(positive * positive))
        features[f"{prefix}_{axis_name}_neg_energy"] = float(np.mean(negative * negative))

    peak_indices = {"peakg": int(np.argmax(gyro_mag)), "peaka": int(np.argmax(acc_mag))}
    for peak_name, idx in peak_indices.items():
        for name, values in {
            "t": t,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "am": acc_mag,
            "gm": gyro_mag,
        }.items():
            features[f"{prefix}_{peak_name}_{name}"] = float(values[idx])

    gyro_median = float(np.median(gyro_mag))
    gyro_mad = float(np.median(np.abs(gyro_mag - gyro_median))) or 0.05
    acc_median = float(np.median(acc_mag))
    acc_mad = float(np.median(np.abs(acc_mag - acc_median))) or 0.05
    features[f"{prefix}_gm_peak_count_adapt"] = float(len(local_peaks(t, gyro_mag, max(3.0, gyro_median + 3.0 * gyro_mad))))
    features[f"{prefix}_am_peak_count_adapt"] = float(len(local_peaks(t, acc_mag, max(2.2, acc_median + 3.0 * acc_mad))))

    for threshold in (2, 3, 5, 8, 12, 16, 20, 25, 30):
        features[f"{prefix}_gm_peak_count_gt{threshold}"] = float(len(local_peaks(t, gyro_mag, float(threshold))))

    for threshold in (1.5, 2, 3, 5, 8, 12, 16, 20):
        key = str(threshold).replace(".", "p")
        features[f"{prefix}_am_peak_count_gt{key}"] = float(len(local_peaks(t, acc_mag, float(threshold))))


def extract_features(path: Path) -> tuple[dict[str, float] | None, dict[str, Any]]:
    data = read_csv_sources(path)
    total_rows = sum(len(rows) for rows in data.values())
    meta: dict[str, Any] = {
        "file": path.name,
        "source_counts": {source: len(data.get(source, [])) for source in SOURCES if data.get(source)},
        "total_rows": total_rows,
    }
    if total_rows < 5:
        meta["status"] = "invalid_too_few_rows"
        return None, meta

    jump_cut_s, jump_meta = detect_jump_cut(data)
    features: dict[str, float] = {
        "jump_cut_s": float(jump_cut_s),
        "jump_found": 1.0 if jump_meta else 0.0,
        "total_rows": float(total_rows),
    }
    meta["jump"] = jump_meta
    meta["jump_cut_s"] = jump_cut_s

    for source in SOURCES:
        add_source_features(features, source, data.get(source, []), jump_cut_s)

    if not any(features.get(f"{source[0].lower()}_n", 0.0) >= 5 for source in SOURCES):
        meta["status"] = "invalid_no_source_segment"
        return None, meta

    meta["status"] = "ok"
    return features, meta


@dataclass
class Node:
    prediction: int
    counts: np.ndarray
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class RuleTree:
    def __init__(self, feature_names: list[str], labels: list[str], max_depth: int = 15, min_samples: int = 2) -> None:
        self.feature_names = feature_names
        self.labels = labels
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Node | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = x
        self._y = y
        self.root = self._build(np.arange(len(y)), 0)

    def _gini(self, indices: np.ndarray) -> float:
        if self._y is None or len(indices) == 0:
            return 0.0
        counts = np.bincount(self._y[indices], minlength=len(self.labels)) / len(indices)
        return 1.0 - float(np.sum(counts * counts))

    def _best_split(self, indices: np.ndarray) -> tuple[float, int, float] | None:
        assert self._x is not None and self._y is not None

        base = self._gini(indices)
        best: tuple[float, int, float] | None = None
        n = len(indices)
        y_subset = self._y[indices]
        class_count = len(self.labels)

        for feature_index in range(self._x.shape[1]):
            values = self._x[indices, feature_index]
            order = np.argsort(values)
            sorted_values = values[order]
            sorted_y = y_subset[order]
            candidates = np.where(sorted_values[:-1] != sorted_values[1:])[0]
            if len(candidates) > 80:
                candidates = candidates[np.linspace(0, len(candidates) - 1, 80, dtype=int)]
            candidate_set = set(int(x) for x in candidates)

            left_counts = np.zeros(class_count, dtype=int)
            right_counts = np.bincount(sorted_y, minlength=class_count)
            for split_pos in range(len(sorted_values) - 1):
                cls = sorted_y[split_pos]
                left_counts[cls] += 1
                right_counts[cls] -= 1
                if split_pos not in candidate_set:
                    continue

                left_n = split_pos + 1
                right_n = n - left_n
                if left_n < 2 or right_n < 2:
                    continue

                left_gini = 1.0 - float(np.sum((left_counts / left_n) ** 2))
                right_gini = 1.0 - float(np.sum((right_counts / right_n) ** 2))
                gain = base - (left_n / n) * left_gini - (right_n / n) * right_gini
                threshold = float((sorted_values[split_pos] + sorted_values[split_pos + 1]) / 2.0)

                if best is None or gain > best[0]:
                    best = (float(gain), feature_index, threshold)

        return best

    def _build(self, indices: np.ndarray, depth: int) -> Node:
        assert self._x is not None and self._y is not None

        counts = np.bincount(self._y[indices], minlength=len(self.labels))
        prediction = int(np.argmax(counts))
        node = Node(prediction=prediction, counts=counts, depth=depth)

        if depth >= self.max_depth or len(indices) < self.min_samples or np.max(counts) == len(indices):
            return node

        split = self._best_split(indices)
        if split is None or split[0] <= 1e-9:
            return node

        _, feature_index, threshold = split
        left_indices = indices[self._x[indices, feature_index] <= threshold]
        right_indices = indices[self._x[indices, feature_index] > threshold]
        if len(left_indices) < 1 or len(right_indices) < 1:
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build(left_indices, depth + 1)
        node.right = self._build(right_indices, depth + 1)
        return node

    def predict_one(self, row: np.ndarray) -> tuple[int, float, list[dict[str, Any]]]:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")

        node = self.root
        trace: list[dict[str, Any]] = []
        while node.feature_index is not None:
            assert node.threshold is not None and node.left is not None and node.right is not None
            value = float(row[node.feature_index])
            go_left = value <= node.threshold
            trace.append(
                {
                    "feature": self.feature_names[node.feature_index],
                    "value": value,
                    "op": "<=" if go_left else ">",
                    "threshold": node.threshold,
                }
            )
            node = node.left if go_left else node.right

        total = int(np.sum(node.counts))
        confidence = float(node.counts[node.prediction] / total) if total else 0.0
        return node.prediction, confidence, trace

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(row)[0] for row in x], dtype=int)

    def to_dict(self, node: Node | None = None) -> dict[str, Any]:
        if node is None:
            if self.root is None:
                raise RuntimeError("Tree has not been fitted")
            node = self.root

        counts = {self.labels[i]: int(count) for i, count in enumerate(node.counts) if int(count)}
        total = int(np.sum(node.counts))
        leaf_info = {
            "prediction": self.labels[node.prediction],
            "confidence": float(node.counts[node.prediction] / total) if total else 0.0,
            "n": total,
            "counts": counts,
        }

        if node.feature_index is None:
            return {"type": "leaf", **leaf_info}

        assert node.threshold is not None and node.left is not None and node.right is not None
        return {
            "type": "split",
            **leaf_info,
            "feature": self.feature_names[node.feature_index],
            "threshold": node.threshold,
            "left": self.to_dict(node.left),
            "right": self.to_dict(node.right),
        }

    def text(self, max_depth: int = 6) -> str:
        if self.root is None:
            return ""

        lines: list[str] = []

        def walk(node: Node, indent: str) -> None:
            total = int(np.sum(node.counts))
            purity = float(node.counts[node.prediction] / total) if total else 0.0
            if node.feature_index is None or node.depth >= max_depth:
                lines.append(f"{indent}-> {self.labels[node.prediction]}  n={total}  purity={purity:.2f}")
                return

            assert node.threshold is not None and node.left is not None and node.right is not None
            feature = self.feature_names[node.feature_index]
            lines.append(f"{indent}if {feature} <= {node.threshold:.5g}:")
            walk(node.left, indent + "  ")
            lines.append(f"{indent}else:")
            walk(node.right, indent + "  ")

        walk(self.root, "")
        return "\n".join(lines)


def make_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.array([[row["features"].get(name, 0.0) for name in feature_names] for row in rows], dtype=float)


def stratified_folds(labels: list[str], k: int = 5, seed: int = 7) -> list[list[int]]:
    rng = Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    folds = [[] for _ in range(k)]
    for indices in by_label.values():
        rng.shuffle(indices)
        for offset, idx in enumerate(indices):
            folds[offset % k].append(idx)
    return folds


def measurement_group(row: dict[str, Any]) -> str:
    match = re.search(r"_(\d{8})_(\d{6})$", Path(row["file"]).stem)
    timestamp = f"{match.group(1)}_{match.group(2)}" if match else Path(row["file"]).stem
    return f"{row['label']}|{timestamp}"


def source_priority(signature: str) -> int:
    return {
        "AirPods+Watch+iPhone": 4,
        "AirPods+iPhone": 3,
        "iPhone": 2,
        "Watch": 1,
    }.get(signature, 0)


def representative_measurement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[measurement_group(row)].append(row)

    representatives = []
    for group_rows in grouped.values():
        representatives.append(max(group_rows, key=lambda row: source_priority(source_signature(row["meta"]))))
    return representatives


def measurement_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[measurement_group(row)].append(row)

    signature_combos = Counter(
        tuple(sorted(source_signature(row["meta"]) for row in group_rows)) for group_rows in grouped.values()
    )
    return {
        "file_count": len(rows),
        "measurement_group_count": len(grouped),
        "group_size_counts": dict(Counter(len(group_rows) for group_rows in grouped.values())),
        "signature_combo_counts": {" | ".join(combo): count for combo, count in signature_combos.most_common()},
        "representative_count": len(representative_measurement_rows(rows)),
    }


def grouped_measurement_folds(rows: list[dict[str, Any]], k: int = 5, seed: int = 7) -> list[list[int]]:
    rng = Random(seed)
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        group_to_indices[measurement_group(row)].append(idx)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group, indices in group_to_indices.items():
        groups_by_label[rows[indices[0]]["label"]].append(group)

    folds = [[] for _ in range(k)]
    for groups in groups_by_label.values():
        rng.shuffle(groups)
        for offset, group in enumerate(groups):
            folds[offset % k].extend(group_to_indices[group])
    return folds


def feature_pool_indices(feature_names: list[str], pool: str) -> np.ndarray:
    if pool == "watch":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("w_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "all":
        return np.arange(len(feature_names), dtype=int)
    if pool == "iphone":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("i_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "airpods":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("a_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "iphone_airpods":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("i_") or name.startswith("a_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "watch_iphone":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("w_") or name.startswith("i_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "watch_airpods":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if name.startswith("w_") or name.startswith("a_") or name.startswith("jump") or name == "total_rows"
            ],
            dtype=int,
        )
    if pool == "motion_all":
        return np.array(
            [
                idx
                for idx, name in enumerate(feature_names)
                if "present" not in name and "raw_n" not in name and name != "total_rows"
            ],
            dtype=int,
        )
    if pool == "no_jump":
        return np.array([idx for idx, name in enumerate(feature_names) if not name.startswith("jump")], dtype=int)
    raise ValueError(f"unknown feature pool: {pool}")


def anova_f_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    overall = np.mean(x, axis=0)
    between = np.zeros(x.shape[1], dtype=float)
    within = np.zeros(x.shape[1], dtype=float)

    for cls in np.unique(y):
        cls_x = x[y == cls]
        if len(cls_x) == 0:
            continue
        cls_mean = np.mean(cls_x, axis=0)
        between += len(cls_x) * (cls_mean - overall) ** 2
        within += np.sum((cls_x - cls_mean) ** 2, axis=0)

    return between / (within + 1e-9)


def robust_scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = q75 - q25
    small = scale < 1e-9
    if np.any(small):
        scale[small] = np.std(x[:, small], axis=0) + 1e-6
    return median, scale


def robust_scale_transform(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - median) / scale


def fit_nearest_neighbor_model(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    labels: list[str],
    feature_pool: str = "all",
    top_n: int = 180,
) -> dict[str, Any]:
    label_to_int = {label: idx for idx, label in enumerate(labels)}
    x = make_matrix(rows, feature_names)
    y = np.array([label_to_int[row["label"]] for row in rows], dtype=int)

    pool_indices = feature_pool_indices(feature_names, feature_pool)
    scores = anova_f_scores(x[:, pool_indices], y)
    if top_n < len(pool_indices):
        selected_pool_offsets = np.argsort(scores)[-top_n:]
        selected_indices = pool_indices[selected_pool_offsets]
        selected_scores = scores[selected_pool_offsets]
    else:
        selected_indices = pool_indices
        selected_scores = scores

    order = np.argsort(-selected_scores)
    selected_indices = selected_indices[order]
    selected_scores = selected_scores[order]

    selected_x = x[:, selected_indices]
    median, scale = robust_scale_fit(selected_x)
    vectors = robust_scale_transform(selected_x, median, scale)

    return {
        "type": "selected_feature_1nn",
        "feature_pool": feature_pool,
        "top_n": int(min(top_n, len(pool_indices))),
        "distance": "mean_absolute_robust_scaled",
        "selected_features": [feature_names[idx] for idx in selected_indices],
        "selected_feature_scores": [float(score) for score in selected_scores],
        "median": [float(v) for v in median],
        "scale": [float(v) for v in scale],
        "vectors": [[float(v) for v in row] for row in vectors],
        "labels": [row["label"] for row in rows],
        "files": [row["file"] for row in rows],
    }


def predict_nearest_neighbor(model: dict[str, Any], features: dict[str, float]) -> tuple[str, float, list[dict[str, Any]]]:
    feature_names = list(model["selected_features"])
    values = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
    median = np.array(model["median"], dtype=float)
    scale = np.array(model["scale"], dtype=float)
    vector = robust_scale_transform(values, median, scale)
    train_vectors = np.array(model["vectors"], dtype=float)
    distances = np.mean(np.abs(train_vectors - vector), axis=1)
    nearest = int(np.argmin(distances))
    best_distance = float(distances[nearest])
    confidence = 1.0 / (1.0 + best_distance)
    trace = [
        {
            "model": model["type"],
            "neighbor_file": model["files"][nearest],
            "neighbor_label": model["labels"][nearest],
            "distance": best_distance,
            "features_used": len(feature_names),
        }
    ]
    return str(model["labels"][nearest]), confidence, trace


def source_signature_from_features(features: dict[str, float]) -> str:
    present = []
    if features.get("a_raw_n", 0.0) >= 5:
        present.append("AirPods")
    if features.get("w_raw_n", 0.0) >= 5:
        present.append("Watch")
    if features.get("i_raw_n", 0.0) >= 5:
        present.append("iPhone")
    return "+".join(sorted(present)) if present else "too_short"


def fit_source_hybrid_nearest_neighbor_models(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    labels: list[str],
    watch_top_n: int = 100,
    all_top_n: int = 320,
) -> dict[str, Any]:
    return {
        "type": "source_signature_hybrid_1nn",
        "strategy": f"use watch_{watch_top_n} for Watch-only files; use all_{all_top_n} otherwise",
        "watch_top_n": watch_top_n,
        "all_top_n": all_top_n,
        "watch_model": fit_nearest_neighbor_model(rows, feature_names, labels, feature_pool="watch", top_n=watch_top_n),
        "all_model": fit_nearest_neighbor_model(rows, feature_names, labels, feature_pool="all", top_n=all_top_n),
    }


def predict_source_hybrid_nearest_neighbor(
    models: dict[str, Any],
    features: dict[str, float],
) -> tuple[str, float, list[dict[str, Any]]]:
    signature = source_signature_from_features(features)
    if signature == "Watch":
        prediction, confidence, trace = predict_nearest_neighbor(models["watch_model"], features)
        trace[0]["source_strategy"] = "watch_only"
        return prediction, confidence, trace

    prediction, confidence, trace = predict_nearest_neighbor(models["all_model"], features)
    trace[0]["source_strategy"] = "all_sensor_or_missing_watch"
    return prediction, confidence, trace


def evaluate_cv(rows: list[dict[str, Any]], feature_names: list[str], labels: list[str]) -> dict[str, Any]:
    label_to_int = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_int[row["label"]] for row in rows], dtype=int)
    x = make_matrix(rows, feature_names)
    folds = stratified_folds([row["label"] for row in rows], k=5)

    predictions = np.full(len(rows), -1, dtype=int)
    for fold_idx in range(5):
        test_indices = np.array(folds[fold_idx], dtype=int)
        train_indices = np.array([idx for j, fold in enumerate(folds) if j != fold_idx for idx in fold], dtype=int)
        tree = RuleTree(feature_names, labels, max_depth=12, min_samples=3)
        tree.fit(x[train_indices], y[train_indices])
        for idx in test_indices:
            override = manual_override(rows[idx]["features"])
            if override is not None and override[0] in label_to_int:
                predictions[idx] = label_to_int[override[0]]
            else:
                predictions[idx] = tree.predict_one(x[idx])[0]

    overall = float(np.mean(predictions == y))
    per_label: dict[str, Any] = {}
    for label in labels:
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        correct = sum(int(predictions[i] == y[i]) for i in indices)
        confused = Counter(labels[predictions[i]] for i in indices if predictions[i] != y[i])
        per_label[label] = {
            "n": len(indices),
            "accuracy": correct / len(indices) if indices else 0.0,
            "top_confusions": confused.most_common(5),
        }

    return {"overall_accuracy": overall, "per_label": per_label}


def evaluate_tree_cv(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    labels: list[str],
    grouped: bool = False,
) -> dict[str, Any]:
    label_to_int = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_int[row["label"]] for row in rows], dtype=int)
    x = make_matrix(rows, feature_names)
    folds = grouped_measurement_folds(rows, k=5) if grouped else stratified_folds([row["label"] for row in rows], k=5)

    predictions = np.full(len(rows), -1, dtype=int)
    for fold_idx in range(5):
        test_indices = np.array(folds[fold_idx], dtype=int)
        if len(test_indices) == 0:
            continue
        train_indices = np.array([idx for j, fold in enumerate(folds) if j != fold_idx for idx in fold], dtype=int)
        tree = RuleTree(feature_names, labels, max_depth=12, min_samples=3)
        tree.fit(x[train_indices], y[train_indices])
        for idx in test_indices:
            override = manual_override(rows[idx]["features"])
            if override is not None and override[0] in label_to_int:
                predictions[idx] = label_to_int[override[0]]
            else:
                predictions[idx] = tree.predict_one(x[idx])[0]

    summary = summarize_predictions(rows, labels, predictions)
    summary["grouped"] = grouped
    summary["strategy"] = "manual_step_override_plus_induced_tree"
    return summary


def summarize_predictions(rows: list[dict[str, Any]], labels: list[str], predictions: np.ndarray) -> dict[str, Any]:
    label_to_int = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_int[row["label"]] for row in rows], dtype=int)
    overall = float(np.mean(predictions == y))
    per_label: dict[str, Any] = {}
    for label in labels:
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        correct = sum(int(predictions[i] == y[i]) for i in indices)
        confused = Counter(labels[predictions[i]] for i in indices if predictions[i] != y[i])
        per_label[label] = {
            "n": len(indices),
            "accuracy": correct / len(indices) if indices else 0.0,
            "top_confusions": confused.most_common(5),
        }
    return {"overall_accuracy": overall, "per_label": per_label}


def evaluate_nearest_neighbor_cv(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    labels: list[str],
    grouped: bool,
    feature_pool: str = "all",
    top_n: int = 180,
) -> dict[str, Any]:
    label_to_int = {label: i for i, label in enumerate(labels)}
    folds = grouped_measurement_folds(rows, k=5) if grouped else stratified_folds([row["label"] for row in rows], k=5)
    predictions = np.full(len(rows), -1, dtype=int)

    for fold_idx in range(5):
        test_indices = np.array(folds[fold_idx], dtype=int)
        train_indices = np.array([idx for j, fold in enumerate(folds) if j != fold_idx for idx in fold], dtype=int)
        train_rows = [rows[idx] for idx in train_indices]
        model = fit_nearest_neighbor_model(train_rows, feature_names, labels, feature_pool=feature_pool, top_n=top_n)

        for idx in test_indices:
            override = manual_override(rows[idx]["features"])
            if override is not None:
                predictions[idx] = label_to_int[override[0]]
            else:
                predicted, _, _ = predict_nearest_neighbor(model, rows[idx]["features"])
                predictions[idx] = label_to_int[predicted]

    summary = summarize_predictions(rows, labels, predictions)
    summary["grouped"] = grouped
    summary["feature_pool"] = feature_pool
    summary["top_n"] = top_n
    return summary


def evaluate_source_hybrid_nearest_neighbor_cv(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    labels: list[str],
    grouped: bool,
    watch_top_n: int = 100,
    all_top_n: int = 320,
) -> dict[str, Any]:
    label_to_int = {label: i for i, label in enumerate(labels)}
    folds = grouped_measurement_folds(rows, k=5) if grouped else stratified_folds([row["label"] for row in rows], k=5)
    predictions = np.full(len(rows), -1, dtype=int)

    for fold_idx in range(5):
        test_indices = np.array(folds[fold_idx], dtype=int)
        train_indices = np.array([idx for j, fold in enumerate(folds) if j != fold_idx for idx in fold], dtype=int)
        train_rows = [rows[idx] for idx in train_indices]
        models = fit_source_hybrid_nearest_neighbor_models(
            train_rows,
            feature_names,
            labels,
            watch_top_n=watch_top_n,
            all_top_n=all_top_n,
        )

        for idx in test_indices:
            override = manual_override(rows[idx]["features"])
            if override is not None and override[0] in label_to_int:
                predictions[idx] = label_to_int[override[0]]
            else:
                predicted, _, _ = predict_source_hybrid_nearest_neighbor(models, rows[idx]["features"])
                predictions[idx] = label_to_int[predicted]

    summary = summarize_predictions(rows, labels, predictions)
    summary["grouped"] = grouped
    summary["strategy"] = "source_signature_hybrid_1nn"
    return summary


def source_signature(meta: dict[str, Any]) -> str:
    counts = meta.get("source_counts", {})
    present = sorted(source for source, count in counts.items() if count >= 5)
    return "+".join(present) if present else "too_short"


def feature_summary(rows: list[dict[str, Any]], feature_names: list[str], selected: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by_label: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row["features"])

    for label in sorted(by_label):
        out[label] = {}
        for feature in selected:
            values = [features.get(feature, 0.0) for features in by_label[label]]
            out[label][feature] = float(statistics.median(values)) if values else 0.0
    return out


def manual_override(features: dict[str, float]) -> tuple[str, float, str] | None:
    """High-precision rules that should run before the induced tree."""

    full_sensor_step = (
        features.get("i_am_peak_count_gt2", 0.0) >= 10
        and features.get("a_am_peak_count_gt2", 0.0) >= 10
        and features.get("w_am_peak_count_gt5", 0.0) <= 1
    )
    watch_only_step = (
        features.get("w_am_peak_count_gt2", 0.0) >= 9
        and features.get("w_am_peak_count_gt3", 0.0) <= 1
        and features.get("w_ax_p95", 0.0) <= 0.2
        and features.get("w_ax_p05", 0.0) < -1.5
        and features.get("w_dur", 0.0) > 10
    )
    if full_sensor_step or watch_only_step:
        return "Step", 1.0, "step_peak_pattern"

    return None


def write_report(
    out_dir: Path,
    data_root: Path,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    labels: list[str],
    train_accuracy: float,
    cv: dict[str, Any],
    tree_text: str,
    selected_summary: dict[str, dict[str, float]],
    model_metrics: dict[str, Any] | None = None,
    measurement_summary: dict[str, Any] | None = None,
) -> None:
    counts = Counter(row["label"] for row in valid_rows)
    source_counts = Counter(source_signature(row["meta"]) for row in valid_rows)

    lines: list[str] = []
    lines.append("# Motion Rule Base Report")
    lines.append("")
    lines.append(f"- Data root: `{data_root}`")
    lines.append(f"- Valid CSV files used for rules: {len(valid_rows)}")
    lines.append(f"- Invalid/too-short CSV files: {len(invalid_rows)}")
    lines.append(f"- Classes: {len(labels)}")
    if model_metrics and model_metrics.get("excluded_subjects"):
        lines.append(f"- Excluded subjects: {', '.join(model_metrics['excluded_subjects'])}")
    classifier = model_metrics.get("classifier", "hybrid") if model_metrics else "hybrid"
    if classifier == "tree":
        lines.append("- Deployed classifier: manual Step override + induced if/then decision tree")
    else:
        lines.append("- Deployed classifier: manual Step override + selected-feature 1-nearest-neighbor")
    lines.append(f"- Training fit accuracy: {train_accuracy:.3f}")
    lines.append(f"- File-level 5-fold stratified accuracy: {cv['overall_accuracy']:.3f}")
    if model_metrics:
        grouped = model_metrics.get("grouped_measurement_5fold_accuracy")
        tree_file = model_metrics.get("tree_file_5fold_accuracy")
        if grouped is not None:
            lines.append(f"- Grouped-measurement 5-fold accuracy: {grouped:.3f}")
        representative = model_metrics.get("representative_measurement_5fold_accuracy")
        if representative is not None:
            lines.append(f"- Deduplicated representative-measurement 5-fold accuracy: {representative:.3f}")
        if tree_file is not None and classifier != "tree":
            lines.append(f"- Previous tree-style file-level 5-fold accuracy: {tree_file:.3f}")
    if measurement_summary:
        lines.append(f"- Measurement groups after merging same label/timestamp exports: {measurement_summary['measurement_group_count']}")
        lines.append(f"- Representative measurement rows used for deduplicated evaluation: {measurement_summary['representative_count']}")
    lines.append("")
    lines.append("## Preprocessing Rule")
    lines.append("")
    lines.append("1. Read all rows grouped by `source`.")
    lines.append("2. Find the first synchronization jump in `iPhone`, then `AirPods`, then `Watch`.")
    lines.append("3. In the first `min(5.0 s, max(1.5 s, 50% of recording duration))`, take the first local acceleration-magnitude peak above `max(2.15 g, median + 2.5 * MAD)`.")
    lines.append("4. Start classification at `jump_peak_time + 0.55 s`. If no jump is found, classify from the beginning but set `jump_found = 0`.")
    lines.append("")
    lines.append("## Core Human-Readable Rules")
    lines.append("")
    lines.append("These are the compact rules discovered from the tree and the repetition protocol. The full executable if/then tree is in `rules.json`.")
    lines.append("")
    watch_top_n = model_metrics.get("watch_top_n", 100) if model_metrics else 100
    all_top_n = model_metrics.get("all_top_n", 320) if model_metrics else 320
    if classifier == "tree":
        lines.append("- Primary classifier: executable CART-style if/then tree over engineered Watch/iPhone/AirPods features. No nearest-neighbor or learned distance model is used in this output.")
    else:
        lines.append(f"- Primary classifier: source-aware 1-nearest-neighbor. Watch-only files use the top {watch_top_n} Watch F-score features; files with iPhone/AirPods use the top {all_top_n} all-sensor F-score features. Both branches use median/IQR robust scaling and mean absolute distance.")
    lines.append("- `w_am_peak_count_gt5 >= 4`: lead-side single punch group. Then `w_ay_mean <= 0.059` -> `Lead Jab`; else `w_az_mean <= -0.231` -> `Lead Hook`; else -> `Lead Uppercut`.")
    lines.append("- `w_am_peak_count_gt8 >= 2` and `w_am_peak_count_gt5 < 4`: repeated high lead-jab impacts -> `Lead Jab-Lead Jab`.")
    lines.append("- Step override: if `i_am_peak_count_gt2 >= 10` and `a_am_peak_count_gt2 >= 10` with `w_am_peak_count_gt5 <= 1`, classify `Step`. For Watch-only step files: `w_am_peak_count_gt2 >= 9`, `w_am_peak_count_gt3 <= 1`, `w_ax_p95 <= 0.2`, `w_ax_p05 < -1.5`, and `w_dur > 10`.")
    lines.append("- Very low Watch acceleration and gyro peak counts across the post-jump window: `Guard`, slips, rolls, and rear-side punches. These are separated by the later tree leaves using lateral-axis signs and body/head peaks.")
    lines.append("- Combination labels are interpreted as one primitive sequence per file after the jump. The three repeats are repeated measurements across files, not three combination cycles inside one file.")
    lines.append("")
    lines.append("## Primitive Sequence Grammar")
    lines.append("")
    grammar = {
        "Lead Jab": "LeadJab x5",
        "Rear Jab": "RearJab x5",
        "Lead Hook": "LeadHook x5",
        "Rear Hook": "RearHook x5",
        "Lead Uppercut": "LeadUppercut x5",
        "Rear Uppercut": "RearUppercut x5",
        "Left Slip": "LeftSlip x5",
        "Right Slip": "RightSlip x5",
        "Left Roll": "LeftRoll x5",
        "Right Roll": "RightRoll x5",
        "Guard": "low-motion guard hold",
        "Step": "Step x5",
        "Lead Jab-Lead Jab": "(LeadJab, LeadJab) x1 per file",
        "Lead Jab-Rear Jab": "(LeadJab, RearJab) x1 per file",
        "Lead Jab-Rear Hook": "(LeadJab, RearHook) x1 per file",
        "Lead Jab-Right Slip-Rear Jab": "(LeadJab, RightSlip, RearJab) x1 per file",
        "Lead Jab-Right Slip-Rear Hook": "(LeadJab, RightSlip, RearHook) x1 per file",
        "Lead Hook-Rear Hook": "(LeadHook, RearHook) x1 per file",
        "Rear Hook-Lead Hook": "(RearHook, LeadHook) x1 per file",
        "Rear Uppercut-Lead Hook": "(RearUppercut, LeadHook) x1 per file",
        "Left Roll-Lead Hook-Rear Hook": "(LeftRoll, LeadHook, RearHook) x1 per file",
        "Right Roll-Rear Hook-Lead Hook": "(RightRoll, RearHook, LeadHook) x1 per file",
    }
    for label in labels:
        if label in grammar:
            lines.append(f"- `{label}`: {grammar[label]}")
    lines.append("")
    lines.append("## Valid Files By Class")
    lines.append("")
    for label in labels:
        lines.append(f"- `{label}`: {counts[label]}")
    lines.append("")
    lines.append("## Source Signatures")
    lines.append("")
    for signature, count in source_counts.most_common():
        lines.append(f"- `{signature}`: {count}")
    lines.append("")
    if measurement_summary:
        lines.append("## Measurement-Level Deduplication")
        lines.append("")
        lines.append("Watch-only files with the same label/timestamp as full-sensor files are duplicate exports of the same measurement, not independent measurements.")
        lines.append(f"- File rows used by the model: {measurement_summary['file_count']}")
        lines.append(f"- Measurement groups: {measurement_summary['measurement_group_count']}")
        lines.append(f"- Representative rows after preferring full-sensor exports: {measurement_summary['representative_count']}")
        lines.append("- Group size counts:")
        for size, count in sorted(measurement_summary["group_size_counts"].items(), key=lambda item: int(item[0])):
            lines.append(f"  - `{size}` files per measurement: {count}")
        lines.append("")
    if invalid_rows:
        lines.append("## Excluded Files")
        lines.append("")
        for row in invalid_rows:
            lines.append(f"- `{row['file']}`: {row['meta'].get('status', 'invalid')}, rows={row['meta'].get('total_rows', 0)}")
        lines.append("")
    lines.append("## Median Feature Signatures")
    lines.append("")
    selected = list(next(iter(selected_summary.values())).keys()) if selected_summary else []
    lines.append("| class | " + " | ".join(selected) + " |")
    lines.append("|---" + "|---" * len(selected) + "|")
    for label in labels:
        values = selected_summary.get(label, {})
        lines.append("| " + label + " | " + " | ".join(f"{values.get(feature, 0.0):.2f}" for feature in selected) + " |")
    lines.append("")
    lines.append("## Induced If/Then Tree Preview")
    lines.append("")
    lines.append("```text")
    lines.append(tree_text)
    lines.append("```")
    lines.append("")
    lines.append("## Validation Notes")
    lines.append("")
    lines.append("The file-level 5-fold score can be optimistic because Watch-only and full-sensor exports from the same measurement may land in different folds. The grouped-measurement score keeps same-label/same-timestamp exports together and is the stricter estimate. Rear-side punches, slips, and rolls remain the hardest classes because their Watch signatures are weak and similar.")

    (out_dir / "RULE_BASE_REPORT.md").write_text("\n".join(lines) + "\n")


def write_predictions(
    out_dir: Path,
    rows: list[dict[str, Any]],
    labels: list[str],
    tree: RuleTree,
    x: np.ndarray,
    nearest_neighbor_model: dict[str, Any] | None = None,
) -> None:
    with (out_dir / "classification_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        label_to_int = {label: i for i, label in enumerate(labels)}
        writer.writerow(["file", "actual", "predicted", "confidence", "correct", "rule_source", "jump_cut_s", "source_signature"])
        for row, values in zip(rows, x):
            override = manual_override(row["features"])
            if override is not None:
                predicted, confidence, rule_source = override
                if predicted not in label_to_int:
                    pred_idx, confidence, _ = tree.predict_one(values)
                    predicted = labels[pred_idx]
                    rule_source = "induced_tree"
                else:
                    pred_idx = label_to_int[predicted]
            elif nearest_neighbor_model is not None:
                if nearest_neighbor_model.get("type") == "source_signature_hybrid_1nn":
                    predicted, confidence, _ = predict_source_hybrid_nearest_neighbor(nearest_neighbor_model, row["features"])
                else:
                    predicted, confidence, _ = predict_nearest_neighbor(nearest_neighbor_model, row["features"])
                pred_idx = label_to_int[predicted]
                rule_source = nearest_neighbor_model.get("type", "selected_feature_1nn")
            else:
                pred_idx, confidence, _ = tree.predict_one(values)
                predicted = labels[pred_idx]
                rule_source = "induced_tree"
            writer.writerow(
                [
                    row["file"],
                    row["label"],
                    predicted,
                    f"{confidence:.6f}",
                    int(predicted == row["label"]),
                    rule_source,
                    f"{row['meta'].get('jump_cut_s', 0.0):.6f}",
                    source_signature(row["meta"]),
                ]
            )


def build_rule_base(
    data_root: Path,
    out_dir: Path,
    excluded_subjects: set[str] | None = None,
    watch_top_n: int = 100,
    all_top_n: int = 320,
    classifier: str = "hybrid",
) -> None:
    if classifier not in {"hybrid", "tree"}:
        raise ValueError("classifier must be 'hybrid' or 'tree'")

    excluded_subjects = excluded_subjects or set()
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    all_features: set[str] = set()

    for path in sorted(data_root.glob("*.csv")):
        label = label_from_name(path)
        if label is None:
            continue
        subject = subject_from_name(path)
        if subject in excluded_subjects:
            continue

        features, meta = extract_features(path)
        row = {"file": path.name, "label": label, "features": features or {}, "meta": meta, "subject": subject}
        if features is None:
            invalid_rows.append(row)
            continue

        valid_rows.append(row)
        all_features.update(features.keys())

    labels = sorted({row["label"] for row in valid_rows})
    label_to_int = {label: i for i, label in enumerate(labels)}
    feature_names = sorted(all_features)
    x = make_matrix(valid_rows, feature_names)
    y = np.array([label_to_int[row["label"]] for row in valid_rows], dtype=int)

    tree = RuleTree(feature_names, labels, max_depth=15, min_samples=2)
    tree.fit(x, y)
    nearest_neighbor_model = None
    if classifier == "hybrid":
        nearest_neighbor_model = fit_source_hybrid_nearest_neighbor_models(
            valid_rows,
            feature_names,
            labels,
            watch_top_n=watch_top_n,
            all_top_n=all_top_n,
        )

    predictions = []
    for row, values in zip(valid_rows, x):
        override = manual_override(row["features"])
        if override is not None and override[0] in label_to_int:
            predictions.append(label_to_int[override[0]])
        elif nearest_neighbor_model is not None:
            predicted, _, _ = predict_source_hybrid_nearest_neighbor(nearest_neighbor_model, row["features"])
            predictions.append(label_to_int[predicted])
        else:
            predictions.append(tree.predict_one(values)[0])
    predictions = np.array(predictions, dtype=int)
    train_accuracy = float(np.mean(predictions == y))
    tree_cv = evaluate_cv(valid_rows, feature_names, labels)
    representatives = representative_measurement_rows(valid_rows)
    if classifier == "hybrid":
        cv = evaluate_source_hybrid_nearest_neighbor_cv(
            valid_rows,
            feature_names,
            labels,
            grouped=False,
            watch_top_n=watch_top_n,
            all_top_n=all_top_n,
        )
        grouped_cv = evaluate_source_hybrid_nearest_neighbor_cv(
            valid_rows,
            feature_names,
            labels,
            grouped=True,
            watch_top_n=watch_top_n,
            all_top_n=all_top_n,
        )
        representative_cv = evaluate_source_hybrid_nearest_neighbor_cv(
            representatives,
            feature_names,
            labels,
            grouped=False,
            watch_top_n=watch_top_n,
            all_top_n=all_top_n,
        )
    else:
        cv = evaluate_tree_cv(valid_rows, feature_names, labels, grouped=False)
        grouped_cv = evaluate_tree_cv(valid_rows, feature_names, labels, grouped=True)
        representative_cv = evaluate_tree_cv(representatives, feature_names, labels, grouped=False)
    measurement_summary = measurement_stats(valid_rows)

    selected_features = [
        "w_am_peak_count_gt5",
        "w_am_peak_count_gt8",
        "w_gm_peak_count_gt8",
        "w_gm_peak_count_gt20",
        "w_ay_mean",
        "w_az_mean",
        "w_gy_pos_p95",
        "w_gz_neg_p95",
        "i_am_peak_count_gt2",
        "a_am_peak_count_gt2",
    ]
    selected_summary = feature_summary(valid_rows, feature_names, selected_features)

    rules = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "excluded_subjects": sorted(excluded_subjects),
        "preprocessing": {
            "jump_source_priority": list(JUMP_SOURCE_PRIORITY),
            "early_window": "min(5.0 s, max(1.5 s, 50% of recording duration))",
            "jump_peak_threshold": "max(2.15 g, median + 2.5 * MAD)",
            "classification_starts_after_jump_seconds": 0.55,
        },
        "labels": labels,
        "feature_names": feature_names,
        "tree": tree.to_dict(),
        "manual_rules": [
            {
                "prediction": "Step",
                "confidence": 1.0,
                "name": "step_peak_pattern",
                "logic": "(i_am_peak_count_gt2 >= 10 and a_am_peak_count_gt2 >= 10 and w_am_peak_count_gt5 <= 1) or (w_am_peak_count_gt2 >= 9 and w_am_peak_count_gt3 <= 1 and w_ax_p95 <= 0.2 and w_ax_p05 < -1.5 and w_dur > 10)",
            }
        ],
        "metrics": {
            "classifier": classifier,
            "valid_files": len(valid_rows),
            "invalid_files": len(invalid_rows),
            "excluded_subjects": sorted(excluded_subjects),
            "training_accuracy": train_accuracy,
            "stratified_5fold_accuracy": cv["overall_accuracy"],
            "stratified_5fold_per_label": cv["per_label"],
            "grouped_measurement_5fold_accuracy": grouped_cv["overall_accuracy"],
            "grouped_measurement_5fold_per_label": grouped_cv["per_label"],
            "representative_measurement_5fold_accuracy": representative_cv["overall_accuracy"],
            "representative_measurement_5fold_per_label": representative_cv["per_label"],
            "tree_stratified_5fold_accuracy": tree_cv["overall_accuracy"],
            "tree_stratified_5fold_per_label": tree_cv["per_label"],
            "measurement_summary": measurement_summary,
        },
        "median_feature_signatures": selected_summary,
    }
    if nearest_neighbor_model is not None:
        rules["nearest_neighbor_model"] = nearest_neighbor_model
    (out_dir / "rules.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False) + "\n")

    write_predictions(out_dir, valid_rows, labels, tree, x, nearest_neighbor_model)
    write_report(
        out_dir,
        data_root,
        valid_rows,
        invalid_rows,
        labels,
        train_accuracy,
        cv,
        tree.text(max_depth=7),
        selected_summary,
        {
            "classifier": classifier,
            "grouped_measurement_5fold_accuracy": grouped_cv["overall_accuracy"],
            "representative_measurement_5fold_accuracy": representative_cv["overall_accuracy"],
            "tree_file_5fold_accuracy": tree_cv["overall_accuracy"],
            "watch_top_n": watch_top_n,
            "all_top_n": all_top_n,
            "excluded_subjects": sorted(excluded_subjects),
        },
        measurement_summary,
    )


def load_rules(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def predict_from_tree(tree: dict[str, Any], features: dict[str, float], feature_names: list[str]) -> tuple[str, float, list[dict[str, Any]]]:
    override = manual_override(features)
    if override is not None:
        prediction, confidence, rule_name = override
        return prediction, confidence, [{"manual_rule": rule_name}]

    values = {name: features.get(name, 0.0) for name in feature_names}
    node = tree
    trace: list[dict[str, Any]] = []
    while node["type"] == "split":
        feature = node["feature"]
        threshold = float(node["threshold"])
        value = float(values.get(feature, 0.0))
        go_left = value <= threshold
        trace.append({"feature": feature, "value": value, "op": "<=" if go_left else ">", "threshold": threshold})
        node = node["left"] if go_left else node["right"]
    return str(node["prediction"]), float(node["confidence"]), trace


def predict_from_rules(rules: dict[str, Any], features: dict[str, float]) -> tuple[str, float, list[dict[str, Any]]]:
    override = manual_override(features)
    if override is not None:
        prediction, confidence, rule_name = override
        return prediction, confidence, [{"manual_rule": rule_name}]

    model = rules.get("nearest_neighbor_model")
    if model is not None:
        if model.get("type") == "source_signature_hybrid_1nn":
            return predict_source_hybrid_nearest_neighbor(model, features)
        return predict_nearest_neighbor(model, features)

    return predict_from_tree(rules["tree"], features, list(rules["feature_names"]))


def classify_paths(rules_path: Path, paths: list[Path]) -> None:
    rules = load_rules(rules_path)
    for path in paths:
        candidates = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for csv_path in candidates:
            features, meta = extract_features(csv_path)
            if features is None:
                print(json.dumps({"file": str(csv_path), "status": meta.get("status", "invalid"), "meta": meta}, ensure_ascii=False))
                continue
            prediction, confidence, trace = predict_from_rules(rules, features)
            print(
                json.dumps(
                    {
                        "file": str(csv_path),
                        "prediction": prediction,
                        "confidence": confidence,
                        "jump_cut_s": meta.get("jump_cut_s", 0.0),
                        "source_counts": meta.get("source_counts", {}),
                        "trace": trace,
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/apply rule-based classifier for Phantom CSV motion exports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="derive rules from labeled CSV files")
    build_parser.add_argument("data_root", type=Path)
    build_parser.add_argument("--out", type=Path, default=Path("motion_rule_base_output"))
    build_parser.add_argument("--exclude-subject", action="append", default=[], help="subject id to exclude, e.g. P01")
    build_parser.add_argument("--classifier", choices=["hybrid", "tree"], default="hybrid", help="classifier backend to store in rules.json")
    build_parser.add_argument("--watch-top-n", type=int, default=100, help="top Watch features for Watch-only branch")
    build_parser.add_argument("--all-top-n", type=int, default=320, help="top all-sensor features for fusion branch")

    classify_parser = subparsers.add_parser("classify", help="classify CSV files with an existing rules.json")
    classify_parser.add_argument("rules", type=Path)
    classify_parser.add_argument("paths", type=Path, nargs="+")

    args = parser.parse_args()
    if args.command == "build":
        build_rule_base(
            args.data_root.resolve(),
            args.out.resolve(),
            excluded_subjects=set(args.exclude_subject),
            watch_top_n=args.watch_top_n,
            all_top_n=args.all_top_n,
            classifier=args.classifier,
        )
    elif args.command == "classify":
        classify_paths(args.rules.resolve(), [path.resolve() for path in args.paths])


if __name__ == "__main__":
    main()
