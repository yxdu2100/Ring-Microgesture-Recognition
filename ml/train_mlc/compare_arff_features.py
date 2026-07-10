"""Compare our MLC feature frontend against MEMS Studio's ARFF features."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_mlc.arff import read_arff
from train_mlc.export_memsstudio import raw_to_physical as export_raw_to_physical
from train_mlc.features import (
    MEMS_STUDIO_GYRO_INTERNAL_PER_LSB,
    mlc_feature_value,
    parse_mlc_feature_token,
)
from train_mlc.st_tree import ST_TO_PROJECT_CLASS

SEED = 20260706
LINEAR_GYRO_FEATURES = {"PEAK_TO_PEAK", "MINIMUM", "MAXIMUM", "MEAN"}
SQUARED_GYRO_FEATURES = {"VARIANCE", "ENERGY"}


def _feature_matrix(windows, feature_names: list[str], gyro_lsb_scale: float) -> np.ndarray:
    return np.array(
        [
            [
                float(
                    mlc_feature_value(
                        window.raw,
                        *parse_mlc_feature_token(token),
                        precision="fp64",
                        gyro_lsb_scale=gyro_lsb_scale,
                    )
                )
                for token in feature_names
            ]
            for window in windows
        ],
        dtype=np.float64,
    )


def _best_drop_one(group, arff_rows: np.ndarray, feature_names: list[str]) -> list:
    if len(group) != len(arff_rows) + 1:
        return group
    best_score = None
    best_group = group[:-1]
    scale = np.maximum(np.nanmax(np.abs(arff_rows), axis=0), 1e-9)
    for omit in range(len(group)):
        candidate = group[:omit] + group[omit + 1 :]
        computed = _feature_matrix(candidate, feature_names, MEMS_STUDIO_GYRO_INTERNAL_PER_LSB)
        score = float(np.median(np.abs((computed - arff_rows) / scale)))
        if best_score is None or score < best_score:
            best_score = score
            best_group = candidate
    return best_group


def _windows_in_export_file_order(windows, export_dir: Path) -> dict[str, list] | None:
    train_by_label = {}
    for st_label, project_label in ST_TO_PROJECT_CLASS.items():
        path = export_dir / f"{project_label}.csv"
        if not path.exists():
            return None
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] % 128 != 0:
            raise ValueError(f"{path}: row count {data.shape[0]} is not a multiple of 128")
        chunks = data.reshape(-1, 128, 6)
        candidates = [w for w in windows if w.label == project_label]
        ordered = []
        used: set[str] = set()
        for chunk in chunks:
            best = None
            for window in candidates:
                if window.window_id in used:
                    continue
                physical = export_raw_to_physical(window.raw, 8, 2000)
                error = float(np.max(np.abs(physical - chunk)))
                if best is None or error < best[0]:
                    best = (error, window)
            if best is None or best[0] > 1e-3:
                raise ValueError(f"{path}: could not match exported chunk to a known window")
            used.add(best[1].window_id)
            ordered.append(best[1])
        train_by_label[st_label] = ordered
    return train_by_label


def mems_ordered_windows(
    windows,
    splits: dict,
    labels: list[str],
    arff_rows: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    export_dir: str | Path | None = None,
):
    train_w = select_windows(windows, splits["cross_session"]["train"])
    by_st_label = None
    if export_dir is not None:
        export_path = Path(export_dir)
        if export_path.exists():
            by_st_label = _windows_in_export_file_order(train_w, export_path)
    if by_st_label is None:
        by_st_label = {}
        for st_label, project_label in ST_TO_PROJECT_CLASS.items():
            group = sorted(
                [w for w in train_w if w.label == project_label],
                key=lambda w: (w.session_id, w.start_sample_id, w.window_id),
            )
            by_st_label[st_label] = group
    if arff_rows is not None and feature_names is not None:
        counts = Counter(labels)
        offset = 0
        for label, count in counts.items():
            label_rows = arff_rows[offset : offset + count]
            by_st_label[label] = _best_drop_one(by_st_label[label], label_rows, feature_names)
            offset += count
    else:
        by_st_label = {label: group[:-1] for label, group in by_st_label.items()}
    cursors = Counter()
    ordered = []
    for label in labels:
        if label not in by_st_label:
            raise ValueError(f"ARFF label {label!r} is not mapped to a project class")
        idx = cursors[label]
        group = by_st_label[label]
        if idx >= len(group):
            raise ValueError(f"ARFF has more {label} rows than reconstructed MEMS windows")
        ordered.append(group[idx])
        cursors[label] += 1
    return ordered, dict(cursors)


def infer_gyro_scale(arff, windows) -> float:
    estimates = []
    for col, token in enumerate(arff.feature_names):
        feature, axis = parse_mlc_feature_token(token)
        if not axis.startswith("GYR_") or feature not in LINEAR_GYRO_FEATURES:
            continue
        for row, window in zip(arff.rows, windows):
            raw_feature = float(
                mlc_feature_value(
                    window.raw,
                    feature,
                    axis,
                    precision="fp64",
                    gyro_lsb_scale=1.0,
                )
            )
            if abs(raw_feature) > 1e-12 and abs(row[col]) > 1e-12:
                estimates.append(row[col] / raw_feature)
    if not estimates:
        raise ValueError("could not infer gyro scale from ARFF")
    return float(np.median(estimates))


def compare_features(arff, windows, gyro_lsb_scale: float, sample_count: int) -> tuple[list[dict], dict]:
    rows = []
    all_abs_errors = []
    all_rel_errors = []
    for row_idx, (arff_row, window, label) in enumerate(zip(arff.rows, windows, arff.labels)):
        if row_idx >= sample_count:
            break
        for col, token in enumerate(arff.feature_names):
            feature, axis = parse_mlc_feature_token(token)
            actual = float(
                mlc_feature_value(
                    window.raw,
                    feature,
                    axis,
                    precision="fp64",
                    gyro_lsb_scale=gyro_lsb_scale,
                )
            )
            expected = float(arff_row[col])
            abs_error = abs(actual - expected)
            rel_error = abs_error / max(abs(expected), 1e-12)
            rows.append(
                {
                    "row": row_idx,
                    "label": label,
                    "window_id": window.window_id,
                    "feature": token,
                    "arff": expected,
                    "computed": actual,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                }
            )
            all_abs_errors.append(abs_error)
            all_rel_errors.append(rel_error)
    summary = {
        "sample_rows": min(sample_count, len(windows)),
        "feature_count": len(arff.feature_names),
        "max_abs_error": float(np.max(all_abs_errors)) if all_abs_errors else 0.0,
        "median_abs_error": float(np.median(all_abs_errors)) if all_abs_errors else 0.0,
        "max_rel_error": float(np.max(all_rel_errors)) if all_rel_errors else 0.0,
        "median_rel_error": float(np.median(all_rel_errors)) if all_rel_errors else 0.0,
        "gyro_lsb_scale": gyro_lsb_scale,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arff", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--out-dir", default="ml/results/mlc_st")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--gyro-lsb-scale", type=float, default=None)
    parser.add_argument("--mems-export-dir", default="ml/results/memsstudio_export")
    parser.add_argument("--drop-invalid-windows", action="store_true")
    args = parser.parse_args()

    arff = read_arff(args.arff)
    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not args.drop_invalid_windows)
    if args.drop_invalid_windows:
        windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits, seed=SEED)
    ordered_windows, counts = mems_ordered_windows(
        windows,
        splits,
        arff.labels,
        arff_rows=arff.rows,
        feature_names=arff.feature_names,
        export_dir=args.mems_export_dir,
    )
    inferred = infer_gyro_scale(arff, ordered_windows)
    gyro_lsb_scale = inferred if args.gyro_lsb_scale is None else args.gyro_lsb_scale
    rows, summary = compare_features(arff, ordered_windows, gyro_lsb_scale, args.sample_count)
    summary["inferred_gyro_lsb_scale"] = inferred
    summary["known_default_gyro_lsb_scale"] = MEMS_STUDIO_GYRO_INTERNAL_PER_LSB
    summary["arff_rows"] = len(arff.labels)
    summary["reconstructed_counts"] = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "arff_feature_compare.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "arff_feature_compare_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"inferred gyro_lsb_scale={inferred:.12g}")
    print(
        f"sample max_abs_error={summary['max_abs_error']:.6g}, "
        f"median_abs_error={summary['median_abs_error']:.6g}"
    )


if __name__ == "__main__":
    main()
