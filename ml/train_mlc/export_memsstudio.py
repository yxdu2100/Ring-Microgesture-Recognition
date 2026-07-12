"""Export train-split windows as MEMS Studio-compatible datalog CSVs.

One CSV *per class* (not per window). Each class file is the concatenation of all
that class's train windows, each exactly `window_length` samples, with a single
header line on top:

    acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[dps],gyro_y[dps],gyro_z[dps]

Why concatenate: MEMS Studio's MLC uses NON-overlapping windows (stride =
window_length). A pattern file with exactly window_length samples emits 0 feature
vectors (boundary/off-by-one), which is why one-file-per-window produced an EMPTY
ARFF. Concatenating equal-length (128-sample) windows and letting the MLC slide a
128-sample non-overlapping window lands window k exactly on original window k:
perfect 1:1 alignment, no cross-gesture contamination, ~1 window lost at the file
tail (negligible).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, apply_manifest, load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows

# Exact header MEMS Studio's reader matches on. Do NOT rename columns.
MEMS_HEADER = "acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[dps],gyro_y[dps],gyro_z[dps]"

# LSM6DSV16X datasheet sensitivities.
ACC_SENS_MG_PER_LSB = {2: 0.061, 4: 0.122, 8: 0.244, 16: 0.488}          # mg / LSB
GYR_SENS_MDPS_PER_LSB = {125: 4.375, 250: 8.75, 500: 17.5,
                         1000: 35.0, 2000: 70.0, 4000: 140.0}            # mdps / LSB


def raw_to_physical(raw: np.ndarray, acc_fs: int, gyr_fs: int) -> np.ndarray:
    """(N, 6) int16-LSB window [ax,ay,az,gx,gy,gz] -> [mg, mg, mg, dps, dps, dps]."""
    acc_scale = ACC_SENS_MG_PER_LSB[acc_fs]             # LSB -> mg
    gyr_scale = GYR_SENS_MDPS_PER_LSB[gyr_fs] / 1000.0  # LSB -> dps
    out = raw.astype(np.float64).copy()
    out[:, 0:3] *= acc_scale
    out[:, 3:6] *= gyr_scale
    return out


def export_windows(windows, split: dict, out_dir: Path, acc_fs: int, gyr_fs: int,
                   already_physical: bool, window_length: int,
                   balance_classes: bool = True) -> int:
    train_ids = split["train"]
    selected = select_windows(windows, train_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list] = {}
    for w in selected:
        by_label.setdefault(w.label, []).append(w)

    valid_by_label: dict[str, list] = {}
    for label in CLASS_NAMES:
        group = sorted(
            by_label.get(label, []),
            key=lambda w: (w.session_id, w.start_sample_id, w.window_id),
        )
        valid_by_label[label] = []
        for w in group:
            if w.raw.shape[0] != window_length:
                print(f"  skip {w.window_id}: {w.raw.shape[0]} samples != window_length {window_length}")
                continue
            valid_by_label[label].append(w)

    nonempty_counts = [len(group) for group in valid_by_label.values() if group]
    target = min(nonempty_counts) if balance_classes and nonempty_counts else None
    total = 0
    for label in CLASS_NAMES:
        group = valid_by_label[label]
        if target is not None and len(group) > target:
            # Even spacing through session/start-sorted windows prevents the
            # null cap from silently selecting only the first null recording.
            indices = np.linspace(0, len(group) - 1, target, dtype=np.int64)
            group = [group[int(index)] for index in indices]
        rows = []
        for w in group:
            rows.append(w.raw if already_physical
                        else raw_to_physical(w.raw, acc_fs, gyr_fs))
        if not rows:
            print(f"WARNING: class '{label}' has 0 valid windows")
            continue
        stacked = np.vstack(rows)
        path = out_dir / f"{label}.csv"
        np.savetxt(path, stacked, fmt="%.4f", delimiter=",",
                   header=MEMS_HEADER, comments="")
        print(f"  {label}: {len(rows)} windows -> {stacked.shape[0]} rows -> {path.name}")
        total += len(rows)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    parser.add_argument("--out-dir", default="ml/results/memsstudio_export")
    parser.add_argument("--drop-invalid-windows", action="store_true")
    parser.add_argument("--window-length", type=int, default=128,
                        help="Samples per window; must match MEMS Studio 'Window length'")
    parser.add_argument("--acc-fs", type=int, default=8, choices=[2, 4, 8, 16],
                        help="Accelerometer full-scale in g (must match MEMS Studio)")
    parser.add_argument("--gyr-fs", type=int, default=2000,
                        choices=[125, 250, 500, 1000, 2000, 4000],
                        help="Gyroscope full-scale in dps (must match MEMS Studio)")
    parser.add_argument("--already-physical", action="store_true",
                        help="Set if window.raw is already in mg/dps (skip LSB scaling)")
    parser.add_argument("--all-folds", action="store_true",
                        help="Export one MEMS Studio training bundle per within-user fold")
    parser.add_argument("--keep-class-imbalance", action="store_true",
                        help="Do not cap every MEMS Studio class to the smallest class")
    parser.add_argument("--rebuild-splits", action="store_true")
    args = parser.parse_args()

    sessions, manifest_warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in manifest_warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    dropped = [w for w in windows if w.perform_window_overrun_samples > 0]
    windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
    for w in dropped:
        print(f"dropping invalid overrun window: {w.window_id} overrun={w.perform_window_overrun_samples}")
    splits = build_or_load_splits(
        windows,
        args.splits,
        force_rebuild=args.rebuild_splits,
    )
    selected_splits = (
        splits["within_user_session_folds"]
        if args.all_folds else
        [{"fold_id": "cross_session", **splits["cross_session"]}]
    )
    total = 0
    for split in selected_splits:
        fold_out = Path(args.out_dir) / split["fold_id"] if args.all_folds else Path(args.out_dir)
        n = export_windows(
            windows,
            split,
            fold_out,
            args.acc_fs,
            args.gyr_fs,
            args.already_physical,
            args.window_length,
            balance_classes=not args.keep_class_imbalance,
        )
        (fold_out / "split_sessions.json").write_text(json.dumps({
            "fold_id": split["fold_id"],
            "train_sessions": split["train_sessions"],
            "val_sessions": split["val_sessions"],
            "test_sessions": split["test_sessions"],
        }, indent=2) + "\n")
        total += n
    print(f"exported {total} train windows to {args.out_dir} "
          f"(acc_fs=+/-{args.acc_fs}g, gyr_fs=+/-{args.gyr_fs}dps, "
          f"{'physical' if args.already_physical else 'LSB->physical'})")


if __name__ == "__main__":
    main()
