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
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, load_sessions, segment_sessions
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


def export_windows(windows, splits: dict, out_dir: Path, acc_fs: int, gyr_fs: int,
                   already_physical: bool, window_length: int) -> int:
    train_ids = splits["cross_session"]["train"]
    selected = select_windows(windows, train_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list] = {}
    for w in selected:
        by_label.setdefault(w.label, []).append(w)

    total = 0
    for label in CLASS_NAMES:
        group = sorted(by_label.get(label, []), key=lambda w: w.window_id)
        rows = []
        for w in group:
            n = w.raw.shape[0]
            if n != window_length:
                # A wrong-length window would shift every subsequent window in
                # this class file, so drop it rather than corrupt alignment.
                print(f"  skip {w.window_id}: {n} samples != window_length {window_length}")
                continue
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
    parser.add_argument("--splits", default="ml/splits.json")
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
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not args.drop_invalid_windows)
    if args.drop_invalid_windows:
        dropped = [w for w in windows if w.perform_window_overrun_samples > 0]
        windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
        for w in dropped:
            print(f"dropping invalid overrun window: {w.window_id} "
                  f"overrun={w.perform_window_overrun_samples}")
    splits = build_or_load_splits(windows, args.splits)
    n = export_windows(windows, splits, Path(args.out_dir), args.acc_fs, args.gyr_fs,
                       args.already_physical, args.window_length)
    print(f"exported {n} train windows across {len(CLASS_NAMES)} class files "
          f"to {args.out_dir} (acc_fs=+/-{args.acc_fs}g, gyr_fs=+/-{args.gyr_fs}dps, "
          f"{'physical' if args.already_physical else 'LSB->physical'})")


if __name__ == "__main__":
    main()