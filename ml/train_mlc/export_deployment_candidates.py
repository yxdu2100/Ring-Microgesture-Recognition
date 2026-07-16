"""Export deterministic MEMS Studio datasets for a final deployment tree.

This export is intentionally separate from the frozen paper folds.  It uses all
six participants' valid guided windows, balances the four gesture classes to
the smallest class, and creates three candidate datasets that differ only in
the amount of null training data.  Long null recordings are sampled evenly by
session so that no participant or recording dominates the tree.

The paper's development and final-test free-living sessions are excluded.  They
remain available for selecting among the exported candidate trees without
training/evaluation overlap.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, apply_manifest, load_sessions, segment_sessions
from train_mlc.export_memsstudio import MEMS_HEADER, raw_to_physical


GESTURE_LABELS = [label for label in CLASS_NAMES if label != "null"]
DEFAULT_NULL_RATIOS = (1, 2, 4)


def _even_subset(items: list, count: int) -> list:
    if count > len(items):
        raise ValueError(f"requested {count} items from a group of {len(items)}")
    if count == len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def _session_balanced_null(windows: list, count: int) -> list:
    by_session: dict[str, list] = defaultdict(list)
    for window in windows:
        by_session[window.session_id].append(window)
    for group in by_session.values():
        group.sort(key=lambda window: (window.start_sample_id, window.window_id))

    session_ids = sorted(by_session)
    base, remainder = divmod(count, len(session_ids))
    selected = []
    for index, session_id in enumerate(session_ids):
        target = base + (1 if index < remainder else 0)
        selected.extend(_even_subset(by_session[session_id], target))
    return sorted(selected, key=lambda window: (window.session_id, window.start_sample_id))


def _write_class(path: Path, windows: list, acc_fs: int, gyr_fs: int) -> None:
    rows = np.vstack([raw_to_physical(window.raw, acc_fs, gyr_fs) for window in windows])
    np.savetxt(
        path,
        rows,
        fmt="%.4f",
        delimiter=",",
        header=MEMS_HEADER,
        comments="",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest_lopo.csv")
    parser.add_argument(
        "--out-dir",
        default="ml/results/memsstudio_deployment_candidates",
    )
    parser.add_argument("--acc-fs", type=int, default=8)
    parser.add_argument("--gyr-fs", type=int, default=2000)
    parser.add_argument(
        "--null-ratios",
        type=int,
        nargs="+",
        default=list(DEFAULT_NULL_RATIOS),
    )
    args = parser.parse_args()

    sessions, warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    all_windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [
        window
        for window in all_windows
        if window.perform_window_overrun_samples <= 0
    ]

    by_gesture: dict[str, list] = {}
    for label in GESTURE_LABELS:
        group = sorted(
            [
                window
                for window in windows
                if window.data_role == "gesture" and window.label == label
            ],
            key=lambda window: (
                window.participant_id,
                window.session_id,
                window.start_sample_id,
                window.window_id,
            ),
        )
        if not group:
            raise ValueError(f"no guided windows for {label}")
        by_gesture[label] = group

    gesture_target = min(len(group) for group in by_gesture.values())
    selected_gestures = {
        label: _even_subset(group, gesture_target)
        for label, group in by_gesture.items()
    }

    # Keep the paper's development and final-test streams outside training.
    eligible_null = sorted(
        [
            window
            for window in windows
            if window.label == "null"
            and (
                window.data_role == "structured_null"
                or (
                    window.data_role == "free_living_null"
                    and window.usage == "fold"
                )
            )
        ],
        key=lambda window: (
            window.participant_id,
            window.session_id,
            window.start_sample_id,
            window.window_id,
        ),
    )
    if not eligible_null:
        raise ValueError("no eligible deployment null windows")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    exports = []
    for ratio in args.null_ratios:
        if ratio < 1:
            raise ValueError("null ratios must be positive integers")
        null_target = ratio * gesture_target
        if null_target > len(eligible_null):
            raise ValueError(
                f"null ratio {ratio} requires {null_target} windows, "
                f"but only {len(eligible_null)} are eligible"
            )
        candidate_dir = out_root / f"null_{ratio}x"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for label, group in selected_gestures.items():
            _write_class(candidate_dir / f"{label}.csv", group, args.acc_fs, args.gyr_fs)
        selected_null = _session_balanced_null(eligible_null, null_target)
        _write_class(candidate_dir / "null.csv", selected_null, args.acc_fs, args.gyr_fs)

        metadata = {
            "purpose": "final_deployment_candidate_not_paper_evaluation",
            "null_ratio_relative_to_each_gesture_class": ratio,
            "gesture_windows_per_class": gesture_target,
            "null_windows": len(selected_null),
            "total_windows": 4 * gesture_target + len(selected_null),
            "gesture_participants": sorted(
                {window.participant_id for group in selected_gestures.values() for window in group}
            ),
            "null_sessions": sorted({window.session_id for window in selected_null}),
            "excluded_selection_sessions": sorted(
                {
                    window.session_id
                    for window in windows
                    if window.data_role == "free_living_null"
                    and window.usage in {"development", "final_test"}
                }
            ),
            "sample_rate_hz": 120,
            "window_samples": 128,
            "window_overlap": "none_in_mems_studio",
            "accelerometer_full_scale_g": args.acc_fs,
            "gyroscope_full_scale_dps": args.gyr_fs,
            "class_files": [f"{label}.csv" for label in CLASS_NAMES],
        }
        (candidate_dir / "export_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        exports.append(metadata)
        print(
            f"{candidate_dir}: {gesture_target} windows/gesture, "
            f"{len(selected_null)} null windows"
        )

    (out_root / "export_summary.json").write_text(json.dumps({
        "gesture_source_counts_before_balancing": {
            label: len(group) for label, group in by_gesture.items()
        },
        "balanced_gesture_windows_per_class": gesture_target,
        "eligible_null_windows": len(eligible_null),
        "candidates": exports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
