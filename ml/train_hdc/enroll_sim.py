"""Leakage-free HDC enrollment simulation across participants and sessions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes
from ringdata import CLASS_NAMES, apply_manifest, load_sessions, segment_sessions
from train_hdc.encode import encode_window, fit_level_bounds, make_codebooks
from train_hdc.train import _signed, predict_hdc, train_hdc

SEED = 20260706
NULL_CLASS_ID = CLASS_NAMES.index("null")


def _select_enrollment_examples(windows, examples_per_class: int, seed: int):
    by_class = defaultdict(list)
    for window in windows:
        if window.class_id != NULL_CLASS_ID:
            by_class[window.class_id].append(window)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(NULL_CLASS_ID):
        candidates = sorted(by_class[class_id], key=lambda window: window.window_id)
        if len(candidates) < examples_per_class:
            raise ValueError(
                f"class {CLASS_NAMES[class_id]} has {len(candidates)} enrollment examples; "
                f"need {examples_per_class}"
            )
        indices = rng.permutation(len(candidates))[:examples_per_class]
        selected.extend(candidates[int(index)] for index in indices)
    return selected


def enrollment_curve(
    windows,
    out_csv: Path,
    dim: int = 2048,
    enrollment_weight: int = 1,
) -> list[dict]:
    gesture_windows = [window for window in windows if window.data_role == "gesture"]
    participants = sorted({window.participant_id for window in gesture_windows})
    if len(participants) < 2:
        raise ValueError("LOPO enrollment requires at least two participants")
    rows = []
    for held_participant in participants:
        held_sessions = sorted(
            {window.session_id for window in gesture_windows if window.participant_id == held_participant}
        )
        if len(held_sessions) < 2:
            raise ValueError(f"participant {held_participant} needs two gesture sessions")
        enrollment_session, test_session = held_sessions[0], held_sessions[1]
        base_windows = [
            window
            for window in windows
            if window.participant_id != held_participant
            and (
                window.data_role == "gesture"
                or (window.data_role == "structured_null" and window.usage == "train")
            )
        ]
        enrollment_pool = [
            window for window in gesture_windows if window.session_id == enrollment_session
        ]
        test_windows = [window for window in gesture_windows if window.session_id == test_session]
        lo, hi = fit_level_bounds(base_windows)
        codebooks = make_codebooks(dim=dim, seed=SEED, level_min=lo, level_max=hi)
        base_memories = train_hdc(base_windows, codebooks)

        for examples_per_class in (0, 1, 3, 5, 10):
            memories = base_memories.copy()
            enrolled_ids: set[str] = set()
            if examples_per_class:
                selected = _select_enrollment_examples(
                    enrollment_pool,
                    examples_per_class,
                    SEED + examples_per_class,
                )
                for window in selected:
                    enrolled_ids.add(window.window_id)
                    signed = _signed(encode_window(window.raw, codebooks))
                    memories[window.class_id] += enrollment_weight * signed
            if any(window.window_id in enrolled_ids for window in test_windows):
                raise AssertionError("enrollment examples leaked into test session")
            y_true, y_pred = predict_hdc(test_windows, memories, codebooks)
            macro_f1, macro_f1_all, *_ = macro_f1_present_classes(
                y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
            )
            rows.append({
                "held_participant": held_participant,
                "enrollment_session": enrollment_session,
                "test_session": test_session,
                "examples_per_gesture": examples_per_class,
                "enrollment_weight": enrollment_weight,
                "test_windows": len(test_windows),
                "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
                "macro_f1": macro_f1,
                "macro_f1_all_classes": macro_f1_all,
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--out", default="ml/results/hdc/enroll_curve.csv")
    parser.add_argument("--enrollment-weight", type=int, default=1)
    args = parser.parse_args()
    sessions, warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    rows = enrollment_curve(
        windows,
        Path(args.out),
        enrollment_weight=args.enrollment_weight,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
