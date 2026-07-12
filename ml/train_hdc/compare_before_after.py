"""Build the protocol-preserving HDC before/after comparison table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ringdata import (
    apply_manifest,
    correct_activation_survival_fraction,
    load_sessions,
    segment_sessions,
    stream_windows,
)
from ringdata.splits import build_or_load_splits, select_windows


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _hdc_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("method", "").startswith("hdc_")]


def _event_row(rows: list[dict], fold_id: str, stream_kind: str, consecutive: int) -> dict:
    matches = [
        row for row in rows
        if row["fold_id"] == fold_id
        and row["stream_kind"] == stream_kind
        and int(row["consecutive_windows"]) == consecutive
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one event row for {fold_id}/{stream_kind}/M{consecutive}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _recompute_survival(
    result_dir: Path,
    fold: dict,
    sessions_by_id: dict,
    windows,
) -> float:
    prediction_rows = _hdc_rows(_read_csv(result_dir / "chronological_predictions.csv"))
    prediction_rows = [
        row for row in prediction_rows
        if row["fold_id"] == fold["fold_id"] and row["stream_kind"] == "guided_test"
    ]
    test_sessions = [sessions_by_id[session_id] for session_id in fold["test_gesture_sessions"]]
    stream = stream_windows(test_sessions, hop_samples=64)
    expected_ids = [window.window_id for window in stream]
    actual_ids = [row["window_id"] for row in prediction_rows]
    if expected_ids != actual_ids:
        raise ValueError(f"chronological prediction order mismatch in {fold['fold_id']}")
    predictions = [int(row["predicted_class_id"]) for row in prediction_rows]
    references = select_windows(windows, fold["test"])
    return correct_activation_survival_fraction(
        stream,
        predictions,
        references,
        minimum_run_windows=2,
        grace_samples=64,
    )


def summarize(
    stage: str,
    result_dir: Path,
    splits: dict,
    sessions_by_id: dict,
    windows,
) -> list[dict]:
    window_rows = _hdc_rows(_read_csv(result_dir / "fold_window_metrics.csv"))
    event_rows = _hdc_rows(_read_csv(result_dir / "event_metrics.csv"))
    by_fold = {row["fold_id"]: row for row in window_rows}
    output = []
    for fold in splits["within_user_session_folds"]:
        fold_id = fold["fold_id"]
        window = by_fold[fold_id]
        guided_m1 = _event_row(event_rows, fold_id, "guided_test", 1)
        guided_m2 = _event_row(event_rows, fold_id, "guided_test", 2)
        free_m1 = _event_row(event_rows, fold_id, "free_living_development", 1)
        free_m2 = _event_row(event_rows, fold_id, "free_living_development", 2)
        recorded = guided_m1.get("correct_m1_activation_survival_to_m2", "")
        survival = float(recorded) if recorded not in {"", "nan"} else _recompute_survival(
            result_dir, fold, sessions_by_id, windows
        )
        output.append({
            "stage": stage,
            "fold_id": fold_id,
            "gesture_macro_f1": float(window["gesture_macro_f1"]),
            "validation_macro_f1_at_rejection": float(window["validation_macro_f1"]),
            "guided_event_recall_m1": float(guided_m1["event_recall"]),
            "guided_event_recall_m2": float(guided_m2["event_recall"]),
            "development_fp_per_hour_m1": float(free_m1["false_activations_per_hour"]),
            "development_fp_per_hour_m2": float(free_m2["false_activations_per_hour"]),
            "correct_m1_activation_survival_to_m2": survival,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    args = parser.parse_args()

    sessions, warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits)
    sessions_by_id = {session.session_id: session for session in sessions}
    rows = summarize("before", args.before, splits, sessions_by_id, windows)
    rows += summarize("after", args.after, splits, sessions_by_id, windows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
