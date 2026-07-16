"""Exploratory participant-balanced leave-one-participant-out evaluation.

This runner is intentionally separate from ``run_all.py`` so adding new
participants cannot alter the frozen P1 session folds or canonical results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_utils import prediction_report
from ringdata import (
    CLASS_NAMES,
    apply_manifest,
    load_sessions,
    recorded_hours,
    segment_sessions,
    stream_windows,
)
from run_all import (
    _aggregate,
    _evaluate_events,
    _predict_cnn_model,
    _predict_tree,
    _st_tree_path,
    _window_metrics,
    _write_csv,
)
from train_mlc.tree import train_tree


SEED = 20260713
NULL_CLASS_ID = CLASS_NAMES.index("null")
GESTURE_CLASS_IDS = list(range(NULL_CLASS_ID))


def _balanced_fold(windows, held_participant: str, seed: int) -> tuple[dict, dict]:
    """Build a participant-balanced outer fold with train-only validation.

    The held participant is absent from both train and validation. Because the
    additional participants each have one guided session, validation is
    stratified within the remaining participants rather than removing an
    entire training participant.
    """
    gesture = [window for window in windows if window.data_role == "gesture"]
    participants = sorted({window.participant_id for window in gesture})
    train_participants = [participant for participant in participants if participant != held_participant]
    if len(train_participants) < 2:
        raise ValueError("LOPO requires at least two non-held training participants")

    by_participant_class: dict[tuple[str, int], list] = defaultdict(list)
    for window in gesture:
        if window.participant_id in train_participants:
            by_participant_class[(window.participant_id, window.class_id)].append(window)
    target_per_participant_class = min(
        len(by_participant_class[(participant, class_id)])
        for participant in train_participants
        for class_id in GESTURE_CLASS_IDS
    )
    if target_per_participant_class < 5:
        raise ValueError(
            f"{held_participant}: too few gesture examples per participant/class: "
            f"{target_per_participant_class}"
        )

    rng = np.random.default_rng(seed)
    train_windows = []
    val_windows = []
    gesture_val_count = max(1, round(0.20 * target_per_participant_class))
    for participant in train_participants:
        for class_id in GESTURE_CLASS_IDS:
            group = sorted(
                by_participant_class[(participant, class_id)],
                key=lambda window: (window.session_id, window.start_sample_id, window.window_id),
            )
            chosen = [group[int(index)] for index in rng.permutation(len(group))[:target_per_participant_class]]
            val_windows.extend(chosen[:gesture_val_count])
            train_windows.extend(chosen[gesture_val_count:])

    # Use equal null exposure from each training participant. Sampling four
    # times the per-gesture target preserves null diversity without letting P1's
    # much longer recordings dominate the two newer participants.
    null_target_per_participant = 4 * target_per_participant_class
    null_val_count = max(1, round(0.20 * null_target_per_participant))
    for participant in train_participants:
        group = sorted(
            [
                window
                for window in windows
                if window.participant_id == participant
                and window.data_role in {"structured_null", "free_living_null"}
            ],
            key=lambda window: (window.session_id, window.start_sample_id, window.window_id),
        )
        if len(group) < null_target_per_participant:
            raise ValueError(
                f"{held_participant}: participant {participant} has only {len(group)} null windows; "
                f"need {null_target_per_participant}"
            )
        chosen = [group[int(index)] for index in rng.permutation(len(group))[:null_target_per_participant]]
        val_windows.extend(chosen[:null_val_count])
        train_windows.extend(chosen[null_val_count:])

    test_windows = [window for window in gesture if window.participant_id == held_participant]
    train_ids = {window.window_id for window in train_windows}
    val_ids = {window.window_id for window in val_windows}
    test_ids = {window.window_id for window in test_windows}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError(f"{held_participant}: LOPO window leakage")
    if any(window.participant_id == held_participant for window in train_windows + val_windows):
        raise AssertionError(f"{held_participant}: participant leakage into train/validation")

    split = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    metadata = {
        "held_participant": held_participant,
        "train_participants": train_participants,
        "target_gesture_examples_per_participant_class": target_per_participant_class,
        "train_gesture_examples_per_participant_class": target_per_participant_class - gesture_val_count,
        "validation_gesture_examples_per_participant_class": gesture_val_count,
        "train_null_examples_per_participant": null_target_per_participant - null_val_count,
        "validation_null_examples_per_participant": null_val_count,
        "train_windows": len(train_windows),
        "validation_windows": len(val_windows),
        "test_gesture_windows": len(test_windows),
    }
    return split, metadata


def _write_summary(window_rows: list[dict], event_rows: list[dict], path: Path) -> None:
    methods = list(dict.fromkeys(row["method"] for row in window_rows))
    participants = sorted({row["held_participant"] for row in window_rows})
    participant_labels = [f"P{int(participant)}" for participant in participants]
    lines = [
        "# Exploratory participant-balanced LOPO results",
        "",
        "## Held-participant window gesture macro-F1",
        "",
        "| Method | " + " | ".join(participant_labels) + " | Mean ± sample SD |",
        "|---|" + "---:|" * (len(participants) + 1),
    ]
    for method in methods:
        rows = [row for row in window_rows if row["method"] == method]
        by_participant = {row["held_participant"]: row["gesture_macro_f1"] for row in rows}
        values = np.asarray(list(by_participant.values()), dtype=np.float64)
        lines.append(
            f"| {method} | "
            + " | ".join(
                f"{by_participant.get(participant, float('nan')):.4f}"
                for participant in participants
            )
            + f" | {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f} |"
        )

    lines.extend([
        "",
        "## Continuous guided recall and held-participant free-living FP/hr",
        "",
        "| Method | Rule | Recall by participant | Equal-participant mean | FP/hr by participant |",
        "|---|---|---:|---:|---:|",
    ])
    for method in methods:
        for consecutive in (1, 2):
            guided = [
                row for row in event_rows
                if row["method"] == method
                and row["stream_kind"] == "guided_test"
                and row["consecutive_windows"] == consecutive
            ]
            recall = {
                row["held_participant"]: row["correct_events"] / row["gesture_events"]
                for row in guided
            }
            recall_values = np.asarray(list(recall.values()), dtype=np.float64)
            free = {
                row["held_participant"]: row["false_activations_per_hour"]
                for row in event_rows
                if row["method"] == method
                and row["stream_kind"] == "held_participant_free_living"
                and row["consecutive_windows"] == consecutive
            }
            lines.append(
                f"| {method} | {'one decision' if consecutive == 1 else 'two decisions'} | "
                + " / ".join(
                    f"{recall.get(participant, float('nan')):.4f}"
                    for participant in participants
                )
                + f" | {np.mean(recall_values):.4f} | "
                + " / ".join(
                    f"{free.get(participant, float('nan')):.2f}"
                    for participant in participants
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest_lopo.csv")
    parser.add_argument("--results-dir", default="ml/results/lopo_p1_p3")
    parser.add_argument(
        "--export-mlc-dir",
        type=Path,
        default=None,
        help="Export one class-balanced MEMS Studio training bundle per LOPO fold",
    )
    parser.add_argument(
        "--export-mlc-only",
        action="store_true",
        help="Create MEMS Studio bundles and exit without training CNN/HDC",
    )
    parser.add_argument(
        "--st-tree-dir",
        type=Path,
        default=None,
        help="Directory containing lopo_01.txt, lopo_02.txt, ... MEMS Studio trees",
    )
    parser.add_argument(
        "--skip-mlc-proxy",
        action="store_true",
        help="Omit the Python software-tree diagnostic",
    )
    args = parser.parse_args()

    if args.export_mlc_only and args.export_mlc_dir is None:
        parser.error("--export-mlc-only requires --export-mlc-dir")

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    sessions, warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    all_windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in all_windows if window.perform_window_overrun_samples <= 0]
    dropped = [window for window in all_windows if window.perform_window_overrun_samples > 0]
    print(f"dropped {len(dropped)} guided windows exceeding the perform interval")

    participants = sorted({
        window.participant_id for window in windows if window.data_role == "gesture"
    })
    if len(participants) < 3:
        raise ValueError(f"expected at least three participants; found {participants}")
    sessions_by_id = {session.session_id: session for session in sessions}

    window_rows: list[dict] = []
    event_rows: list[dict] = []
    event_match_rows: list[dict] = []
    fold_metadata: list[dict] = []

    for fold_index, held_participant in enumerate(participants):
        fold_id = f"lopo_{held_participant}"
        fold_dir = results / "folds" / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        split, metadata = _balanced_fold(windows, held_participant, SEED + fold_index)
        fold_metadata.append({"fold_id": fold_id, **metadata})

        if args.export_mlc_dir is not None:
            from train_mlc.export_memsstudio import export_windows

            export_dir = args.export_mlc_dir / fold_id
            exported = export_windows(
                windows,
                split,
                export_dir,
                acc_fs=8,
                gyr_fs=2000,
                already_physical=False,
                window_length=128,
                balance_classes=True,
            )
            (export_dir / "split_window_ids.json").write_text(json.dumps({
                "fold_id": fold_id,
                "held_participant": held_participant,
                "train_participants": metadata["train_participants"],
                "train_window_ids_before_mlc_class_balance": split["train"],
                "validation_window_ids": split["val"],
                "test_window_ids": split["test"],
                "exported_windows": exported,
                "accel_full_scale_g": 8,
                "gyro_full_scale_dps": 2000,
                "sample_rate_hz": 120,
                "window_samples": 128,
            }, indent=2) + "\n")
        if args.export_mlc_only:
            continue

        wrapper = {"lopo": split}
        test_windows = [window for window in windows if window.window_id in set(split["test"])]
        test_session_ids = sorted({window.session_id for window in test_windows})
        test_sessions = [sessions_by_id[session_id] for session_id in test_session_ids]
        held_null_sessions = [
            session for session in sessions
            if session.participant_id == held_participant
            and session.data_role == "free_living_null"
        ]
        if not held_null_sessions:
            raise ValueError(f"{fold_id}: held participant has no free-living null session")

        predictors: list[tuple[str, object, int, bool]] = []

        if not args.skip_mlc_proxy:
            tree, _, y_true, prediction = train_tree(windows, wrapper, split_type="lopo")
            tree_method = "mlc_proxy_tree_lopo_diagnostic"
            predictors.append((tree_method, lambda target, model=tree: _predict_tree(model, target), 128, True))
            window_rows.append({
                "method": tree_method,
                "diagnostic": True,
                "held_participant": held_participant,
                "fold_id": fold_id,
                **_window_metrics(y_true, prediction),
            })
            prediction_report(
                y_true, prediction, tree_method, 120, fold_id, fold_dir,
                fail_on_collapse=False,
            )

        if args.st_tree_dir is not None:
            from train_mlc.st_tree import MLCTreeClassifier

            st_path = _st_tree_path(args.st_tree_dir, fold_id)
            if st_path is None:
                raise FileNotFoundError(
                    f"missing MEMS Studio tree for {fold_id} in {args.st_tree_dir}; "
                    f"expected {fold_id}.txt or ST_decision_tree_{fold_id}.txt"
                )
            classifier = MLCTreeClassifier.from_file(st_path, precision="fp16")
            y_true, prediction = classifier.predict_windows(test_windows)
            tree_method = "mlc_sensor_tree_lopo"
            predictors.append((
                tree_method,
                lambda target, model=classifier: model.predict_windows(target),
                128,
                False,
            ))
            window_rows.append({
                "method": tree_method,
                "diagnostic": False,
                "held_participant": held_participant,
                "fold_id": fold_id,
                "tree_path": str(st_path),
                **_window_metrics(y_true, prediction),
            })
            prediction_report(
                y_true, prediction, tree_method, 120, fold_id, fold_dir,
                fail_on_collapse=False,
            )

        from train_hdc.encode import fit_level_bounds, make_codebooks
        from train_hdc.train import (
            SEED as HDC_SEED,
            fit_rejection_thresholds,
            predict_hdc_with_rejection,
            train_hdc,
        )
        train_windows = [window for window in windows if window.window_id in set(split["train"])]
        val_windows = [window for window in windows if window.window_id in set(split["val"])]
        lo, hi = fit_level_bounds(train_windows)
        codebooks = make_codebooks(dim=2048, seed=HDC_SEED, level_min=lo, level_max=hi)
        memories = train_hdc(train_windows, codebooks)
        thresholds = fit_rejection_thresholds(val_windows, memories, codebooks)

        def hdc_predict(target, m=memories, c=codebooks, t=thresholds):
            y, pred, _ = predict_hdc_with_rejection(target, m, c, t)
            return y, pred

        hdc_method = "hdc_D2048_reject_lopo"
        predictors.append((hdc_method, hdc_predict, 64, False))
        y_true, prediction = hdc_predict(test_windows)
        window_rows.append({
            "method": hdc_method,
            "diagnostic": False,
            "held_participant": held_participant,
            "fold_id": fold_id,
            "max_distance_fraction": thresholds.max_distance_fraction,
            "min_margin_fraction": thresholds.min_margin_fraction,
            "validation_macro_f1": thresholds.validation_macro_f1,
            **_window_metrics(y_true, prediction),
        })
        prediction_report(
            y_true, prediction, hdc_method, 120, fold_id, fold_dir,
            fail_on_collapse=False,
        )

        from train_cnn.train import train_one_rate
        cnn_dir = fold_dir / "cnn"
        cnn_result = train_one_rate(
            windows,
            wrapper,
            120,
            cnn_dir,
            split_key="lopo",
            report_split_type=fold_id,
            return_predictions=True,
        )
        cnn_method = "cnn_float_lopo"
        model_path = Path(cnn_result["model"])
        stats_path = cnn_dir / "cnn_120hz_standardizer.npz"
        predictors.append((
            cnn_method,
            lambda target, model=model_path, stats=stats_path: _predict_cnn_model(model, stats, target),
            64,
            False,
        ))
        window_rows.append({
            "method": cnn_method,
            "diagnostic": False,
            "held_participant": held_participant,
            "fold_id": fold_id,
            **_window_metrics(cnn_result["_y_true"], cnn_result["_y_pred"]),
        })

        for method, predictor, hop_samples, diagnostic in predictors:
            guided_stream = stream_windows(test_sessions, hop_samples=hop_samples)
            _, guided_prediction = predictor(guided_stream)
            metrics, matches = _evaluate_events(
                method,
                fold_id,
                guided_stream,
                guided_prediction,
                test_windows,
                "guided_test",
                hop_samples,
                diagnostic,
            )
            for row in metrics:
                row["held_participant"] = held_participant
            for row in matches:
                row["held_participant"] = held_participant
            event_rows.extend(metrics)
            event_match_rows.extend(matches)

            null_stream = stream_windows(held_null_sessions, hop_samples=hop_samples)
            _, null_prediction = predictor(null_stream)
            metrics, _ = _evaluate_events(
                method,
                fold_id,
                null_stream,
                null_prediction,
                [],
                "held_participant_free_living",
                hop_samples,
                diagnostic,
                exposure_hours=recorded_hours(held_null_sessions),
            )
            for row in metrics:
                row["held_participant"] = held_participant
            event_rows.extend(metrics)

    if args.export_mlc_only:
        (args.export_mlc_dir / "export_protocol.json").write_text(json.dumps({
            "status": "lopo_mems_studio_training_export",
            "manifest": args.manifest,
            "seed": SEED,
            "held_participant_excluded_from_train_and_validation": True,
            "folds": fold_metadata,
        }, indent=2) + "\n")
        print(f"wrote LOPO MEMS Studio bundles to {args.export_mlc_dir}")
        return

    _write_csv(window_rows, results / "fold_window_metrics.csv")
    _write_csv(_aggregate(window_rows), results / "summary.csv")
    _write_csv(event_rows, results / "event_metrics.csv")
    _write_csv(event_match_rows, results / "event_matches.csv")
    (results / "split_protocol.json").write_text(json.dumps({
        "status": "exploratory_lopo",
        "manifest": args.manifest,
        "seed": SEED,
        "validation": "20% participant-stratified windows from non-held participants",
        "participant_balancing": "equal gesture and null examples per training participant",
        "mlc": (
            "MEMS Studio trees loaded from " + str(args.st_tree_dir)
            if args.st_tree_dir is not None
            else "software proxy only; MEMS Studio trees not supplied"
        ),
        "folds": fold_metadata,
    }, indent=2) + "\n")
    _write_summary(window_rows, event_rows, results / "summary.md")
    print(f"wrote exploratory LOPO results to {results}")


if __name__ == "__main__":
    main()
