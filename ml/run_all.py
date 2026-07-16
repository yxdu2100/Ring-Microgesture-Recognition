"""Run the frozen within-user comparison and continuous event evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes, prediction_report
from ringdata import (
    CLASS_NAMES,
    apply_manifest,
    confirm_consecutive_predictions,
    correct_activation_survival_fraction,
    load_sessions,
    match_events_to_gestures,
    recorded_hours,
    segment_sessions,
    stream_windows,
)
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows

SEED = 20260711
NULL_CLASS_ID = CLASS_NAMES.index("null")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _session_report(sessions, windows, path: Path) -> None:
    rows = []
    for session in sessions:
        session_windows = [window for window in windows if window.session_id == session.session_id]
        guided = [window for window in session_windows if window.source == "guided"]
        rows.append({
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "data_role": session.data_role,
            "usage": session.usage,
            "guided_protocol": session.guided_protocol,
            "samples": len(session.raw),
            "recorded_minutes": len(session.raw) / session.sample_rate_hz / 60.0,
            "sample_rate_hz": session.sample_rate_hz,
            "gap_count": session.gap_count,
            "missing_samples": session.missing_sample_count,
            "hardware_pct": session.hardware_flag_percentage,
            "go_markers": sum(marker.event_type == "go" for marker in session.markers),
            "valid_windows": len(session_windows),
            "reanchor_pct": (
                100.0 * sum(window.reanchored for window in guided) / len(guided)
                if guided else ""
            ),
        })
    _write_csv(rows, path)


def _window_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    macro, macro_all, precision, recall, f1, support = macro_f1_present_classes(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    gesture_present = (support > 0) & (np.arange(len(CLASS_NAMES)) != NULL_CLASS_ID)
    gesture_macro = float(np.mean(f1[gesture_present])) if np.any(gesture_present) else float("nan")
    row = {
        "macro_f1_present": macro,
        "macro_f1_all_classes": macro_all,
        "gesture_macro_f1": gesture_macro,
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
    }
    for index, name in enumerate(CLASS_NAMES):
        row[f"{name}_precision"] = float(precision[index])
        row[f"{name}_recall"] = float(recall[index])
        row[f"{name}_f1"] = float(f1[index])
        row[f"{name}_support"] = int(support[index])
    return row


def _prediction_rows(
    windows,
    predictions,
    method: str,
    fold_id: str,
    stream_kind: str,
    diagnostic: bool,
):
    return [
        {
            "method": method,
            "diagnostic": diagnostic,
            "fold_id": fold_id,
            "stream_kind": stream_kind,
            "session_id": window.session_id,
            "window_id": window.window_id,
            "start_sample_id": window.start_sample_id,
            "end_sample_id": window.end_sample_id,
            "predicted_class_id": int(prediction),
            "predicted_class": CLASS_NAMES[int(prediction)],
        }
        for window, prediction in zip(windows, predictions)
    ]


def _evaluate_events(
    method: str,
    fold_id: str,
    windows,
    predictions,
    references,
    stream_kind: str,
    hop_samples: int,
    diagnostic: bool,
    exposure_hours: float | None = None,
) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    match_rows = []
    survival_fraction = (
        correct_activation_survival_fraction(
            windows,
            predictions,
            references,
            minimum_run_windows=2,
            grace_samples=hop_samples,
        )
        if references else float("nan")
    )
    for consecutive in (1, 2):
        events = confirm_consecutive_predictions(
            windows,
            predictions,
            consecutive=consecutive,
            refractory_samples=120,
        )
        row = {
            "method": method,
            "diagnostic": diagnostic,
            "fold_id": fold_id,
            "stream_kind": stream_kind,
            "hop_samples": hop_samples,
            "consecutive_windows": consecutive,
            "activation_events": len(events),
            "correct_m1_activation_survival_to_m2": survival_fraction,
        }
        if references:
            metrics, matches = match_events_to_gestures(
                events,
                references,
                grace_samples=hop_samples,
            )
            row.update(metrics)
            for match in matches:
                match.update({
                    "method": method,
                    "diagnostic": diagnostic,
                    "fold_id": fold_id,
                    "stream_kind": stream_kind,
                    "consecutive_windows": consecutive,
                })
            match_rows.extend(matches)
        if exposure_hours is not None:
            row["exposure_hours"] = exposure_hours
            row["false_activations_per_hour"] = (
                len(events) / exposure_hours if exposure_hours > 0 else float("nan")
            )
        metric_rows.append(row)
    return metric_rows, match_rows


def _predict_tree(classifier, windows):
    from train_mlc.features import featurize

    x, y, _ = featurize(windows)
    return y, classifier.predict(x)


def _predict_cnn_model(model_path: Path, stats_path: Path, windows):
    import tensorflow as tf
    from train_cnn.train import _arrays

    stats = np.load(stats_path)
    x, y = _arrays(windows, stats["mean"], stats["std"])
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(x, verbose=0)
    return y, np.argmax(probabilities, axis=1)


def _st_tree_path(directory: Path | None, fold_id: str) -> Path | None:
    if directory is None:
        return None
    for name in (f"{fold_id}.txt", f"ST_decision_tree_{fold_id}.txt"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _aggregate(rows: list[dict], key: str = "gesture_macro_f1") -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = []
    for method, group in sorted(grouped.items()):
        values = np.array([row[key] for row in group], dtype=np.float64)
        output.append({
            "method": method,
            "diagnostic": bool(group[0].get("diagnostic", False)),
            "folds": len(group),
            f"{key}_mean": float(np.nanmean(values)),
            f"{key}_sample_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    return output


def _diagnostic_comparison(window_rows: list[dict], event_rows: list[dict]) -> list[dict]:
    """Summarize representation methods without changing the frozen primary."""
    output = []
    for method in sorted({row["method"] for row in window_rows}):
        method_windows = [row for row in window_rows if row["method"] == method]
        row = {
            "method": method,
            "diagnostic": bool(method_windows[0].get("diagnostic", False)),
            "folds": len(method_windows),
            "gesture_macro_f1_mean": float(np.mean([
                item["gesture_macro_f1"] for item in method_windows
            ])),
        }
        for stream_kind, metric_name, key in (
            ("guided_test", "guided_event_recall", "event_recall"),
            ("free_living_development", "development_fp_per_hour", "false_activations_per_hour"),
        ):
            for consecutive in (1, 2):
                values = [
                    item[key]
                    for item in event_rows
                    if item["method"] == method
                    and item["stream_kind"] == stream_kind
                    and item["consecutive_windows"] == consecutive
                    and key in item
                ]
                row[f"{metric_name}_m{consecutive}_mean"] = (
                    float(np.mean(values)) if values else float("nan")
                )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    parser.add_argument("--results-dir", default="ml/results/current")
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-hdc", action="store_true")
    parser.add_argument(
        "--skip-hdc-features",
        action="store_true",
        help="Skip the Python-only feature-representation HDC diagnostic",
    )
    parser.add_argument(
        "--hdc-experimental-phase-scaled",
        action="store_true",
        help="Run the rejected/timeboxed phase-augmentation + scaled-update ablation",
    )
    parser.add_argument(
        "--train-development-free-living-null",
        action="store_true",
        help=(
            "Post-hoc diagnostic: add development free-living windows to CNN/HDC "
            "null training; validation and final-test sessions remain unchanged"
        ),
    )
    parser.add_argument("--skip-mlc-proxy", action="store_true")
    parser.add_argument("--st-tree-dir", type=Path, default=None)
    args = parser.parse_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    sessions, manifest_warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in manifest_warnings:
        print(f"warning: {warning}")
    all_windows = segment_sessions(sessions, enforce_perform_window=False)
    dropped = [window for window in all_windows if window.perform_window_overrun_samples > 0]
    windows = [window for window in all_windows if window.perform_window_overrun_samples <= 0]
    if dropped:
        print(f"dropped {len(dropped)} guided windows exceeding the perform interval")
    _session_report(sessions, windows, results / "session_report.csv")
    splits = build_or_load_splits(
        windows,
        args.splits,
        seed=SEED,
        force_rebuild=args.rebuild_splits,
    )
    assert_no_cross_session_leakage(splits)
    (results / "split_summary.json").write_text(json.dumps(splits, indent=2) + "\n")

    sessions_by_id = {session.session_id: session for session in sessions}
    window_rows: list[dict] = []
    event_rows: list[dict] = []
    event_match_rows: list[dict] = []
    chronological_prediction_rows: list[dict] = []

    development_null_windows = [
        window
        for window in windows
        if window.data_role == "free_living_null" and window.usage == "development"
    ]
    if args.train_development_free_living_null and not development_null_windows:
        raise ValueError(
            "--train-development-free-living-null requires included development "
            "free-living sessions"
        )
    if args.train_development_free_living_null:
        development_session_ids = sorted({
            window.session_id for window in development_null_windows
        })
        final_session_ids = sorted({
            session.session_id
            for session in sessions
            if session.data_role == "free_living_null" and session.usage == "final_test"
        })
        (results / "training_diagnostic.json").write_text(json.dumps({
            "status": "post_hoc_diagnostic_not_primary",
            "applies_to": ["cnn_float", "hdc_D2048_reject"],
            "development_free_living_training_sessions": development_session_ids,
            "development_free_living_training_windows": len(development_null_windows),
            "final_test_sessions_excluded_from_training": final_session_ids,
            "validation_assignments": "unchanged_frozen_split",
            "hdc_rejection_thresholds": "fit_on_frozen_validation_only",
            "mlc_sensor_tree": "unchanged_reference_not_retrained",
        }, indent=2) + "\n")

    for fold in splits["within_user_session_folds"]:
        fold_id = fold["fold_id"]
        fold_dir = results / "folds" / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_fold = dict(fold)
        if args.train_development_free_living_null:
            added_ids = [window.window_id for window in development_null_windows]
            forbidden = set(fold["val"]) | set(fold["test"])
            if forbidden.intersection(added_ids):
                raise AssertionError(
                    f"{fold_id}: development free-living training windows overlap val/test"
                )
            training_fold["train"] = list(fold["train"]) + added_ids
        fold_wrapper = {"cross_session": training_fold}
        test_aligned = select_windows(windows, fold["test"])
        val_aligned = select_windows(windows, fold["val"])
        test_sessions = [sessions_by_id[session_id] for session_id in fold["test_gesture_sessions"]]
        stream_sets = [("guided_test", test_sessions, test_aligned)]
        for usage, stream_kind in (("development", "free_living_development"), ("final_test", "free_living_final_test")):
            null_sessions = [
                session for session in sessions
                if session.data_role == "free_living_null" and session.usage == usage
            ]
            if null_sessions:
                stream_sets.append((stream_kind, null_sessions, []))

        # MEMS Studio feature windows advance by one complete feature window;
        # the MCU classifiers use the deployed 50% overlap.  Event metrics must
        # preserve those actual cadences or M=2 latency/FP comparisons are false.
        predictors: list[tuple[str, object, int, bool]] = []
        if not args.skip_mlc_proxy:
            from train_mlc.tree import train_tree

            classifier, _, y_true, prediction = train_tree(windows, fold_wrapper)
            predictors.append((
                "mlc_proxy_tree",
                lambda target, c=classifier: _predict_tree(c, target),
                128,
                False,
            ))
            row = {
                "method": "mlc_proxy_tree",
                "diagnostic": False,
                "fold_id": fold_id,
                **_window_metrics(y_true, prediction),
            }
            window_rows.append(row)
            prediction_report(y_true, prediction, "mlc_proxy_tree", 120, fold_id, fold_dir, fail_on_collapse=False)

        st_path = _st_tree_path(args.st_tree_dir, fold_id)
        if st_path is not None:
            from train_mlc.st_tree import MLCTreeClassifier

            classifier = MLCTreeClassifier.from_file(st_path, precision="fp16")
            y_true, prediction = classifier.predict_windows(test_aligned)
            predictors.append((
                "mlc_sensor_tree",
                lambda target, c=classifier: c.predict_windows(target),
                128,
                False,
            ))
            window_rows.append({
                "method": "mlc_sensor_tree",
                "diagnostic": False,
                "fold_id": fold_id,
                "tree_path": str(st_path),
                **_window_metrics(y_true, prediction),
            })
            prediction_report(y_true, prediction, "mlc_sensor_tree", 120, fold_id, fold_dir, fail_on_collapse=False)

        if not args.skip_hdc:
            from train_hdc.encode import fit_level_bounds, make_codebooks
            from train_hdc.train import (
                SEED as HDC_SEED,
                fit_rejection_thresholds,
                predict_hdc_with_rejection,
                train_hdc,
            )

            train_windows = select_windows(windows, training_fold["train"])
            lo, hi = fit_level_bounds(train_windows)
            codebooks = make_codebooks(dim=2048, seed=HDC_SEED, level_min=lo, level_max=hi)
            memories = train_hdc(
                train_windows,
                codebooks,
                phase_augmentation=args.hdc_experimental_phase_scaled,
                confidence_scaled_updates=args.hdc_experimental_phase_scaled,
            )
            thresholds = fit_rejection_thresholds(val_aligned, memories, codebooks)
            if args.train_development_free_living_null:
                hdc_method = "hdc_D2048_reject_freeliving_aug_diagnostic"
            elif args.hdc_experimental_phase_scaled:
                hdc_method = "hdc_D2048_reject_phase_scaled_diagnostic"
            else:
                hdc_method = "hdc_D2048_reject"
            hdc_diagnostic = bool(
                args.train_development_free_living_null
                or args.hdc_experimental_phase_scaled
            )

            def hdc_predict(target, m=memories, c=codebooks, t=thresholds):
                y, prediction, _ = predict_hdc_with_rejection(target, m, c, t)
                return y, prediction

            predictors.append((hdc_method, hdc_predict, 64, hdc_diagnostic))
            y_true, prediction = hdc_predict(test_aligned)
            window_rows.append({
                "method": hdc_method,
                "diagnostic": hdc_diagnostic,
                "diagnostic_reason": (
                    "development_free_living_null_training"
                    if args.train_development_free_living_null
                    else (
                        "phase_augmentation_scaled_updates"
                        if args.hdc_experimental_phase_scaled
                        else ""
                    )
                ),
                "fold_id": fold_id,
                "max_distance_fraction": thresholds.max_distance_fraction,
                "min_margin_fraction": thresholds.min_margin_fraction,
                "validation_macro_f1": thresholds.validation_macro_f1,
                **_window_metrics(y_true, prediction),
            })
            prediction_report(y_true, prediction, hdc_method, 120, fold_id, fold_dir, fail_on_collapse=False)

            if not args.skip_hdc_features:
                from train_hdc.feature_encode import (
                    encode_feature_window,
                    fit_feature_codebooks,
                )

                feature_codebooks = fit_feature_codebooks(
                    train_windows,
                    dim=2048,
                    level_count=32,
                    seed=HDC_SEED,
                )

                def feature_encoder(raw, c=feature_codebooks):
                    return encode_feature_window(raw, c)

                feature_memories = train_hdc(
                    train_windows,
                    feature_codebooks,
                    encoder=feature_encoder,
                )
                feature_thresholds = fit_rejection_thresholds(
                    val_aligned,
                    feature_memories,
                    feature_codebooks,
                    encoder=feature_encoder,
                )
                feature_method = "hdc_D2048_features_reject"

                def feature_predict(
                    target,
                    m=feature_memories,
                    c=feature_codebooks,
                    t=feature_thresholds,
                    e=feature_encoder,
                ):
                    y, feature_prediction, _ = predict_hdc_with_rejection(
                        target,
                        m,
                        c,
                        t,
                        encoder=e,
                    )
                    return y, feature_prediction

                predictors.append((feature_method, feature_predict, 64, True))
                y_true, prediction = feature_predict(test_aligned)
                window_rows.append({
                    "method": feature_method,
                    "diagnostic": True,
                    "diagnostic_reason": "tree_feature_representation",
                    "fold_id": fold_id,
                    "feature_count": len(feature_codebooks.feature_names),
                    "level_count": len(feature_codebooks.levels),
                    "max_distance_fraction": feature_thresholds.max_distance_fraction,
                    "min_margin_fraction": feature_thresholds.min_margin_fraction,
                    "validation_macro_f1": feature_thresholds.validation_macro_f1,
                    **_window_metrics(y_true, prediction),
                })
                prediction_report(
                    y_true,
                    prediction,
                    feature_method,
                    120,
                    fold_id,
                    fold_dir,
                    fail_on_collapse=False,
                )

        if not args.skip_cnn:
            from train_cnn.train import train_one_rate

            cnn_dir = fold_dir / "cnn"
            cnn_result = train_one_rate(
                windows,
                fold_wrapper,
                120,
                cnn_dir,
                return_predictions=True,
                report_split_type=fold_id,
            )
            model_path = Path(cnn_result["model"])
            stats_path = cnn_dir / "cnn_120hz_standardizer.npz"
            cnn_method = (
                "cnn_float_freeliving_aug_diagnostic"
                if args.train_development_free_living_null
                else "cnn_float"
            )
            predictors.append((
                cnn_method,
                lambda target, model=model_path, stats=stats_path: _predict_cnn_model(model, stats, target),
                64,
                bool(args.train_development_free_living_null),
            ))
            window_rows.append({
                "method": cnn_method,
                "diagnostic": bool(args.train_development_free_living_null),
                "diagnostic_reason": (
                    "development_free_living_null_training"
                    if args.train_development_free_living_null
                    else ""
                ),
                "fold_id": fold_id,
                **_window_metrics(cnn_result["_y_true"], cnn_result["_y_pred"]),
            })

        for method, predictor, hop_samples, diagnostic in predictors:
            for stream_kind, exposure_sessions, references in stream_sets:
                if (
                    args.train_development_free_living_null
                    and diagnostic
                    and stream_kind == "free_living_development"
                ):
                    # This stream is part of diagnostic training and is not an
                    # evaluation set for the augmented CNN/HDC models.
                    continue
                stream = stream_windows(exposure_sessions, hop_samples=hop_samples)
                _, prediction = predictor(stream)
                chronological_prediction_rows.extend(
                    _prediction_rows(
                        stream,
                        prediction,
                        method,
                        fold_id,
                        stream_kind,
                        diagnostic,
                    )
                )
                exposure = None if references else recorded_hours(exposure_sessions)
                metrics, matches = _evaluate_events(
                    method,
                    fold_id,
                    stream,
                    prediction,
                    references,
                    stream_kind,
                    hop_samples,
                    diagnostic,
                    exposure,
                )
                event_rows.extend(metrics)
                event_match_rows.extend(matches)

    summary = _aggregate(window_rows)
    diagnostic_comparison = _diagnostic_comparison(window_rows, event_rows)
    _write_csv(window_rows, results / "fold_window_metrics.csv")
    _write_csv(summary, results / "summary.csv")
    _write_csv(diagnostic_comparison, results / "diagnostic_comparison.csv")
    _write_csv(event_rows, results / "event_metrics.csv")
    _write_csv(event_match_rows, results / "event_matches.csv")
    _write_csv(chronological_prediction_rows, results / "chronological_predictions.csv")
    with (results / "summary.md").open("w") as f:
        f.write("# Within-user cross-session summary\n\n")
        f.write("| method | diagnostic | folds | gesture macro-F1 |\n|---|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['method']} | {row['diagnostic']} | {row['folds']} | "
                f"{row['gesture_macro_f1_mean']:.4f} +/- "
                f"{row['gesture_macro_f1_sample_std']:.4f} |\n"
            )
        f.write("\n## Representation diagnostic\n\n")
        f.write(
            "| method | diagnostic | window F1 | guided M1 | guided M2 | "
            "development FP/hr M1 | development FP/hr M2 |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in diagnostic_comparison:
            f.write(
                f"| {row['method']} | {row['diagnostic']} | "
                f"{row['gesture_macro_f1_mean']:.4f} | "
                f"{row['guided_event_recall_m1_mean']:.4f} | "
                f"{row['guided_event_recall_m2_mean']:.4f} | "
                f"{row['development_fp_per_hour_m1_mean']:.2f} | "
                f"{row['development_fp_per_hour_m2_mean']:.2f} |\n"
            )
        f.write(
            "\nThe feature-HDC row is Python-only and cannot inherit the raw-HDC "
            "firmware resource measurements.\n"
        )
    print(f"wrote within-user results to {results}")


if __name__ == "__main__":
    main()
