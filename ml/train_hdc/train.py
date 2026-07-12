"""Train and sweep HDC classifiers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes, prediction_report
from ringdata import CLASS_NAMES, apply_manifest, load_sessions, resample_windows, segment_sessions
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_hdc.encode import (
    HDC_LEVEL_COUNT,
    Codebooks,
    encode_window,
    fit_level_bounds,
    hamming,
    make_codebooks,
)

SEED = 20260706
DEFAULT_ENCODING_MODE = "ngram"
NULL_CLASS_ID = CLASS_NAMES.index("null")


@dataclass(frozen=True)
class RejectionThresholds:
    max_distance_fraction: float
    min_margin_fraction: float
    validation_macro_f1: float


def _signed(bits: np.ndarray) -> np.ndarray:
    return np.where(bits, 1, -1).astype(np.int16)


def _class_bits(memories: np.ndarray, codebooks: Codebooks) -> np.ndarray:
    return np.logical_or(memories > 0, np.logical_and(memories == 0, codebooks.bundle_tie.reshape(1, -1)))


def _shift_with_edge_padding(raw: np.ndarray, shift: int) -> np.ndarray:
    """Translate a window without wrapping, matching the CNN augmentation."""
    shifted = np.empty_like(raw)
    if shift > 0:
        shifted[shift:] = raw[:-shift]
        shifted[:shift] = raw[:1]
    elif shift < 0:
        shifted[:shift] = raw[-shift:]
        shifted[shift:] = raw[-1:]
    else:
        shifted[:] = raw
    return shifted


def _balanced_windows_baseline(windows, seed: int):
    """Preserve the frozen primary trainer's original sampling order."""
    by_class = {}
    for window in windows:
        by_class.setdefault(window.class_id, []).append(window)
    if len(by_class) < len(CLASS_NAMES):
        return windows
    count = min(len(group) for group in by_class.values())
    rng = np.random.default_rng(seed)
    selected = []
    for class_windows in by_class.values():
        ordered = sorted(class_windows, key=lambda window: window.window_id)
        indices = rng.permutation(len(ordered))[:count]
        selected.extend(ordered[int(index)] for index in indices)
    return sorted(selected, key=lambda window: window.window_id)


def _balanced_training_examples(windows, seed: int, shifts_per_gesture: int = 2):
    """Return class-balanced raw examples with gesture-only phase shifts.

    Structured-null recordings are already segmented with the deployment
    64-sample hop in ``ringdata.segment``. Re-windowing them through
    ``stream_windows`` would duplicate identical examples, so this function
    selects a larger train-only null subset to balance the augmented gestures.
    """
    by_class = {}
    for window in windows:
        by_class.setdefault(window.class_id, []).append(window)
    missing_gestures = [class_id for class_id in range(NULL_CLASS_ID) if class_id not in by_class]
    if missing_gestures:
        raise ValueError(
            "HDC training is missing gesture classes: "
            + ", ".join(CLASS_NAMES[class_id] for class_id in missing_gestures)
        )
    gesture_count = min(len(by_class[class_id]) for class_id in range(NULL_CLASS_ID))
    null_windows = by_class.get(NULL_CLASS_ID, [])
    if null_windows:
        invalid_null = [
            window.window_id
            for window in null_windows
            if window.data_role != "structured_null" or window.usage != "train"
        ]
        if invalid_null:
            raise ValueError(
                "HDC training null must be train structured-null only; found "
                + ", ".join(invalid_null[:3])
            )
    rng = np.random.default_rng(seed)
    shift_choices = np.asarray([-32, -16, 16, 32], dtype=np.int16)
    examples: list[tuple[np.ndarray, int]] = []
    for class_id in range(NULL_CLASS_ID):
        ordered = sorted(by_class[class_id], key=lambda window: window.window_id)
        indices = rng.permutation(len(ordered))[:gesture_count]
        for index in indices:
            window = ordered[int(index)]
            examples.append((window.raw, class_id))
            shifts = rng.choice(shift_choices, size=shifts_per_gesture, replace=False)
            for shift in shifts:
                examples.append((_shift_with_edge_padding(window.raw, int(shift)), class_id))

    if null_windows:
        target_null = gesture_count * (1 + shifts_per_gesture)
        ordered_null = sorted(null_windows, key=lambda window: window.window_id)
        indices = rng.permutation(len(ordered_null))[: min(target_null, len(ordered_null))]
        examples.extend((ordered_null[int(index)].raw, NULL_CLASS_ID) for index in indices)
    return examples


def train_hdc(
    train_w,
    codebooks: Codebooks,
    epochs: int = 5,
    mode: str = DEFAULT_ENCODING_MODE,
    phase_augmentation: bool = False,
    confidence_scaled_updates: bool = False,
) -> np.ndarray:
    memories = np.zeros((len(CLASS_NAMES), codebooks.dim), dtype=np.int32)
    encoded = []
    labels = []
    if phase_augmentation:
        examples = _balanced_training_examples(
            train_w,
            SEED + codebooks.dim,
            shifts_per_gesture=2,
        )
    else:
        examples = [
            (window.raw, window.class_id)
            for window in _balanced_windows_baseline(train_w, SEED + codebooks.dim)
        ]
    for raw, class_id in examples:
        q = encode_window(raw, codebooks, mode=mode)
        encoded.append(q)
        labels.append(class_id)
        memories[class_id] += _signed(q)

    rng = np.random.default_rng(SEED + codebooks.dim + 17)
    for _ in range(epochs):
        class_bits = _class_bits(memories, codebooks)
        for index in rng.permutation(len(encoded)):
            q = encoded[int(index)]
            y = labels[int(index)]
            distances = hamming(q, class_bits)
            pred = int(np.argmin(distances))
            if pred != y:
                if confidence_scaled_updates:
                    # Confidence-scaled perceptron update inspired by OnlineHD.
                    # HyperCam does not publish this exact weighting rule, so
                    # this is not a reproduction of its trainer.
                    error = float(distances[y] - distances[pred]) / codebooks.dim
                    weight = max(1, round(8.0 * error))
                else:
                    weight = 1
                s = _signed(q)
                memories[y] += weight * s
                memories[pred] -= weight * s
                class_bits = _class_bits(memories, codebooks)
    return memories


def validation_diagnostic_sweep(
    windows,
    splits: dict,
    out_csv: Path,
    mode: str = DEFAULT_ENCODING_MODE,
    split_key: str = "cross_session",
    phase_augmentation: bool = False,
    confidence_scaled_updates: bool = False,
) -> list[dict]:
    """Evaluate non-primary capacity variants on validation data only."""
    split = splits[split_key]
    train_w = select_windows(windows, split["train"])
    val_w = select_windows(windows, split["val"])
    if not train_w or not val_w:
        raise ValueError("HDC diagnostics require non-empty train and validation windows")
    lo, hi = fit_level_bounds(train_w)
    rows = []
    for dim, level_count, variant in (
        (8192, HDC_LEVEL_COUNT, "D8192_L32"),
        (2048, 64, "D2048_L64"),
    ):
        codebooks = make_codebooks(
            dim=dim,
            levels=level_count,
            seed=SEED,
            level_min=lo,
            level_max=hi,
        )
        memories = train_hdc(
            train_w,
            codebooks,
            mode=mode,
            phase_augmentation=phase_augmentation,
            confidence_scaled_updates=confidence_scaled_updates,
        )
        thresholds = fit_rejection_thresholds(val_w, memories, codebooks, mode=mode)
        rows.append({
            "status": "diagnostic_not_primary",
            "selection_split": "validation_only",
            "variant": variant,
            "encoding_mode": mode,
            "phase_augmentation": phase_augmentation,
            "confidence_scaled_updates": confidence_scaled_updates,
            "dim": dim,
            "level_count": level_count,
            "validation_macro_f1_at_fitted_rejection": thresholds.validation_macro_f1,
            "max_distance_fraction": thresholds.max_distance_fraction,
            "min_margin_fraction": thresholds.min_margin_fraction,
            "estimated_export_bytes": (dim // 8) * (level_count + 6 + 2 + len(CLASS_NAMES)),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def predict_hdc(windows, memories: np.ndarray, codebooks: Codebooks, mode: str = DEFAULT_ENCODING_MODE) -> tuple[np.ndarray, np.ndarray]:
    class_bits = _class_bits(memories, codebooks)
    y_true = []
    y_pred = []
    for window in windows:
        q = encode_window(window.raw, codebooks, mode=mode)
        y_true.append(window.class_id)
        y_pred.append(int(np.argmin(hamming(q, class_bits))))
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def hdc_distance_features(windows, memories: np.ndarray, codebooks: Codebooks, mode: str = DEFAULT_ENCODING_MODE):
    """Return four-gesture nearest-prototype distances and confidence margins."""
    class_bits = _class_bits(memories, codebooks)[:NULL_CLASS_ID]
    y_true = []
    candidates = []
    best_fraction = []
    margin_fraction = []
    all_distances = []
    for window in windows:
        query = encode_window(window.raw, codebooks, mode=mode)
        distances = hamming(query, class_bits).astype(np.float32)
        order = np.argsort(distances)
        best = int(order[0])
        second = float(distances[order[1]]) if len(order) > 1 else float(codebooks.dim)
        y_true.append(window.class_id)
        candidates.append(best)
        best_fraction.append(float(distances[best] / codebooks.dim))
        margin_fraction.append(float((second - distances[best]) / codebooks.dim))
        all_distances.append(distances)
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(candidates, dtype=np.int64),
        np.asarray(best_fraction, dtype=np.float32),
        np.asarray(margin_fraction, dtype=np.float32),
        np.asarray(all_distances, dtype=np.float32),
    )


def fit_rejection_thresholds(
    validation_windows,
    memories: np.ndarray,
    codebooks: Codebooks,
    mode: str = DEFAULT_ENCODING_MODE,
) -> RejectionThresholds:
    """Fit gesture-to-null rejection using validation data only."""
    y_true, candidates, best, margin, _ = hdc_distance_features(
        validation_windows, memories, codebooks, mode=mode
    )
    if len(y_true) == 0 or not np.any(y_true == NULL_CLASS_ID):
        raise ValueError("HDC rejection requires validation null windows")
    distance_grid = np.unique(
        np.concatenate(([float(np.min(best))], np.quantile(best, np.linspace(0.40, 1.0, 25)), [1.0]))
    )
    margin_grid = np.unique(
        np.concatenate(([0.0], np.quantile(margin, np.linspace(0.0, 0.60, 16))))
    )
    best_choice: tuple[float, float, float, float] | None = None
    for max_distance in distance_grid:
        for min_margin in margin_grid:
            prediction = candidates.copy()
            prediction[(best > max_distance) | (margin < min_margin)] = NULL_CLASS_ID
            macro_f1, *_ = macro_f1_present_classes(
                y_true, prediction, labels=list(range(len(CLASS_NAMES)))
            )
            gesture_recall = float(
                np.mean(prediction[y_true != NULL_CLASS_ID] == y_true[y_true != NULL_CLASS_ID])
            ) if np.any(y_true != NULL_CLASS_ID) else 0.0
            choice = (macro_f1, gesture_recall, -float(max_distance), float(min_margin))
            if best_choice is None or choice > best_choice:
                best_choice = choice
                selected_distance = float(max_distance)
                selected_margin = float(min_margin)
    assert best_choice is not None
    return RejectionThresholds(selected_distance, selected_margin, best_choice[0])


def predict_hdc_with_rejection(
    windows,
    memories: np.ndarray,
    codebooks: Codebooks,
    thresholds: RejectionThresholds,
    mode: str = DEFAULT_ENCODING_MODE,
):
    y_true, candidates, best, margin, distances = hdc_distance_features(
        windows, memories, codebooks, mode=mode
    )
    prediction = candidates.copy()
    prediction[
        (best > thresholds.max_distance_fraction)
        | (margin < thresholds.min_margin_fraction)
    ] = NULL_CLASS_ID
    return y_true, prediction, distances


def evaluate_hdc(
    windows,
    memories: np.ndarray,
    codebooks: Codebooks,
    rate_hz: int,
    dim: int,
    out_dir: Path,
    mode: str = DEFAULT_ENCODING_MODE,
    split_type: str = "cross_session",
) -> tuple[float, dict]:
    y_true, y_pred = predict_hdc(windows, memories, codebooks, mode=mode)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    report = prediction_report(y_true, y_pred, f"hdc_{mode}_D{dim}", rate_hz, split_type, out_dir, fail_on_collapse=False)
    return acc, report


def sweep(
    windows,
    splits: dict,
    out_csv: Path,
    mode: str = DEFAULT_ENCODING_MODE,
    split_key: str = "cross_session",
    report_split_type: str | None = None,
) -> list[dict]:
    report_split_type = report_split_type or split_key
    rows = []
    for rate in (120, 60, 30):
        rate_windows = windows if rate == 120 else resample_windows(windows, rate)
        train_w = select_windows(rate_windows, splits[split_key]["train"])
        lo, hi = fit_level_bounds(train_w)
        for dim in (1024, 2048, 4096):
            codebooks = make_codebooks(dim=dim, seed=SEED, level_min=lo, level_max=hi)
            test_w = select_windows(rate_windows, splits[split_key]["test"])
            if not train_w or not test_w:
                raise ValueError(f"HDC sweep rate {rate} D {dim}: empty train/test split")
            memories = train_hdc(train_w, codebooks, mode=mode)
            acc, report = evaluate_hdc(
                test_w,
                memories,
                codebooks,
                rate,
                dim,
                out_csv.parent,
                mode=mode,
                split_type=report_split_type,
            )
            rows.append(
                {
                    "method": "hdc",
                    "encoding_mode": mode,
                    "rate_hz": rate,
                    "split_type": report_split_type,
                    "dim": dim,
                    "accuracy": acc,
                    "macro_f1": report["macro_f1"],
                    "macro_f1_all_classes": report["macro_f1_all_classes"],
                    "present_class_count": report["present_class_count"],
                    "top_true_class": report["top_true_class"],
                    "top_true_fraction": report["top_true_fraction"],
                    "top_predicted_class": report["top_predicted_class"],
                    "top_predicted_fraction": report["top_predicted_fraction"],
                    "collapse_allowed_fraction": report["collapse_allowed_fraction"],
                    "collapse_flag": report["collapse_flag"],
                    "memory_bytes": (dim // 8) * (len(CLASS_NAMES) + HDC_EXPORT_CODEBOOK_FACTOR),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


HDC_EXPORT_CODEBOOK_FACTOR = 32 + 6 + 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    parser.add_argument("--out", default="ml/results/hdc/hdc_grid.csv")
    parser.add_argument(
        "--diagnostics-out",
        default="ml/results/hdc/hdc_validation_diagnostics.csv",
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Skip test-set grid and run only validation-only non-primary variants",
    )
    parser.add_argument("--diagnostic-phase-augmentation", action="store_true")
    parser.add_argument("--diagnostic-confidence-scaled-updates", action="store_true")
    parser.add_argument("--mode", default=DEFAULT_ENCODING_MODE, choices=["absolute", "bag", "ngram"])
    args = parser.parse_args()

    sessions, manifest_warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in manifest_warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits)
    assert_no_cross_session_leakage(splits)
    if not args.diagnostics_only:
        rows = sweep(windows, splits, Path(args.out), mode=args.mode)
        for row in rows:
            print(row)
    if not args.skip_diagnostics:
        diagnostic_rows = validation_diagnostic_sweep(
            windows,
            splits,
            Path(args.diagnostics_out),
            mode=args.mode,
            phase_augmentation=args.diagnostic_phase_augmentation,
            confidence_scaled_updates=args.diagnostic_confidence_scaled_updates,
        )
        for row in diagnostic_rows:
            print(row)


if __name__ == "__main__":
    main()
