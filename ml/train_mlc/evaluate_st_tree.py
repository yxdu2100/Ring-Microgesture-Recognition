"""Evaluate a MEMS Studio exported decision tree on ring windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_utils import prediction_report
from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.segment import CLASS_TO_ID
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_mlc.arff import read_arff
from train_mlc.compare_arff_features import infer_gyro_scale, mems_ordered_windows
from train_mlc.features import MEMS_STUDIO_GYRO_INTERNAL_PER_LSB
from train_mlc.st_tree import MLCTreeClassifier, ST_TO_PROJECT_CLASS, parse_st_tree

SEED = 20260706


def _load_windows(data_dir: str, splits_path: str, drop_invalid_windows: bool):
    sessions = load_sessions(data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not drop_invalid_windows)
    if drop_invalid_windows:
        windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, splits_path, seed=SEED)
    assert_no_cross_session_leakage(splits)
    return sessions, windows, splits


def _windows_for_session(windows, session_id: str):
    selected = [w for w in windows if w.session_id == session_id]
    if not selected:
        known = ", ".join(sorted({w.session_id for w in windows}))
        raise ValueError(f"unknown or empty session_id {session_id!r}; known sessions: {known}")
    return selected


def _target_windows(windows, splits: dict, split_type: str, test_session: str | None):
    if test_session:
        return _windows_for_session(windows, test_session), f"session_{test_session}"
    split = splits[split_type]
    selected = select_windows(windows, split["test"])
    if not selected:
        raise ValueError(f"{split_type} test split is empty")
    return selected, split_type


def _mems_studio_train_windows(windows, splits: dict, drop_tail_per_class: bool = True):
    train_w = select_windows(windows, splits["cross_session"]["train"])
    selected = []
    for label in CLASS_NAMES:
        group = sorted(
            [w for w in train_w if w.label == label],
            key=lambda w: (w.session_id, w.start_sample_id, w.window_id),
        )
        if drop_tail_per_class and group:
            group = group[:-1]
        selected.extend(group)
    return selected


def _st_confusion(y_true: np.ndarray, y_pred: np.ndarray, st_class_order: list[str]) -> np.ndarray:
    ids = [CLASS_TO_ID[ST_TO_PROJECT_CLASS[name]] for name in st_class_order]
    out = np.zeros((len(ids), len(ids)), dtype=np.int64)
    id_to_pos = {class_id: idx for idx, class_id in enumerate(ids)}
    for truth, pred in zip(y_true, y_pred):
        if int(truth) in id_to_pos and int(pred) in id_to_pos:
            out[id_to_pos[int(truth)], id_to_pos[int(pred)]] += 1
    return out


def _validation_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, parsed: dict) -> dict:
    stats = parsed["training_stats"]
    correct = int(np.count_nonzero(y_true == y_pred))
    total = int(len(y_true))
    accuracy = 100.0 * correct / total if total else 0.0
    matrix_match = False
    confusion_abs_error = ""
    if stats.confusion is not None and stats.class_order:
        cm = _st_confusion(y_true, y_pred, stats.class_order)
        matrix_match = bool(np.array_equal(cm, stats.confusion))
        confusion_abs_error = int(np.abs(cm - stats.confusion).sum())
    total_match = stats.total is not None and total == stats.total
    correct_match = stats.correct is not None and correct == stats.correct
    correct_delta = "" if stats.correct is None else correct - stats.correct
    near_pass = bool(total_match and stats.correct is not None and abs(correct - stats.correct) <= 3)
    return {
        "name": name,
        "windows": total,
        "correct": correct,
        "correct_delta": correct_delta,
        "accuracy_pct": accuracy,
        "expected_windows": stats.total if stats.total is not None else "",
        "expected_correct": stats.correct if stats.correct is not None else "",
        "expected_accuracy_pct": stats.accuracy if stats.accuracy is not None else "",
        "total_match": total_match,
        "correct_match": correct_match,
        "confusion_match": matrix_match,
        "confusion_abs_error": confusion_abs_error,
        "validation_pass": bool(total_match and correct_match and matrix_match),
        "validation_near_pass": near_pass,
    }


def _scan_sessions(windows, tree_path: Path, parsed: dict, out_csv: Path, precision: str, gyro_lsb_scale: float | None) -> list[dict]:
    rows = []
    clf = MLCTreeClassifier.from_file(tree_path, precision=precision, gyro_lsb_scale=gyro_lsb_scale)
    for session_id in sorted({w.session_id for w in windows}):
        session_w = _windows_for_session(windows, session_id)
        y_true, y_pred = clf.predict_windows(session_w)
        rows.append(_validation_row(session_id, y_true, y_pred, parsed))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_metrics_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, help="MEMS Studio ST_decision_tree_*.txt export")
    parser.add_argument("--arff", default=None, help="Optional MEMS Studio features.arff used to infer gyro feature scale")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--results-dir", default="ml/results/mlc_st")
    parser.add_argument("--split-type", default="cross_session", choices=["cross_session", "within_session"])
    parser.add_argument("--test-session", default=None)
    parser.add_argument("--validation-session", default=None)
    parser.add_argument("--precision", default="both", choices=["fp16", "fp64", "both"])
    parser.add_argument("--accel-fs-g", type=int, default=8)
    parser.add_argument("--gyro-fs-dps", type=int, default=2000)
    parser.add_argument(
        "--gyro-lsb-scale",
        type=float,
        default=None,
        help="Override gyro LSB scale for ST fixed-tree features. Defaults to ARFF-derived MEMS internal scale.",
    )
    parser.add_argument("--drop-invalid-windows", action="store_true")
    args = parser.parse_args()

    tree_path = Path(args.tree)
    parsed = parse_st_tree(tree_path)
    _sessions, windows, splits = _load_windows(args.data_dir, args.splits, args.drop_invalid_windows)
    if args.arff:
        arff = read_arff(args.arff)
        ordered_windows, _counts = mems_ordered_windows(
            windows,
            splits,
            arff.labels,
            arff_rows=arff.rows,
            feature_names=arff.feature_names,
            export_dir="ml/results/memsstudio_export",
        )
        gyro_lsb_scale = infer_gyro_scale(arff, ordered_windows)
        print(f"ARFF-derived gyro_lsb_scale={gyro_lsb_scale:.12g}")
    else:
        gyro_lsb_scale = MEMS_STUDIO_GYRO_INTERNAL_PER_LSB if args.gyro_lsb_scale is None else args.gyro_lsb_scale
    target_w, split_label = _target_windows(windows, splits, args.split_type, args.test_session)
    results_dir = Path(args.results_dir)
    precisions = ["fp16", "fp64"] if args.precision == "both" else [args.precision]

    train_stream = ordered_windows if args.arff else _mems_studio_train_windows(windows, splits, drop_tail_per_class=True)
    train_clf = MLCTreeClassifier.from_file(tree_path, precision="fp16", gyro_lsb_scale=gyro_lsb_scale)
    train_true, train_pred = train_clf.predict_windows(train_stream)
    train_validation = _validation_row("mems_concat_train_drop_tail", train_true, train_pred, parsed)

    scan_rows = _scan_sessions(windows, tree_path, parsed, results_dir / "st_tree_session_scan.csv", "fp16", gyro_lsb_scale)
    validation_rows = [train_validation, *scan_rows]
    _write_metrics_csv(validation_rows, results_dir / "st_tree_validation.csv")
    passing = [row["name"] for row in validation_rows if row["validation_pass"]]
    near_passing = [row["name"] for row in validation_rows if row["validation_near_pass"]]
    if passing:
        print(f"ST training-matrix validation PASS for session(s): {', '.join(passing)}")
    elif near_passing:
        print(f"ST training-matrix validation NEAR-PASS for: {', '.join(near_passing)}")
    else:
        print("ST training-matrix validation did not match; see st_tree_validation.csv")

    metric_rows = []
    for precision in precisions:
        clf = MLCTreeClassifier(
            parsed["rules"],
            parsed["feature_tokens"],
            precision=precision,
            accel_fs_g=args.accel_fs_g,
            gyro_fs_dps=args.gyro_fs_dps,
            gyro_lsb_scale=gyro_lsb_scale,
        )
        y_true, y_pred = clf.predict_windows(target_w)
        report = prediction_report(
            y_true,
            y_pred,
            f"st_mlc_fixed_{precision}",
            120,
            split_label,
            results_dir,
            fail_on_collapse=False,
        )
        row = {
            "method": "st_mlc_fixed",
            "precision": precision,
            "split": split_label,
            "windows": len(target_w),
            "macro_f1": report["macro_f1"],
            "macro_f1_all_classes": report["macro_f1_all_classes"],
            "top_true_class": report["top_true_class"],
            "top_true_fraction": report["top_true_fraction"],
            "top_predicted_class": report["top_predicted_class"],
            "top_predicted_fraction": report["top_predicted_fraction"],
            "collapse_flag": report["collapse_flag"],
            "gyro_lsb_scale": gyro_lsb_scale,
            "st_validation_name": train_validation["name"],
            "st_validation_correct": train_validation["correct"],
            "st_validation_expected_correct": train_validation["expected_correct"],
            "st_validation_near_pass": train_validation["validation_near_pass"],
        }
        if args.validation_session:
            val_w = _windows_for_session(windows, args.validation_session)
            val_true, val_pred = clf.predict_windows(val_w)
            row.update({f"validation_{k}": v for k, v in _validation_row(args.validation_session, val_true, val_pred, parsed).items()})
        metric_rows.append(row)
        print(
            f"{precision} {split_label}: macro_f1={report['macro_f1']:.4f}, "
            f"top_pred={report['top_predicted_class']} {report['top_predicted_fraction']:.1%}"
        )

    _write_metrics_csv(metric_rows, results_dir / "st_tree_fixed_metrics.csv")
    print(f"wrote ST fixed-tree results to {results_dir}")


if __name__ == "__main__":
    main()
