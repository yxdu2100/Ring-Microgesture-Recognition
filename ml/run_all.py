"""One entry point for parsing, segmentation, frozen splits, and evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes, prediction_report
from ringdata import CLASS_NAMES, load_sessions, resample_windows, segment_sessions
from ringdata.splits import EXCLUDED_SESSIONS, assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_mlc.features import MEMS_STUDIO_GYRO_INTERNAL_PER_LSB

SEED = 20260706
NULL_CLASS_ID = CLASS_NAMES.index("null")
WINDOW_SAMPLES = 128
NULL_STEP_SAMPLES = 64


def _session_report(sessions, windows, out_path: Path) -> None:
    counts = Counter(w.label for w in windows)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session_id",
                "mode",
                "samples",
                "sample_rate_hz",
                "gap_count",
                "missing_samples",
                "hardware_pct",
                "marker_go_count",
                "window_count",
            ]
        )
        for session in sessions:
            writer.writerow(
                [
                    session.session_id,
                    session.mode,
                    len(session.raw),
                    session.sample_rate_hz,
                    session.gap_count,
                    session.missing_sample_count,
                    f"{session.hardware_flag_percentage:.3f}",
                    sum(1 for m in session.markers if m.event_type == "go"),
                    sum(1 for w in windows if w.session_id == session.session_id),
                ]
            )
    print(f"class window counts: {dict(counts)}")


def _write_segmentation_quality(windows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_id",
                "session_id",
                "label",
                "cue_sample_id",
                "initial_onset_sample_id",
                "onset_sample_id",
                "window_end_offset_samples",
                "perform_window_overrun_samples",
                "energy_fraction_initial",
                "energy_fraction_final",
                "reanchored",
                "reanchor_reason",
            ]
        )
        for w in windows:
            writer.writerow(
                [
                    w.window_id,
                    w.session_id,
                    w.label,
                    w.cue_sample_id,
                    w.initial_onset_sample_id,
                    w.onset_sample_id,
                    w.cue_to_window_end_samples,
                    w.perform_window_overrun_samples,
                    w.energy_fraction_initial,
                    w.energy_fraction_final,
                    w.reanchored,
                    w.reanchor_reason,
                ]
            )


def _drop_invalid_windows(windows):
    dropped = [w for w in windows if w.perform_window_overrun_samples > 0]
    kept = [w for w in windows if w.perform_window_overrun_samples <= 0]
    if dropped:
        print("dropping invalid overrun windows:")
        for w in dropped:
            print(
                f"  {w.window_id} onset_offset={w.onset_sample_id - w.cue_sample_id} "
                f"window_end_offset={w.cue_to_window_end_samples} "
                f"overrun={w.perform_window_overrun_samples}"
            )
    return kept, dropped


def _drop_excluded_sessions(sessions, windows):
    kept_sessions = [s for s in sessions if s.session_id not in EXCLUDED_SESSIONS]
    kept_windows = [w for w in windows if w.session_id not in EXCLUDED_SESSIONS]
    return kept_sessions, kept_windows


def _prediction_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    macro_f1, macro_f1_all, precision, recall, f1, support = macro_f1_present_classes(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )
    pred_counts = np.bincount(y_pred, minlength=len(CLASS_NAMES)) if len(y_pred) else np.zeros(len(CLASS_NAMES), dtype=np.int64)
    null_mask = y_true == NULL_CLASS_ID
    null_windows = int(np.count_nonzero(null_mask))
    null_fp = int(np.count_nonzero(null_mask & (y_pred != NULL_CLASS_ID)))
    null_fpr = float(null_fp / null_windows) if null_windows else float("nan")
    null_hours = (null_windows * NULL_STEP_SAMPLES / 120.0) / 3600.0
    null_fp_per_hour = float(null_fp / null_hours) if null_hours > 0 else float("nan")

    row = {
        "macro_f1": macro_f1,
        "macro_f1_all_classes": macro_f1_all,
        "null_windows": null_windows,
        "null_effective_independent_windows": null_windows * NULL_STEP_SAMPLES / WINDOW_SAMPLES,
        "null_false_positives": null_fp,
        "null_fpr": null_fpr,
        "null_fp_per_hour": null_fp_per_hour,
    }
    for idx, name in enumerate(CLASS_NAMES):
        row[f"{name}_precision"] = float(precision[idx])
        row[f"{name}_recall"] = float(recall[idx])
        row[f"{name}_f1"] = float(f1[idx])
        row[f"{name}_support"] = int(support[idx])
        row[f"{name}_predicted"] = int(pred_counts[idx])
    return row


def _record_predictions(
    rows: list[dict],
    y_true,
    y_pred,
    method: str,
    rate_hz: int,
    split_type: str,
    fold_id: str,
    out_dir: Path,
) -> None:
    prediction_report(
        y_true,
        y_pred,
        method,
        rate_hz,
        fold_id,
        out_dir,
        fail_on_collapse=False,
    )
    row = {
        "method": method,
        "rate_hz": rate_hz,
        "split_type": split_type,
        "fold_id": fold_id,
    }
    row.update(_prediction_metrics(y_true, y_pred))
    rows.append(row)


def _aggregate_rows(rows: list[dict], split_type: str = "cross_session_loso") -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["split_type"] == split_type:
            groups[(row["method"], int(row["rate_hz"]), row["split_type"])].append(row)

    aggregate = []
    metric_keys = [
        "macro_f1",
        "macro_f1_all_classes",
        "null_fpr",
        "null_fp_per_hour",
        "null_effective_independent_windows",
    ]
    for name in CLASS_NAMES:
        metric_keys.extend([f"{name}_f1", f"{name}_recall"])

    for (method, rate_hz, group_split), group in sorted(groups.items()):
        out = {
            "method": method,
            "rate_hz": rate_hz,
            "split_type": group_split,
            "folds": len(group),
        }
        for key in metric_keys:
            values = np.array([float(row[key]) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            out[f"{key}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out["null_windows_total"] = int(sum(int(row["null_windows"]) for row in group))
        out["null_false_positives_total"] = int(sum(int(row["null_false_positives"]) for row in group))
        aggregate.append(out)
    return aggregate


def _write_rows(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_mean_std(row: dict, key: str, digits: int = 4) -> str:
    return f"{row[f'{key}_mean']:.{digits}f} +/- {row[f'{key}_std']:.{digits}f}"


def _write_summary(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w") as f:
        f.write("# ML Summary\n\n")
        f.write("Primary metric: cross-session LOSO mean +/- sample std across gesture-session folds. ")
        f.write("The left-hand session `20260706_002` is excluded from all model splits.\n\n")
        f.write("| method | rate_hz | split_type | folds | macro-F1 | null FPR | null FP/hr | gesture recall mean |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            gesture_recalls = [
                row[f"{name}_recall_mean"]
                for name in CLASS_NAMES
                if name != "null"
            ]
            gesture_recall = float(np.mean(gesture_recalls)) if gesture_recalls else float("nan")
            f.write(
                f"| {row['method']} | {row['rate_hz']} | {row['split_type']} | {row['folds']} | "
                f"{_fmt_mean_std(row, 'macro_f1')} | {_fmt_mean_std(row, 'null_fpr')} | "
                f"{_fmt_mean_std(row, 'null_fp_per_hour', digits=2)} | {gesture_recall:.4f} |\n"
            )
        f.write("\nNull-only test windows use 50% overlap: report `null_effective_independent_windows_*` ")
        f.write("in `summary.csv` as the approximate independent null count.\n")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--results-dir", default="ml/results")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--drop-invalid-windows", action="store_true")
    parser.add_argument("--use-existing-splits", action="store_true")
    parser.add_argument("--st-tree", default=None, help="Optional MEMS Studio ST_decision_tree_*.txt fixed tree to evaluate")
    parser.add_argument("--st-tree-gyro-lsb-scale", type=float, default=MEMS_STUDIO_GYRO_INTERNAL_PER_LSB)
    args = parser.parse_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not args.drop_invalid_windows)
    _write_segmentation_quality(windows, results / "segmentation_quality.csv")
    if args.drop_invalid_windows:
        windows, _dropped = _drop_invalid_windows(windows)
    sessions, windows = _drop_excluded_sessions(sessions, windows)
    _session_report(sessions, windows, results / "session_report.csv")
    splits = build_or_load_splits(
        windows,
        args.splits,
        seed=SEED,
        force_rebuild=not args.use_existing_splits,
    )
    assert_no_cross_session_leakage(splits)
    (results / "split_summary.json").write_text(json.dumps(splits, indent=2) + "\n")

    folds = splits.get("cross_session_loso") or [splits["cross_session"]]
    if not folds or not folds[0]["test"]:
        raise ValueError(
            "No held-out cross-session test split exists. Collect at least two sessions, "
            "or run `PYTHONPATH=ml python3 ml/make_synthetic.py` and then use "
            "`--data-dir ml/synthetic_data` for a plumbing test."
        )

    fold_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    from train_mlc.tree import train_tree

    fold_dir = results / "folds"
    for fold in folds:
        fold_id = fold.get("fold_id", "cross_session")
        fold_splits = {"cross_session": fold}

        clf, names, y_test, pred = train_tree(windows, fold_splits)
        _record_predictions(
            fold_rows,
            y_test,
            pred,
            "mlc_tree",
            120,
            "cross_session_loso",
            fold_id,
            fold_dir,
        )

        if args.st_tree:
            from train_mlc.st_tree import MLCTreeClassifier

            st_test_w = select_windows(windows, fold["test"])
            for precision in ("fp16", "fp64"):
                st_clf = MLCTreeClassifier.from_file(
                    args.st_tree,
                    precision=precision,
                    gyro_lsb_scale=args.st_tree_gyro_lsb_scale,
                )
                y_true, y_pred = st_clf.predict_windows(st_test_w)
                _record_predictions(
                    fold_rows,
                    y_true,
                    y_pred,
                    f"st_mlc_fixed_{precision}",
                    120,
                    "cross_session_loso",
                    fold_id,
                    fold_dir,
                )

        from train_hdc.encode import fit_level_bounds, make_codebooks
        from train_hdc.train import DEFAULT_ENCODING_MODE, HDC_EXPORT_CODEBOOK_FACTOR, predict_hdc, train_hdc

        for rate in (120, 60, 30):
            rate_windows = windows if rate == 120 else resample_windows(windows, rate)
            train_w = select_windows(rate_windows, fold["train"])
            test_w = select_windows(rate_windows, fold["test"])
            lo, hi = fit_level_bounds(train_w)
            for dim in (1024, 2048, 4096):
                codebooks = make_codebooks(dim=dim, seed=SEED, level_min=lo, level_max=hi)
                memories = train_hdc(train_w, codebooks, mode=DEFAULT_ENCODING_MODE)
                y_true, y_pred = predict_hdc(test_w, memories, codebooks, mode=DEFAULT_ENCODING_MODE)
                _record_predictions(
                    fold_rows,
                    y_true,
                    y_pred,
                    f"hdc_D{dim}",
                    rate,
                    "cross_session_loso",
                    fold_id,
                    fold_dir,
                )

        if not args.skip_cnn:
            try:
                from train_cnn.train import train_one_rate

                cnn_dir = results / "cnn" / fold_id
                for rate in (120, 60, 30):
                    row = train_one_rate(
                        windows,
                        fold_splits,
                        rate,
                        cnn_dir,
                        return_predictions=True,
                        report_split_type=fold_id,
                    )
                    _record_predictions(
                        fold_rows,
                        row["_y_true"],
                        row["_y_pred"],
                        "cnn_float",
                        rate,
                        "cross_session_loso",
                        fold_id,
                        fold_dir,
                    )
            except ImportError as exc:
                print(f"skipping CNN because TensorFlow is not importable: {exc}")

    # Within-session is a ceiling/upper-bound diagnostic. It is not the headline result.
    within_wrapper = {"within_session": splits["within_session"]}
    try:
        clf, names, y_test, pred = train_tree(windows, within_wrapper, split_type="within_session")
        within_report = prediction_report(
            y_test,
            pred,
            "mlc_tree",
            120,
            "within_session_ceiling",
            results / "within_session_ceiling",
            fail_on_collapse=False,
        )
        diagnostic = {
            "method": "mlc_tree",
            "rate_hz": 120,
            "split_type": "within_session_ceiling",
            "note": "upper_bound_diagnostic_only",
        }
        diagnostic.update(_prediction_metrics(y_test, pred))
        diagnostic_rows.append(diagnostic)
    except Exception as exc:
        print(f"within-session diagnostic skipped: {exc}")

    aggregate_rows = _aggregate_rows(fold_rows)
    _write_rows(fold_rows, results / "fold_metrics.csv")
    _write_rows(aggregate_rows, results / "summary.csv")
    _write_rows(diagnostic_rows, results / "within_session_ceiling.csv")
    _write_summary(aggregate_rows, results / "summary.md")
    collapsed = [r for r in aggregate_rows if r.get("macro_f1_mean", 0.0) <= 0.0]
    if collapsed:
        msg = "; ".join(f"{r['method']}@{r['rate_hz']}Hz" for r in collapsed)
        raise RuntimeError(f"Degenerate zero macro-F1 detected after writing reports: {msg}")
    print(f"wrote LOSO results to {results}")


if __name__ == "__main__":
    main()
