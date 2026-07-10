"""One entry point for parsing, segmentation, frozen splits, and evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_mlc.features import MEMS_STUDIO_GYRO_INTERNAL_PER_LSB

SEED = 20260706


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


def _fp_per_hour_null(windows, splits: dict) -> float:
    null_test = [w for w in windows if w.window_id in splits["cross_session"]["test"] and w.label == "null"]
    if not null_test:
        return float("nan")
    seconds = len(null_test) * 64 / 120.0
    return 0.0 / (seconds / 3600.0) ## what is this?


def _write_summary(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w") as f:
        f.write("# ML Summary\n\n")
        f.write("| method | rate_hz | split_type | macro_f1 | FP/hr held-out null | top predicted | top fraction |\n")
        f.write("|---|---:|---|---:|---:|---|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['rate_hz']} | {row['split_type']} | "
                f"{row['macro_f1']:.4f} | {row.get('fp_per_hr_null', float('nan')):.3f} | "
                f"{row.get('top_predicted_class', '')} | {row.get('top_predicted_fraction', float('nan')):.3f} |\n"
            )


def _append_metrics_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "rate_hz",
                "split_type",
                "macro_f1",
                "fp_per_hr_null",
                "top_predicted_class",
                "top_predicted_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


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
    _session_report(sessions, windows, results / "session_report.csv")
    splits = build_or_load_splits(windows, args.splits, seed=SEED)
    assert_no_cross_session_leakage(splits)
    (results / "split_summary.json").write_text(json.dumps(splits, indent=2) + "\n")

    if not splits["cross_session"]["test"]:
        raise ValueError(
            "No held-out cross-session test split exists. Collect at least two sessions, "
            "or run `PYTHONPATH=ml python3 ml/make_synthetic.py` and then use "
            "`--data-dir ml/synthetic_data` for a plumbing test."
        )

    summary_rows: list[dict] = []
    from train_mlc.tree import export_tree_header, train_tree
    from eval_utils import prediction_report

    clf, names, y_test, pred = train_tree(windows, splits)
    tree_report = prediction_report(y_test, pred, "mlc_tree", 120, "cross_session", results, fail_on_collapse=False)
    export_tree_header(
        clf,
        names,
        Path("firmware/src/classifiers/generated/tree_sw.h"),
        _git_hash(),
        tree_report["macro_f1"],
    )
    summary_rows.append(
        {
            "method": "mlc_tree",
            "rate_hz": 120,
            "split_type": "cross_session",
            "macro_f1": tree_report["macro_f1"],
            "fp_per_hr_null": _fp_per_hour_null(windows, splits),
            "top_predicted_class": tree_report["top_predicted_class"],
            "top_predicted_fraction": tree_report["top_predicted_fraction"],
        }
    )

    if args.st_tree:
        from train_mlc.st_tree import MLCTreeClassifier

        st_test_w = select_windows(windows, splits["cross_session"]["test"])
        for precision in ("fp16", "fp64"):
            st_clf = MLCTreeClassifier.from_file(
                args.st_tree,
                precision=precision,
                gyro_lsb_scale=args.st_tree_gyro_lsb_scale,
            )
            y_true, y_pred = st_clf.predict_windows(st_test_w)
            st_report = prediction_report(
                y_true,
                y_pred,
                f"st_mlc_fixed_{precision}",
                120,
                "cross_session",
                results,
                fail_on_collapse=False,
            )
            summary_rows.append(
                {
                    "method": f"st_mlc_fixed_{precision}",
                    "rate_hz": 120,
                    "split_type": "cross_session",
                    "macro_f1": st_report["macro_f1"],
                    "fp_per_hr_null": _fp_per_hour_null(windows, splits),
                    "top_predicted_class": st_report["top_predicted_class"],
                    "top_predicted_fraction": st_report["top_predicted_fraction"],
                }
            )

    from train_hdc.train import sweep as hdc_sweep

    hdc_rows = hdc_sweep(windows, splits, results / "hdc_grid.csv")
    for row in hdc_rows:
        summary_rows.append(
            {
                "method": f"hdc_D{row['dim']}",
                "rate_hz": row["rate_hz"],
                "split_type": row["split_type"],
                "macro_f1": row["macro_f1"],
                "fp_per_hr_null": _fp_per_hour_null(windows, splits),
                "top_predicted_class": row["top_predicted_class"],
                "top_predicted_fraction": row["top_predicted_fraction"],
            }
        )

    if not args.skip_cnn:
        try:
            from train_cnn.train import train_one_rate

            cnn_dir = results / "cnn"
            for rate in (120, 60, 30):
                row = train_one_rate(windows, splits, rate, cnn_dir)
                summary_rows.append(
                    {
                        "method": "cnn_float",
                        "rate_hz": rate,
                        "split_type": row["split_type"],
                        "macro_f1": row["macro_f1"],
                        "fp_per_hr_null": _fp_per_hour_null(windows, splits),
                        "top_predicted_class": row["top_predicted_class"],
                        "top_predicted_fraction": row["top_predicted_fraction"],
                    }
                )
        except ImportError as exc:
            print(f"skipping CNN because TensorFlow is not importable: {exc}")

    _append_metrics_csv(summary_rows, results / "summary.csv")
    _write_summary(summary_rows, results / "summary.md")
    collapsed = [r for r in summary_rows if r.get("top_predicted_fraction", 0.0) > 0.90]
    if collapsed:
        msg = "; ".join(
            f"{r['method']}@{r['rate_hz']}Hz predicts {r['top_predicted_class']} "
            f"{r['top_predicted_fraction']:.1%}"
            for r in collapsed
        )
        raise RuntimeError(f"Degenerate prediction collapse detected after writing reports: {msg}")
    print(f"wrote results to {results}")


if __name__ == "__main__":
    main()
