"""Generate the canonical paper-number digest from frozen result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "mlc_sensor_tree": "MLC sensor tree",
    "cnn_float": "CNN float",
    "hdc_D2048_reject": "HDC D=2048 + rejection",
}
METHOD_ORDER = list(METHOD_LABELS)
FOLDS = [f"within_user_{index:02d}" for index in range(1, 6)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _mean_sd(values: pd.Series, digits: int = 2) -> str:
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def _guided_summary(events: pd.DataFrame, matches: pd.DataFrame, method: str, consecutive: int) -> tuple:
    rows = events[
        (events.method == method)
        & (events.stream_kind == "guided_test")
        & (events.consecutive_windows == consecutive)
    ]
    correct = int(rows.correct_events.sum())
    total = int(rows.gesture_events.sum())
    correct_matches = matches[
        (matches.method == method)
        & (matches.stream_kind == "guided_test")
        & (matches.consecutive_windows == consecutive)
        & (matches.outcome == "correct")
    ]
    return correct, total, correct / total, float(correct_matches.latency_ms.median())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ml/results/final"))
    parser.add_argument(
        "--feature-results-dir",
        type=Path,
        default=Path("ml/results/hdc_feature_diagnostic"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS_DIGEST.md"))
    args = parser.parse_args()

    session = pd.read_csv(args.results_dir / "session_report.csv")
    window = pd.read_csv(args.results_dir / "fold_window_metrics.csv")
    events = pd.read_csv(args.results_dir / "event_metrics.csv")
    matches = pd.read_csv(args.results_dir / "event_matches.csv")
    split = json.loads(Path("ml/splits_within_user.json").read_text())
    quant = json.loads(
        (args.results_dir / "folds/within_user_01/cnn/cnn_120hz_int8.metrics.json").read_text()
    )
    feature_window = pd.read_csv(args.feature_results_dir / "fold_window_metrics.csv")
    feature_events = pd.read_csv(args.feature_results_dir / "event_metrics.csv")
    feature_matches = pd.read_csv(args.feature_results_dir / "event_matches.csv")

    lines: list[str] = []
    lines += [
        "# Canonical results digest",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `ml/results/final`.",
        "The final free-living streams were used only for locked post-training evaluation; all rejection thresholds were fitted on the frozen validation sessions.",
        "",
        "## Reproducibility identifiers",
        "",
        f"- Git commit: `{_git_commit()}` (canonical manifest/split additions may be uncommitted; use the hashes below).",
        f"- Dataset hash stored in the split: `{split['dataset_hash']}`.",
        f"- Split JSON SHA-256: `{_sha256(Path('ml/splits_within_user.json'))}`.",
        f"- Dataset manifest SHA-256: `{_sha256(Path('ml/dataset_manifest.csv'))}`.",
        "- Split guard: all five gesture train/validation/test assignments, structured-null roles, and their window IDs matched the pre-final split byte-for-byte.",
        "- Canonical command: `PYTHONPATH=ml python ml/run_all.py --st-tree-dir ml/st_trees --results-dir ml/results/final --skip-mlc-proxy --skip-hdc-features`.",
        "",
        "## Dataset",
        "",
    ]

    grouped = session.groupby(["data_role", "usage"], sort=False).agg(
        sessions=("session_id", "count"),
        samples=("samples", "sum"),
        minutes=("recorded_minutes", "sum"),
        go_markers=("go_markers", "sum"),
        valid_windows=("valid_windows", "sum"),
        gaps=("gap_count", "sum"),
        missing=("missing_samples", "sum"),
    ).reset_index()
    lines += [
        "| Role | Usage | Sessions | Samples | Recorded min | GO markers | Valid windows | Gaps | Missing samples |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped.itertuples(index=False):
        lines.append(
            f"| {row.data_role} | {row.usage} | {row.sessions} | {row.samples:,} | "
            f"{row.minutes:.2f} | {row.go_markers:,} | {row.valid_windows:,} | {row.gaps} | {row.missing} |"
        )
    lines += [
        "",
        "Seven of 1,200 guided gesture windows exceeded the perform interval and were excluded by the frozen rule, leaving 1,193 valid guided windows. The final-test exposure is 159.71 recorded minutes (2.662 h) across two recording parts.",
        "",
        "### Session-level record",
        "",
        "| Session | Role / usage | Protocol | Samples | Min | GO | Valid windows | Gaps / missing | HW timestamps |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in session.itertuples(index=False):
        lines.append(
            f"| `{row.session_id}` | {row.data_role} / {row.usage} | {row.guided_protocol} | "
            f"{row.samples:,} | {row.recorded_minutes:.2f} | {row.go_markers} | {row.valid_windows:,} | "
            f"{row.gap_count} / {row.missing_samples} | {row.hardware_pct:.1f}% |"
        )

    lines += [
        "",
        "## Window-level gesture classification",
        "",
        "Gesture macro-F1 is calculated over the four gesture classes present in the held-out guided windows; rejection to null is penalized but null has no test support in this table.",
        "",
        "| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± sample SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        rows = window[window.method == method].set_index("fold_id").loc[FOLDS]
        values = rows.gesture_macro_f1
        fold_cells = " | ".join(_fmt(value) for value in values)
        lines.append(
            f"| {METHOD_LABELS[method]} | {fold_cells} | {_fmt(values.mean())} ± {_fmt(values.std(ddof=1))} |"
        )

    lines += [
        "",
        "## Continuous guided-event evaluation",
        "",
        "Recall is pooled across all 1,193 held-out gesture events. Latency is the pooled median onset-to-correct-activation latency; missed and wrong-class events do not have a correct latency.",
        "",
        "| Method | M | Correct / events | Event recall | Median latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        for consecutive in (1, 2):
            correct, total, recall, latency = _guided_summary(events, matches, method, consecutive)
            lines.append(
                f"| {METHOD_LABELS[method]} | {consecutive} | {correct:,} / {total:,} | {recall:.4f} | {latency:.1f} |"
            )
    lines += [
        "",
        "CNN M=2 recall can exceed M=1 because consecutive confirmation suppresses early false/wrong activations that otherwise consume the evaluator's one-to-one event match; it does not mean M=2 creates more raw positive windows.",
        "",
        "## Free-living false activations per hour",
        "",
        "Each fold model is evaluated over the same exposure. Values are per-fold FP/hr followed by mean ± sample SD across the five frozen fold models. Development and final test are never pooled.",
        "",
        "### Development stream (0.669 h)",
        "",
        "| Method | M | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        for consecutive in (1, 2):
            rows = events[
                (events.method == method)
                & (events.stream_kind == "free_living_development")
                & (events.consecutive_windows == consecutive)
            ].set_index("fold_id").loc[FOLDS]
            values = rows.false_activations_per_hour
            lines.append(
                f"| {METHOD_LABELS[method]} | {consecutive} | "
                + " | ".join(f"{value:.2f}" for value in values)
                + f" | {_mean_sd(values)} |"
            )
    lines += [
        "",
        "### Final test stream (2.662 h; frozen thresholds)",
        "",
        "| Method | M | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        for consecutive in (1, 2):
            rows = events[
                (events.method == method)
                & (events.stream_kind == "free_living_final_test")
                & (events.consecutive_windows == consecutive)
            ].set_index("fold_id").loc[FOLDS]
            values = rows.false_activations_per_hour
            lines.append(
                f"| {METHOD_LABELS[method]} | {consecutive} | "
                + " | ".join(f"{value:.2f}" for value in values)
                + f" | {_mean_sd(values)} |"
            )

    feature_method = "hdc_D2048_features_reject"
    feature_rows = feature_window[feature_window.method == feature_method]
    lines += [
        "",
        "## Feature-HDC representation ablation — Python-only diagnostic",
        "",
        "This diagnostic uses tree-style engineered features and has no firmware export or resource claim. It was run before final-test collection; because the non-free-living split is unchanged, its window, guided, and development results remain comparable, but it has no final-test row.",
        "",
        "| Window macro-F1 mean ± SD | M | Guided correct / events | Guided recall | Median latency (ms) | Development FP/hr mean ± SD |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for consecutive in (1, 2):
        correct, total, recall, latency = _guided_summary(
            feature_events, feature_matches, feature_method, consecutive
        )
        fp = feature_events[
            (feature_events.method == feature_method)
            & (feature_events.stream_kind == "free_living_development")
            & (feature_events.consecutive_windows == consecutive)
        ].false_activations_per_hour
        lines.append(
            f"| {feature_rows.gesture_macro_f1.mean():.4f} ± {feature_rows.gesture_macro_f1.std(ddof=1):.4f} | "
            f"{consecutive} | {correct:,} / {total:,} | {recall:.4f} | {latency:.1f} | {_mean_sd(fp)} |"
        )

    lines += [
        "",
        "## Fold-1 CNN full-int8 gate",
        "",
        "Acceptance was decided on validation retention only (required ≥95%); test results were reported after the gate.",
        "",
        "| Split | Float macro-F1 | Int8 macro-F1 | Retention | Gate |",
        "|---|---:|---:|---:|---|",
        f"| Validation | {quant['float_macro_f1']:.5f} | {quant['int8_macro_f1']:.5f} | {quant['retention'] * 100:.2f}% | {'PASS' if quant['accepted'] else 'FAIL'} |",
        f"| Test | {quant['test_float_macro_f1']:.5f} | {quant['test_int8_macro_f1']:.5f} | 100.00% | Report-only |",
        "",
        "## Frozen HDC rejection thresholds",
        "",
        "| Fold | Max distance fraction | Min margin fraction | Validation macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    hdc = window[window.method == "hdc_D2048_reject"].set_index("fold_id").loc[FOLDS]
    for fold_id, row in hdc.iterrows():
        lines.append(
            f"| {fold_id} | {row.max_distance_fraction:.8f} | {row.min_margin_fraction:.8f} | {row.validation_macro_f1:.4f} |"
        )

    lines += [
        "",
        "## Paper figures and system measurements",
        "",
        "- Aggregated row-normalized 4×5 confusion panel: [`ml/results/final/figures/primary_methods_confusion_4x5.pdf`](../ml/results/final/figures/primary_methods_confusion_4x5.pdf). Individual PNG/PDF figures and count/normalized CSVs are in the same directory.",
        "- Measured power, flash, static RAM, and measurement limitations: [POWER_MEMORY_RESULTS.md](POWER_MEMORY_RESULTS.md).",
        "- Primary result source files: `fold_window_metrics.csv`, `event_metrics.csv`, `event_matches.csv`, and `chronological_predictions.csv` under `ml/results/final`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote canonical digest to {args.out}")


if __name__ == "__main__":
    main()
