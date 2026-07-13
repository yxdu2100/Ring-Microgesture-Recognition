"""Aggregate the five frozen folds into paper-format 4x5 confusion matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GESTURES = ["double_side_tap", "double_pinch", "pinch_hold", "double_flick"]
OUTPUTS = [*GESTURES, "null"]
METHODS = {
    "mlc_sensor_tree": "MLC sensor tree",
    "cnn_float": "CNN (float evaluation)",
    "hdc_D2048_reject": "HDC D=2048",
}


def _confusion_path(results: Path, fold_id: str, method: str) -> Path:
    fold = results / "folds" / fold_id
    if method == "cnn_float":
        return fold / "cnn" / f"{method}_120hz_{fold_id}_confusion.csv"
    return fold / f"{method}_120hz_{fold_id}_confusion.csv"


def _read_confusion(path: Path) -> np.ndarray:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if rows[0][1:] != OUTPUTS:
        raise ValueError(f"{path}: unexpected prediction columns {rows[0][1:]}")
    by_label = {row[0]: np.asarray(row[1:], dtype=np.int64) for row in rows[1:]}
    return np.vstack([by_label[label] for label in GESTURES])


def _write_matrix(path: Path, matrix: np.ndarray, normalized: bool) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *OUTPUTS])
        for label, row in zip(GESTURES, matrix, strict=True):
            writer.writerow([label, *([f"{value:.8f}" for value in row] if normalized else row.tolist())])


def _draw(ax, normalized: np.ndarray, title: str) -> None:
    image = ax.imshow(normalized * 100.0, vmin=0.0, vmax=100.0, cmap="Blues", aspect="auto")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = normalized[row, col] * 100.0
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 55.0 else "black",
            )
    ax.set_xticks(range(len(OUTPUTS)), ["Side tap", "Pinch", "Hold", "Flick", "Reject"], rotation=35, ha="right")
    ax.set_yticks(range(len(GESTURES)), ["Side tap", "Pinch", "Hold", "Flick"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True gesture")
    ax.set_title(title)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ml/results/final"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = [f"within_user_{index:02d}" for index in range(1, 6)]

    normalized_by_method: dict[str, np.ndarray] = {}
    for method, title in METHODS.items():
        counts = sum(
            (_read_confusion(_confusion_path(args.results_dir, fold_id, method)) for fold_id in fold_ids),
            start=np.zeros((len(GESTURES), len(OUTPUTS)), dtype=np.int64),
        )
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals != 0,
        )
        normalized_by_method[method] = normalized
        _write_matrix(out_dir / f"{method}_confusion_4x5_counts.csv", counts, normalized=False)
        _write_matrix(out_dir / f"{method}_confusion_4x5_normalized.csv", normalized, normalized=True)

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        image = _draw(ax, normalized, title)
        fig.colorbar(image, ax=ax, label="Row percentage (%)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_confusion_4x5.png", dpi=300)
        fig.savefig(out_dir / f"{method}_confusion_4x5.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharey=True, constrained_layout=True)
    image = None
    for ax, (method, title) in zip(axes, METHODS.items(), strict=True):
        image = _draw(ax, normalized_by_method[method], title)
    assert image is not None
    fig.colorbar(image, ax=axes, label="Row percentage (%)", shrink=0.92)
    fig.savefig(out_dir / "primary_methods_confusion_4x5.png", dpi=300)
    fig.savefig(out_dir / "primary_methods_confusion_4x5.pdf")
    plt.close(fig)
    print(f"wrote 4x5 confusion matrices to {out_dir}")


if __name__ == "__main__":
    main()
