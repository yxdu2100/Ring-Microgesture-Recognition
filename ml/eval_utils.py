"""Evaluation reporting helpers shared by trainers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

from ringdata.segment import CLASS_NAMES


class DegeneratePredictionError(RuntimeError):
    pass


def prediction_report(
    y_true,
    y_pred,
    method: str,
    rate_hz: int,
    split_type: str,
    out_dir: str | Path,
    fail_on_collapse: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(CLASS_NAMES)))
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < len(labels) and 0 <= pred < len(labels):
            cm[truth, pred] += 1

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    pred_counts = np.bincount(y_pred, minlength=len(labels)) if len(y_pred) else np.zeros(len(labels), dtype=np.int64)
    top_pred = int(np.argmax(pred_counts)) if len(pred_counts) else 0
    top_fraction = float(pred_counts[top_pred] / len(y_pred)) if len(y_pred) else 0.0
    stem = f"{method}_{rate_hz}hz_{split_type}"

    with (out_dir / f"{stem}_per_class.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support", "predicted"])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow([idx, name, precision[idx], recall[idx], f1[idx], int(support[idx]), int(pred_counts[idx])])

    with (out_dir / f"{stem}_confusion.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *cm[idx].tolist()])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{method} {rate_hz} Hz {split_type}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_confusion.png", dpi=160)
    plt.close(fig)

    report = {
        "macro_f1": macro_f1,
        "top_predicted_class": CLASS_NAMES[top_pred],
        "top_predicted_fraction": top_fraction,
    }
    if fail_on_collapse and top_fraction > 0.90:
        raise DegeneratePredictionError(
            f"{method} {rate_hz} Hz predicts {CLASS_NAMES[top_pred]} for "
            f"{top_fraction:.1%} of test windows"
        )
    return report
