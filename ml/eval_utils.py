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


def macro_f1_present_classes(y_true, y_pred, labels: list[int] | None = None) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return macro-F1 over classes present in y_true, plus all-class details."""
    if labels is None:
        labels = list(range(len(CLASS_NAMES)))
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0
    macro_present = float(np.mean(f1[present])) if np.any(present) else 0.0
    macro_all = float(np.mean(f1)) if len(f1) else 0.0
    return macro_present, macro_all, precision, recall, f1, support


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

    macro_f1, macro_f1_all, precision, recall, f1, support = macro_f1_present_classes(
        y_true, y_pred, labels
    )
    true_counts = np.bincount(y_true, minlength=len(labels)) if len(y_true) else np.zeros(len(labels), dtype=np.int64)
    pred_counts = np.bincount(y_pred, minlength=len(labels)) if len(y_pred) else np.zeros(len(labels), dtype=np.int64)
    top_true = int(np.argmax(true_counts)) if len(true_counts) else 0
    top_pred = int(np.argmax(pred_counts)) if len(pred_counts) else 0
    top_true_fraction = float(true_counts[top_true] / len(y_true)) if len(y_true) else 0.0
    top_fraction = float(pred_counts[top_pred] / len(y_pred)) if len(y_pred) else 0.0
    present_class_count = int(np.count_nonzero(support))
    if present_class_count < 2:
        collapse_allowed_fraction = 1.0
    elif top_pred == top_true:
        collapse_allowed_fraction = max(0.90, min(0.99, top_true_fraction + 0.05))
    else:
        collapse_allowed_fraction = 0.90
    collapse_flag = bool(present_class_count >= 2 and top_fraction > collapse_allowed_fraction)
    stem = f"{method}_{rate_hz}hz_{split_type}"

    with (out_dir / f"{stem}_per_class.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support", "predicted", "present_in_truth"])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow(
                [
                    idx,
                    name,
                    precision[idx],
                    recall[idx],
                    f1[idx],
                    int(support[idx]),
                    int(pred_counts[idx]),
                    bool(support[idx] > 0),
                ]
            )

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
        "macro_f1_present_classes": macro_f1,
        "macro_f1_all_classes": macro_f1_all,
        "present_class_count": present_class_count,
        "top_true_class": CLASS_NAMES[top_true],
        "top_true_fraction": top_true_fraction,
        "top_predicted_class": CLASS_NAMES[top_pred],
        "top_predicted_fraction": top_fraction,
        "collapse_allowed_fraction": collapse_allowed_fraction,
        "collapse_flag": collapse_flag,
    }
    if fail_on_collapse and collapse_flag:
        raise DegeneratePredictionError(
            f"{method} {rate_hz} Hz predicts {CLASS_NAMES[top_pred]} for "
            f"{top_fraction:.1%} of test windows; true top class is "
            f"{CLASS_NAMES[top_true]} at {top_true_fraction:.1%}, allowed "
            f"threshold is {collapse_allowed_fraction:.1%}"
        )
    return report
