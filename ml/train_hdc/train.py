"""Train and sweep HDC classifiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_utils import prediction_report
from ringdata import CLASS_NAMES, load_sessions, resample_windows, segment_sessions
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_hdc.encode import Codebooks, encode_window, fit_level_bounds, hamming, make_codebooks

SEED = 20260706


def _signed(bits: np.ndarray) -> np.ndarray:
    return np.where(bits, 1, -1).astype(np.int16)


def _balanced_windows(windows, seed: int):
    by_class = {}
    for window in windows:
        by_class.setdefault(window.class_id, []).append(window)
    if len(by_class) < len(CLASS_NAMES):
        return windows
    n = min(len(v) for v in by_class.values())
    rng = np.random.default_rng(seed)
    selected = []
    for class_windows in by_class.values():
        ordered = sorted(class_windows, key=lambda w: w.window_id)
        idx = rng.permutation(len(ordered))[:n]
        selected.extend(ordered[int(i)] for i in idx)
    return sorted(selected, key=lambda w: w.window_id)


def train_hdc(train_w, codebooks: Codebooks, epochs: int = 5) -> np.ndarray:
    memories = np.zeros((len(CLASS_NAMES), codebooks.dim), dtype=np.int32)
    encoded = []
    labels = []
    balanced = _balanced_windows(train_w, SEED + codebooks.dim)
    for window in balanced:
        q = encode_window(window.raw, codebooks)
        encoded.append(q)
        labels.append(window.class_id)
        memories[window.class_id] += _signed(q)

    for _ in range(epochs):
        class_bits = memories > 0
        for q, y in zip(encoded, labels):
            pred = int(np.argmin(hamming(q, class_bits)))
            if pred != y:
                s = _signed(q)
                memories[y] += s
                memories[pred] -= s
                class_bits = memories > 0
    return memories


def predict_hdc(windows, memories: np.ndarray, codebooks: Codebooks) -> tuple[np.ndarray, np.ndarray]:
    class_bits = memories > 0
    y_true = []
    y_pred = []
    for window in windows:
        q = encode_window(window.raw, codebooks)
        y_true.append(window.class_id)
        y_pred.append(int(np.argmin(hamming(q, class_bits))))
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def evaluate_hdc(windows, memories: np.ndarray, codebooks: Codebooks, rate_hz: int, dim: int, out_dir: Path) -> tuple[float, dict]:
    y_true, y_pred = predict_hdc(windows, memories, codebooks)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    report = prediction_report(y_true, y_pred, f"hdc_D{dim}", rate_hz, "cross_session", out_dir, fail_on_collapse=False)
    return acc, report


def sweep(windows, splits: dict, out_csv: Path) -> list[dict]:
    rows = []
    for rate in (120, 60, 30):
        rate_windows = windows if rate == 120 else resample_windows(windows, rate)
        train_w = select_windows(rate_windows, splits["cross_session"]["train"])
        lo, hi = fit_level_bounds(train_w)
        for dim in (1024, 2048, 4096):
            codebooks = make_codebooks(dim=dim, seed=SEED, level_min=lo, level_max=hi)
            test_w = select_windows(rate_windows, splits["cross_session"]["test"])
            if not train_w or not test_w:
                raise ValueError(f"HDC sweep rate {rate} D {dim}: empty train/test split")
            memories = train_hdc(train_w, codebooks)
            acc, report = evaluate_hdc(test_w, memories, codebooks, rate, dim, out_csv.parent)
            rows.append(
                {
                    "method": "hdc",
                    "rate_hz": rate,
                    "split_type": "cross_session",
                    "dim": dim,
                    "accuracy": acc,
                    "macro_f1": report["macro_f1"],
                    "top_predicted_class": report["top_predicted_class"],
                    "top_predicted_fraction": report["top_predicted_fraction"],
                    "memory_bytes": (dim // 8) * (len(CLASS_NAMES) + HDC_EXPORT_CODEBOOK_FACTOR),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


HDC_EXPORT_CODEBOOK_FACTOR = 32 + 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--out", default="ml/results/hdc/hdc_grid.csv")
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions)
    splits = build_or_load_splits(windows, args.splits, seed=SEED)
    assert_no_cross_session_leakage(splits)
    rows = sweep(windows, splits, Path(args.out))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
