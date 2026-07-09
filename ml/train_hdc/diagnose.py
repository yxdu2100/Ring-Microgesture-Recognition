"""Diagnostics for isolating HDC encoder failures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes
from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_hdc.encode import (
    HDC_CHANNEL_COUNT,
    fit_level_bounds,
    hamming,
    level_indices,
    make_codebooks,
    timestep_vector,
    encode_window,
)
from train_hdc.train import _balanced_windows, train_hdc, predict_hdc

SEED = 20260706


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _evaluate_mode(train_w, test_w, codebooks, mode: str) -> dict:
    memories = train_hdc(train_w, codebooks, mode=mode)
    train_true, train_pred = predict_hdc(train_w, memories, codebooks, mode=mode)
    balanced_train_w = _balanced_windows(train_w, SEED + codebooks.dim)
    bal_true, bal_pred = predict_hdc(balanced_train_w, memories, codebooks, mode=mode)
    test_true, test_pred = predict_hdc(test_w, memories, codebooks, mode=mode)
    train_macro, train_macro_all, *_ = macro_f1_present_classes(train_true, train_pred)
    bal_macro, bal_macro_all, *_ = macro_f1_present_classes(bal_true, bal_pred)
    test_macro, test_macro_all, *_ = macro_f1_present_classes(test_true, test_pred)
    return {
        "mode": mode,
        "train_accuracy": _accuracy(train_true, train_pred),
        "train_macro_f1": train_macro,
        "train_macro_f1_all_classes": train_macro_all,
        "balanced_train_accuracy": _accuracy(bal_true, bal_pred),
        "balanced_train_macro_f1": bal_macro,
        "balanced_train_macro_f1_all_classes": bal_macro_all,
        "test_accuracy": _accuracy(test_true, test_pred),
        "test_macro_f1": test_macro,
        "test_macro_f1_all_classes": test_macro_all,
    }


def _similarity_spread(windows, codebooks, mode: str, max_windows: int) -> dict:
    rng = np.random.default_rng(SEED + codebooks.dim)
    selected = list(windows)
    if len(selected) > max_windows:
        idx = rng.choice(len(selected), size=max_windows, replace=False)
        selected = [selected[int(i)] for i in idx]
    encoded = np.stack([encode_window(w.raw, codebooks, mode=mode) for w in selected])
    labels = np.array([w.class_id for w in selected], dtype=np.int16)
    within = []
    between = []
    for i in range(len(selected)):
        distances = hamming(encoded[i], encoded[i + 1 :]) / codebooks.dim
        same = labels[i + 1 :] == labels[i]
        within.extend(distances[same].tolist())
        between.extend(distances[~same].tolist())
    return {
        "mode": mode,
        "pairwise_windows": len(selected),
        "within_mean": float(np.mean(within)) if within else float("nan"),
        "within_std": float(np.std(within)) if within else float("nan"),
        "between_mean": float(np.mean(between)) if between else float("nan"),
        "between_std": float(np.std(between)) if between else float("nan"),
    }


def _codebook_report(train_w, codebooks) -> dict:
    adjacent = np.count_nonzero(np.logical_xor(codebooks.levels[:-1], codebooks.levels[1:]), axis=1) / codebooks.dim
    far = np.count_nonzero(np.logical_xor(codebooks.levels[0], codebooks.levels[-1])) / codebooks.dim
    midpoint = ((codebooks.level_min + codebooks.level_max) / 2.0).reshape(1, HDC_CHANNEL_COUNT)
    mid_indices = level_indices(midpoint, codebooks)[0]
    tie_count = 0
    bit_count = 0
    for window in train_w:
        for sample in window.raw:
            idx = level_indices(sample.reshape(1, -1), codebooks)[0]
            bound = np.logical_xor(codebooks.levels[idx], codebooks.channels)
            counts = np.count_nonzero(bound, axis=0)
            tie_count += int(np.count_nonzero(counts == (HDC_CHANNEL_COUNT // 2)))
            bit_count += codebooks.dim
            timestep_vector(sample, codebooks)
    return {
        "adjacent_level_hamming_mean": float(np.mean(adjacent)),
        "adjacent_level_hamming_min": float(np.min(adjacent)),
        "adjacent_level_hamming_max": float(np.max(adjacent)),
        "endpoint_level_hamming": float(far),
        "midpoint_level_indices": " ".join(str(int(x)) for x in mid_indices),
        "channel_tie_bit_fraction": float(tie_count / bit_count) if bit_count else 0.0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, eval_rows: list[dict], spread_rows: list[dict], codebook: dict) -> None:
    with path.open("w") as f:
        f.write("# HDC Diagnostics\n\n")
        f.write("## Memorization And Ablation\n\n")
        f.write("| mode | balanced train acc | train acc | test acc | test macro_f1 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in eval_rows:
            f.write(
                f"| {row['mode']} | {row['balanced_train_accuracy']:.4f} | "
                f"{row['train_accuracy']:.4f} | {row['test_accuracy']:.4f} | "
                f"{row['test_macro_f1']:.4f} |\n"
            )
        f.write("\n## Similarity Spread\n\n")
        f.write("| mode | within mean | between mean | windows |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in spread_rows:
            f.write(
                f"| {row['mode']} | {row['within_mean']:.4f} | "
                f"{row['between_mean']:.4f} | {row['pairwise_windows']} |\n"
            )
        f.write("\n## Codebook\n\n")
        for key, value in codebook.items():
            f.write(f"- `{key}`: {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--out-dir", default="ml/results/hdc")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--max-pairwise-windows", type=int, default=400)
    parser.add_argument("--drop-invalid-windows", action="store_true")
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not args.drop_invalid_windows)
    if args.drop_invalid_windows:
        windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits, seed=SEED)
    train_w = select_windows(windows, splits["cross_session"]["train"])
    test_w = select_windows(windows, splits["cross_session"]["test"])
    lo, hi = fit_level_bounds(train_w)
    codebooks = make_codebooks(dim=args.dim, seed=SEED, level_min=lo, level_max=hi)

    modes = ["absolute", "bag", "ngram"]
    eval_rows = [_evaluate_mode(train_w, test_w, codebooks, mode) for mode in modes]
    spread_rows = [_similarity_spread(_balanced_windows(train_w, SEED + args.dim), codebooks, mode, args.max_pairwise_windows) for mode in modes]
    codebook = _codebook_report(train_w, codebooks)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "hdc_diagnostics_eval.csv", eval_rows)
    _write_csv(out_dir / "hdc_diagnostics_similarity.csv", spread_rows)
    _write_csv(out_dir / "hdc_diagnostics_codebook.csv", [codebook])
    _write_markdown(out_dir / "hdc_diagnostics.md", eval_rows, spread_rows, codebook)
    print(f"wrote HDC diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
