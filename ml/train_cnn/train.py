"""Train the compact CNN on frozen splits."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes, prediction_report
from ringdata import CLASS_NAMES, apply_manifest, load_sessions, resample_windows, segment_sessions
from ringdata.convert import raw_to_physical
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_cnn.model import build_model

SEED = 20260706


def _set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)


def _physical_stack(windows) -> np.ndarray:
    return np.stack([raw_to_physical(w.raw) for w in windows]).astype(np.float32)


def _fit_standardizer(windows) -> tuple[np.ndarray, np.ndarray]:
    x = _physical_stack(windows)
    mean = np.mean(x, axis=(0, 1)).astype(np.float32)
    std = np.std(x, axis=(0, 1)).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def _random_rotation(rng: np.random.Generator, max_deg: float = 10.0) -> np.ndarray:
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float32,
    )


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    for i in range(len(out)):
        r = _random_rotation(rng)
        out[i, :, 0:3] = out[i, :, 0:3] @ r.T
        out[i, :, 3:6] = out[i, :, 3:6] @ r.T
        shift = int(rng.integers(-12, 13))
        shifted = np.zeros_like(out[i])
        if shift > 0:
            shifted[shift:] = out[i, :-shift]
            shifted[:shift] = out[i, :1]
        elif shift < 0:
            shifted[:shift] = out[i, -shift:]
            shifted[shift:] = out[i, -1:]
        else:
            shifted = out[i]
        out[i] = shifted
        sigma = 0.02 * np.std(out[i], axis=0, keepdims=True)
        out[i] += rng.normal(0.0, sigma, size=out[i].shape).astype(np.float32)
    return out


class MacroF1EarlyStopping:
    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, patience: int = 20):
        import tensorflow as tf

        class Callback(tf.keras.callbacks.Callback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def on_epoch_end(self, epoch, logs=None):
                pred = np.argmax(self.model.predict(x_val, verbose=0), axis=1)
                score, *_ = macro_f1_present_classes(y_val, pred, labels=list(range(len(CLASS_NAMES))))
                logs = logs or {}
                logs["val_macro_f1"] = score
                if score > self.outer.best:
                    self.outer.best = score
                    self.outer.wait = 0
                    self.outer.weights = self.model.get_weights()
                else:
                    self.outer.wait += 1
                    if self.outer.wait >= patience:
                        self.model.stop_training = True
                        if self.outer.weights is not None:
                            self.model.set_weights(self.outer.weights)

            def on_train_end(self, logs=None):
                if self.outer.weights is not None:
                    self.model.set_weights(self.outer.weights)

        self.best = -1.0
        self.wait = 0
        self.weights = None
        self.callback = Callback(self)


def _arrays(windows, mean: np.ndarray, std: np.ndarray):
    return _standardize(_physical_stack(windows), mean, std), np.array([w.class_id for w in windows], dtype=np.int64)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float32)
    present = counts > 0
    weights = np.ones_like(counts, dtype=np.float32)
    if np.any(present):
        weights[present] = float(np.sum(counts[present])) / (float(np.count_nonzero(present)) * counts[present])
    return weights[y].astype(np.float32)


def _balanced_train_windows(windows, seed: int, max_per_class: int | None = None):
    """Return a deterministic class- and session-shuffled calibration subset."""
    by_class: dict[int, list] = {}
    for window in windows:
        by_class.setdefault(window.class_id, []).append(window)
    if not by_class:
        return []
    target = min(len(group) for group in by_class.values())
    if max_per_class is not None:
        target = min(target, max_per_class)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in sorted(by_class):
        group = sorted(
            by_class[class_id],
            key=lambda window: (window.session_id, window.start_sample_id, window.window_id),
        )
        indices = rng.permutation(len(group))[:target]
        selected.extend(group[int(index)] for index in indices)
    rng.shuffle(selected)
    return selected


def _validation_from_train(train_w: list, seed: int) -> tuple[list, list]:
    by_class = {}
    for window in train_w:
        by_class.setdefault(window.class_id, []).append(window)
    val_ids = set()
    rng = random.Random(seed)
    for class_windows in by_class.values():
        class_windows = sorted(class_windows, key=lambda w: w.window_id)
        rng.shuffle(class_windows)
        take = max(1, round(0.2 * len(class_windows)))
        val_ids.update(w.window_id for w in class_windows[:take])
    val_w = [w for w in train_w if w.window_id in val_ids]
    new_train_w = [w for w in train_w if w.window_id not in val_ids]
    return new_train_w, val_w


def train_one_rate(
    windows,
    splits: dict,
    rate_hz: int,
    out_dir: Path,
    split_key: str = "cross_session",
    report_split_type: str | None = None,
    return_predictions: bool = False,
) -> dict:
    import tensorflow as tf

    _set_seeds()
    report_split_type = report_split_type or split_key
    rate_windows = windows if rate_hz == 120 else resample_windows(windows, rate_hz)
    split = splits[split_key]
    train_w = select_windows(rate_windows, split["train"])
    val_w = select_windows(rate_windows, split["val"])
    test_w = select_windows(rate_windows, split["test"])
    val_labels = {w.class_id for w in val_w}
    train_labels = {w.class_id for w in train_w}
    if not val_w or not train_labels.issubset(val_labels):
        train_w, val_w = _validation_from_train(train_w, SEED + rate_hz)
    if not train_w or not val_w or not test_w:
        raise ValueError(
            f"rate {rate_hz}: need train/val/test windows; "
            f"got {len(train_w)}/{len(val_w)}/{len(test_w)}"
        )

    mean, std = _fit_standardizer(train_w)
    x_train, y_train = _arrays(train_w, mean, std)
    x_val, y_val = _arrays(val_w, mean, std)
    x_test, y_test = _arrays(test_w, mean, std)
    sample_weight = _balanced_sample_weights(y_train)
    rng = np.random.default_rng(SEED + rate_hz)

    model = build_model(window_samples=x_train.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")

    class AugmentSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.indices = np.arange(len(x_train), dtype=np.int64)
            rng.shuffle(self.indices)

        def __len__(self):
            return max(1, int(np.ceil(len(x_train) / 32)))

        def __getitem__(self, idx):
            batch_indices = self.indices[idx * 32 : min(len(x_train), (idx + 1) * 32)]
            return (
                augment_batch(x_train[batch_indices], rng),
                y_train[batch_indices],
                sample_weight[batch_indices],
            )

        def on_epoch_end(self):
            rng.shuffle(self.indices)

    stopper = MacroF1EarlyStopping(x_val, y_val, patience=20)
    model.fit(AugmentSequence(), epochs=200, verbose=0, callbacks=[stopper.callback])
    pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = prediction_report(y_test, pred, "cnn_float", rate_hz, report_split_type, out_dir, fail_on_collapse=False)
    macro_f1 = report["macro_f1"]

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"cnn_{rate_hz}hz.keras"
    model.save(model_path)
    np.savez(out_dir / f"cnn_{rate_hz}hz_standardizer.npz", mean=mean, std=std)
    row = {
        "method": "cnn_float",
        "rate_hz": rate_hz,
        "split_type": report_split_type,
        "macro_f1": macro_f1,
        "macro_f1_all_classes": report["macro_f1_all_classes"],
        "present_class_count": report["present_class_count"],
        "top_true_class": report["top_true_class"],
        "top_true_fraction": report["top_true_fraction"],
        "top_predicted_class": report["top_predicted_class"],
        "top_predicted_fraction": report["top_predicted_fraction"],
        "collapse_allowed_fraction": report["collapse_allowed_fraction"],
        "collapse_flag": report["collapse_flag"],
        "model": str(model_path),
    }
    if return_predictions:
        row["_y_true"] = y_test
        row["_y_pred"] = pred
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    parser.add_argument("--out-dir", default="ml/results/cnn")
    args = parser.parse_args()

    _set_seeds()
    sessions, manifest_warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in manifest_warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits)
    assert_no_cross_session_leakage(splits)
    rows = []
    for rate in (120, 60, 30):
        rows.append(train_one_rate(windows, splits, rate, Path(args.out_dir)))
        print(f"cnn {rate}hz macro_f1={rows[-1]['macro_f1']:.4f}")
    with (Path(args.out_dir) / "cnn_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "rate_hz",
                "split_type",
                "macro_f1",
                "macro_f1_all_classes",
                "present_class_count",
                "top_true_class",
                "top_true_fraction",
                "top_predicted_class",
                "top_predicted_fraction",
                "collapse_allowed_fraction",
                "collapse_flag",
                "model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
