"""Full-int8 post-training quantization for the CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes
from ringdata import apply_manifest, load_sessions, resample_windows, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_cnn.train import _arrays, _balanced_train_windows


def quantize_model(
    model_path: Path,
    windows,
    splits: dict,
    rate_hz: int,
    out_path: Path,
    min_retention: float = 0.95,
) -> dict:
    import tensorflow as tf

    out_path.unlink(missing_ok=True)
    out_path.with_suffix(".metrics.json").unlink(missing_ok=True)
    rate_windows = windows if rate_hz == 120 else resample_windows(windows, rate_hz)
    train_w = select_windows(rate_windows, splits["cross_session"]["train"])
    val_w = select_windows(rate_windows, splits["cross_session"]["val"])
    test_w = select_windows(rate_windows, splits["cross_session"]["test"])
    if not train_w or not val_w or not test_w:
        raise ValueError(f"rate {rate_hz}: need non-empty train/validation/test for quantization")
    stats_path = model_path.with_name(model_path.stem + "_standardizer.npz")
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    rep_w = _balanced_train_windows(train_w, seed=20260711 + rate_hz, max_per_class=40)
    x_rep, _ = _arrays(rep_w, mean, std)
    x_val, y_val = _arrays(val_w, mean, std)
    x_test, y_test = _arrays(test_w, mean, std)

    model = tf.keras.models.load_model(model_path)
    float_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    float_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    float_val_f1, *_ = macro_f1_present_classes(y_val, float_val_pred)
    float_f1, *_ = macro_f1_present_classes(y_test, float_pred)

    def representative_dataset():
        for sample in x_rep:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    def int8_predictions(x):
        predictions = []
        for sample in x:
            q = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_detail["index"], q[np.newaxis, ...])
            interpreter.invoke()
            out = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
            predictions.append(int(np.argmax((out - out_zero) * out_scale)))
        return predictions

    int8_val_f1, *_ = macro_f1_present_classes(y_val, int8_predictions(x_val))
    int8_f1, *_ = macro_f1_present_classes(y_test, int8_predictions(x_test))
    gap = float_val_f1 - int8_val_f1
    retention = int8_val_f1 / float_val_f1 if float_val_f1 > 0 else 1.0
    metrics = {
        "acceptance_split": "validation",
        "float_macro_f1": float_val_f1,
        "int8_macro_f1": int8_val_f1,
        "absolute_gap": gap,
        "retention": retention,
        "required_retention": min_retention,
        "accepted": bool(retention >= min_retention),
        "test_float_macro_f1": float_f1,
        "test_int8_macro_f1": int8_f1,
        "path": str(out_path),
        "standardizer": str(stats_path),
    }
    if not metrics["accepted"]:
        raise RuntimeError(
            f"int8 macro-F1 retention {retention:.1%} is below {min_retention:.1%}; "
            "firmware export remains blocked"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    out_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--splits", default="ml/splits_within_user.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rate-hz", type=int, default=120)
    parser.add_argument("--out", default="ml/results/cnn/cnn_model_int8.tflite")
    args = parser.parse_args()

    sessions, manifest_warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in manifest_warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits)
    metrics = quantize_model(Path(args.model), windows, splits, args.rate_hz, Path(args.out))
    print(metrics)


if __name__ == "__main__":
    main()
