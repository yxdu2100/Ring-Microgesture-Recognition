"""Full-int8 post-training quantization for the CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ringdata import CLASS_NAMES, load_sessions, resample_windows, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_cnn.train import _arrays, _balanced_train_windows


def quantize_model(model_path: Path, windows, splits: dict, rate_hz: int, out_path: Path, max_gap: float = 0.01) -> dict:
    import tensorflow as tf

    rate_windows = windows if rate_hz == 120 else resample_windows(windows, rate_hz)
    train_w = select_windows(rate_windows, splits["cross_session"]["train"])
    test_w = select_windows(rate_windows, splits["cross_session"]["test"])
    if not train_w or not test_w:
        raise ValueError(f"rate {rate_hz}: need non-empty train/test for quantization")
    stats_path = model_path.with_name(model_path.stem + "_standardizer.npz")
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    rep_w = _balanced_train_windows(train_w, seed=20260706 + rate_hz)[:200]
    x_rep, _ = _arrays(rep_w, mean, std)
    x_test, y_test = _arrays(test_w, mean, std)

    model = tf.keras.models.load_model(model_path)
    float_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    float_f1 = float(f1_score(y_test, float_pred, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0))

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    preds = []
    for sample in x_test:
        q = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], q[np.newaxis, ...])
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        preds.append(int(np.argmax((out - out_zero) * out_scale)))
    int8_f1 = float(f1_score(y_test, preds, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0))
    gap = float_f1 - int8_f1
    if gap > max_gap:
        raise RuntimeError(f"int8 macro-F1 gap {gap:.4f} exceeds {max_gap:.4f}")
    return {"float_macro_f1": float_f1, "int8_macro_f1": int8_f1, "gap": gap, "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rate-hz", type=int, default=120)
    parser.add_argument("--out", default="ml/results/cnn/cnn_model_int8.tflite")
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions)
    splits = build_or_load_splits(windows, args.splits)
    metrics = quantize_model(Path(args.model), windows, splits, args.rate_hz, Path(args.out))
    print(metrics)


if __name__ == "__main__":
    main()
