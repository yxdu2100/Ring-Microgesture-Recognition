"""Export a TFLite model as a C header."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _git_hash(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def export_tflite_header(
    tflite_path: Path,
    out_path: Path,
    rate_hz: int,
    metrics: str = "",
    stats_path: Path | None = None,
    require_accepted_metrics: bool = True,
) -> None:
    import numpy as np

    metrics_path = tflite_path.with_suffix(".metrics.json")
    quant_metrics = {}
    if require_accepted_metrics:
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing PTQ acceptance record: {metrics_path}")
        quant_metrics = json.loads(metrics_path.read_text())
        if not quant_metrics.get("accepted", False):
            raise RuntimeError(f"PTQ acceptance record rejected export: {metrics_path}")
    data = tflite_path.read_bytes()
    if stats_path is None:
        recorded = quant_metrics.get("standardizer")
        stats_path = Path(recorded) if recorded else tflite_path.with_name(
            tflite_path.name.replace("_int8.tflite", "_standardizer.npz")
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"missing train-fold standardizer {stats_path}; refusing a zero-mean/unit-std export"
        )
    stats = np.load(stats_path)
    mean = stats["mean"].astype(float)
    std = stats["std"].astype(float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("#ifndef GENERATED_CNN_MODEL_H\n#define GENERATED_CNN_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// data_git_hash: {_git_hash(Path('.'))}\n")
        f.write(f"// rate_hz: {rate_hz}\n")
        if metrics:
            f.write(f"// metrics: {metrics}\n")
        f.write("static const float g_cnn_input_mean[6] = { ")
        f.write(", ".join(f"{x:.9g}f" for x in mean))
        f.write(" };\n")
        f.write("static const float g_cnn_input_std[6] = { ")
        f.write(", ".join(f"{x:.9g}f" for x in std))
        f.write(" };\n\n")
        f.write(f"static const uint8_t g_cnn_model[{len(data)}] = {{\n")
        for i in range(0, len(data), 12):
            chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 12])
            f.write(f"\t{chunk},\n")
        f.write("};\n")
        f.write(f"static const unsigned int g_cnn_model_len = {len(data)}U;\n\n")
        f.write("#endif\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--out", default="firmware/src/classifiers/generated/cnn_model.h")
    parser.add_argument("--rate-hz", type=int, default=120)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--stats", type=Path, default=None,
                        help="Train-fold cnn_*_standardizer.npz (normally read from PTQ metrics)")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    export_tflite_header(
        Path(args.tflite),
        Path(args.out),
        args.rate_hz,
        args.metrics,
        stats_path=args.stats,
        require_accepted_metrics=not args.allow_unverified,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
