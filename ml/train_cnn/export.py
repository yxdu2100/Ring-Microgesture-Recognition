"""Export a TFLite model as a C header."""

from __future__ import annotations

import argparse
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


def export_tflite_header(tflite_path: Path, out_path: Path, rate_hz: int, metrics: str = "", stats_path: Path | None = None) -> None:
    import numpy as np

    data = tflite_path.read_bytes()
    if stats_path is None:
        stats_path = tflite_path.with_name(tflite_path.name.replace("_int8.tflite", "_standardizer.npz"))
    if stats_path.exists():
        stats = np.load(stats_path)
        mean = stats["mean"].astype(float)
        std = stats["std"].astype(float)
    else:
        mean = np.zeros(6, dtype=float)
        std = np.ones(6, dtype=float)
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
    args = parser.parse_args()
    export_tflite_header(Path(args.tflite), Path(args.out), args.rate_hz, args.metrics)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
