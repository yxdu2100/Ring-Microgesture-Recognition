"""Export HDC codebooks and class vectors to firmware header format."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from ringdata import load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_hdc.encode import HDC_LEVEL_COUNT, fit_level_bounds, pack_bits, make_codebooks
from train_hdc.train import DEFAULT_ENCODING_MODE, _class_bits, train_hdc


def _git_hash(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def _write_array(f, name: str, rows: np.ndarray) -> None:
    f.write(f"static const uint32_t {name}[{rows.shape[0]}][HDC_DIM_WORDS] = {{\n")
    for row in rows:
        words = pack_bits(row)
        f.write("\t{ ")
        f.write(", ".join(f"0x{int(w):08x}U" for w in words))
        f.write(" },\n")
    f.write("};\n\n")


def export_header(windows, splits: dict, out_path: Path, dim: int = 2048, mode: str = DEFAULT_ENCODING_MODE) -> None:
    if dim != 2048:
        raise ValueError("firmware/src/classifiers/clf_hdc.c currently asserts HDC dimension is 2048")
    codebooks = make_codebooks(dim=dim)
    train_w = select_windows(windows, splits["cross_session"]["train"])
    if not train_w:
        raise ValueError("no train windows available for HDC export")
    lo, hi = fit_level_bounds(train_w)
    codebooks = make_codebooks(dim=dim, level_min=lo, level_max=hi)
    memories = train_hdc(train_w, codebooks, mode=mode)
    class_bits = _class_bits(memories, codebooks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("#ifndef GENERATED_HDC_MEMORIES_H\n#define GENERATED_HDC_MEMORIES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// data_git_hash: {_git_hash(Path('.'))}\n")
        f.write(f"// hdc_encoding_mode: {mode}\n")
        f.write("// word order matches clf_hdc.c: bit 0 is LSB, local permutation rotates whole words forward.\n")
        f.write(f"#define HDC_DIM_WORDS {dim // 32}U\n")
        f.write(f"#define HDC_LEVEL_COUNT {HDC_LEVEL_COUNT}U\n")
        f.write("#define HDC_CHANNEL_COUNT 6U\n")
        f.write("#define HDC_CLASS_VECTOR_COUNT 5U\n\n")
        f.write("static const int16_t hdc_level_min[HDC_CHANNEL_COUNT] = { ")
        f.write(", ".join(f"{int(round(x))}" for x in codebooks.level_min))
        f.write(" };\n")
        f.write("static const int16_t hdc_level_max[HDC_CHANNEL_COUNT] = { ")
        f.write(", ".join(f"{int(round(x))}" for x in codebooks.level_max))
        f.write(" };\n\n")
        _write_array(f, "hdc_level_hv", codebooks.levels)
        _write_array(f, "hdc_channel_hv", codebooks.channels)
        _write_array(f, "hdc_channel_tie_hv", codebooks.channel_tie.reshape(1, -1))
        _write_array(f, "hdc_bundle_tie_hv", codebooks.bundle_tie.reshape(1, -1))
        _write_array(f, "hdc_class_hv", class_bits)
        f.write("#endif\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--out", default="firmware/src/classifiers/generated/hdc_memories.h")
    parser.add_argument("--mode", default=DEFAULT_ENCODING_MODE, choices=["absolute", "bag", "ngram"])
    parser.add_argument("--drop-invalid-windows", action="store_true")
    args = parser.parse_args()
    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, enforce_perform_window=not args.drop_invalid_windows)
    if args.drop_invalid_windows:
        windows = [w for w in windows if w.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, args.splits)
    export_header(windows, splits, Path(args.out), mode=args.mode)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
