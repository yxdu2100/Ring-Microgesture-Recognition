"""Verify exported HDC header data and packed-distance parity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ringdata import apply_manifest, load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows
from train_hdc.encode import (
    encode_window,
    fit_level_bounds,
    hamming,
    make_codebooks,
    pack_bits,
    unpack_bits,
)
from train_hdc.train import _class_bits, train_hdc


def _header_words(text: str, name: str, rows: int) -> np.ndarray:
    match = re.search(rf"{name}\[[^=]*= \{{(.*?)\n\}};", text, re.S)
    if not match:
        raise ValueError(f"missing {name} in generated header")
    values = np.array(
        [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]+)U", match.group(1))],
        dtype=np.uint32,
    )
    return values.reshape(rows, -1)


def _packed_hamming(query_words: np.ndarray, class_words: np.ndarray) -> np.ndarray:
    return np.array(
        [sum((int(q) ^ int(c)).bit_count() for q, c in zip(query_words, row)) for row in class_words],
        dtype=np.int64,
    )


def _firmware_encode_words(raw: np.ndarray, codebooks) -> np.ndarray:
    """Independent word-level translation of ``clf_hdc.c`` for one window."""
    words = codebooks.words
    timestep_words: list[np.ndarray] = []
    for sample in raw.astype(np.int64):
        counts = np.zeros(codebooks.dim, dtype=np.uint8)
        for channel, value in enumerate(sample):
            lo = int(codebooks.level_min[channel])
            hi = int(codebooks.level_max[channel])
            index = ((int(value) - lo) * codebooks.levels.shape[0]) // max(hi - lo, 1)
            index = min(max(index, 0), codebooks.levels.shape[0] - 1)
            bound = np.bitwise_xor(
                pack_bits(codebooks.levels[index]),
                pack_bits(codebooks.channels[channel]),
            )
            counts += unpack_bits(bound, codebooks.dim)
        tie = codebooks.channel_tie
        timestep_words.append(
            pack_bits((2 * counts > 6) | ((2 * counts == 6) & tie), words)
        )

    gram_counts = np.zeros(codebooks.dim, dtype=np.uint16)
    for index in range(2, len(timestep_words)):
        gram = (
            np.roll(timestep_words[index - 2], 2)
            ^ np.roll(timestep_words[index - 1], 1)
            ^ timestep_words[index]
        )
        gram_counts += unpack_bits(gram, codebooks.dim)
    gram_count = len(timestep_words) - 2
    query_bits = (
        (2 * gram_counts > gram_count)
        | ((2 * gram_counts == gram_count) & codebooks.bundle_tie)
    )
    return pack_bits(query_bits, words)


def verify(data_dir: Path, manifest: Path, splits_path: Path, header: Path) -> dict:
    sessions, warnings = apply_manifest(load_sessions(data_dir), manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    windows = segment_sessions(sessions, enforce_perform_window=False)
    windows = [window for window in windows if window.perform_window_overrun_samples <= 0]
    splits = build_or_load_splits(windows, splits_path)
    train_windows = select_windows(windows, splits["cross_session"]["train"])
    test_windows = select_windows(windows, splits["cross_session"]["test"])
    lo, hi = fit_level_bounds(train_windows)
    codebooks = make_codebooks(dim=2048, level_min=lo, level_max=hi)
    memories = train_hdc(train_windows, codebooks)
    class_bits = _class_bits(memories, codebooks)

    text = header.read_text()
    exported_levels = _header_words(text, "hdc_level_hv", codebooks.levels.shape[0])
    exported_channels = _header_words(text, "hdc_channel_hv", codebooks.channels.shape[0])
    exported_classes = _header_words(text, "hdc_class_hv", class_bits.shape[0])
    expected_levels = np.stack([pack_bits(row) for row in codebooks.levels])
    expected_channels = np.stack([pack_bits(row) for row in codebooks.channels])
    expected_classes = np.stack([pack_bits(row) for row in class_bits])
    if not np.array_equal(exported_levels, expected_levels):
        raise AssertionError("exported level codebook differs from Python")
    if not np.array_equal(exported_channels, expected_channels):
        raise AssertionError("exported channel codebook differs from Python")
    if not np.array_equal(exported_classes, expected_classes):
        raise AssertionError("exported class prototypes differ from Python")

    known = test_windows[0]
    query = encode_window(known.raw, codebooks)
    query_words = pack_bits(query)
    firmware_query_words = _firmware_encode_words(known.raw, codebooks)
    if not np.array_equal(query_words, firmware_query_words):
        differing = int(np.count_nonzero(query_words != firmware_query_words))
        raise AssertionError(f"Python/C-style query encoding differs in {differing} words")
    bool_distances = hamming(query, class_bits)
    packed_distances = _packed_hamming(firmware_query_words, exported_classes)
    if not np.array_equal(bool_distances, packed_distances):
        raise AssertionError(
            f"packed Hamming mismatch: Python={bool_distances} packed={packed_distances}"
        )
    return {
        "window_id": known.window_id,
        "distances": bool_distances.tolist(),
        "predicted_class": int(np.argmin(bool_distances)),
        "codebook_words_verified": int(exported_levels.size + exported_channels.size),
        "prototype_words_verified": int(exported_classes.size),
        "query_words_verified": int(query_words.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("ml/dataset_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits_within_user.json"))
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("firmware/src/classifiers/generated/hdc_memories.h"),
    )
    args = parser.parse_args()
    print(verify(args.data_dir, args.manifest, args.splits, args.header))


if __name__ == "__main__":
    main()
