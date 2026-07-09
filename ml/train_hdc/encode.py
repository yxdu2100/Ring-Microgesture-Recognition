"""BSC/MAP-style binary HDC encoder matching firmware bit packing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torchhd  # noqa: F401
except Exception:
    torchhd = None

HDC_LEVEL_COUNT = 32
HDC_CHANNEL_COUNT = 6


@dataclass
class Codebooks:
    dim: int
    levels: np.ndarray
    channels: np.ndarray
    level_min: np.ndarray
    level_max: np.ndarray

    @property
    def words(self) -> int:
        return self.dim // 32


def make_codebooks(
    dim: int = 2048,
    levels: int = HDC_LEVEL_COUNT,
    seed: int = 20260706,
    level_min: np.ndarray | None = None,
    level_max: np.ndarray | None = None,
) -> Codebooks:
    if dim % 32 != 0:
        raise ValueError("HDC dimension must be a multiple of 32 for firmware packing")
    rng = np.random.default_rng(seed + dim)
    base = rng.integers(0, 2, size=dim, dtype=np.uint8).astype(np.bool_)
    order = rng.permutation(dim)
    level_hv = np.empty((levels, dim), dtype=np.bool_)
    for level in range(levels):
        n_on = int(round((level / max(1, levels - 1)) * dim))
        hv = base.copy()
        hv[order[:n_on]] = ~hv[order[:n_on]]
        level_hv[level] = hv
    channel_hv = rng.integers(0, 2, size=(HDC_CHANNEL_COUNT, dim), dtype=np.uint8).astype(np.bool_)
    if level_min is None:
        level_min = np.full(HDC_CHANNEL_COUNT, -32768, dtype=np.float32)
    if level_max is None:
        level_max = np.full(HDC_CHANNEL_COUNT, 32767, dtype=np.float32)
    return Codebooks(
        dim=dim,
        levels=level_hv,
        channels=channel_hv,
        level_min=np.asarray(level_min, dtype=np.float32),
        level_max=np.asarray(level_max, dtype=np.float32),
    )


def fit_level_bounds(windows) -> tuple[np.ndarray, np.ndarray]:
    gesture = [w.raw.astype(np.float32).reshape(-1, HDC_CHANNEL_COUNT) for w in windows if w.label != "null"]
    if not gesture:
        raise ValueError("HDC level bounds require at least one gesture window")
    x = np.concatenate(gesture, axis=0)
    lo = np.percentile(x, 1, axis=0).astype(np.float32)
    hi = np.percentile(x, 99, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1.0)
    return lo, hi


def level_indices(raw: np.ndarray, codebooks: Codebooks) -> np.ndarray:
    level_count = codebooks.levels.shape[0]
    lo = codebooks.level_min.reshape(1, -1)
    hi = codebooks.level_max.reshape(1, -1)
    scaled = (raw.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-6)
    idx = np.floor(scaled * level_count)
    return np.clip(idx, 0, level_count - 1).astype(np.int16)


def encode_window(raw: np.ndarray, codebooks: Codebooks) -> np.ndarray:
    idx = level_indices(raw, codebooks)
    words = codebooks.words
    window_counts = np.zeros(codebooks.dim, dtype=np.uint16)
    for timestep in range(raw.shape[0]):
        bound = np.logical_xor(codebooks.levels[idx[timestep]], codebooks.channels)
        timestep_bits = np.count_nonzero(bound, axis=0) > (HDC_CHANNEL_COUNT // 2)
        word_bits = timestep_bits.reshape(words, 32)
        rotated_bits = np.roll(word_bits, timestep % words, axis=0).reshape(codebooks.dim)
        window_counts += rotated_bits
    return window_counts > (raw.shape[0] // 2)


def pack_bits(bits: np.ndarray, words: int | None = None) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.bool_)
    if words is None:
        words = int(np.ceil(bits.size / 32))
    padded = np.zeros(words * 32, dtype=np.uint8)
    padded[: bits.size] = bits.astype(np.uint8)
    packed = np.packbits(padded.reshape(words, 32), axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u4").reshape(words).astype(np.uint32)


def unpack_bits(words: np.ndarray, dim: int) -> np.ndarray:
    byte_view = np.asarray(words, dtype="<u4").view(np.uint8).reshape(-1, 4)
    bits = np.unpackbits(byte_view, axis=1, bitorder="little").reshape(-1)
    return bits[:dim].astype(np.bool_)


def hamming(query: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.logical_xor(classes, query), axis=1)
