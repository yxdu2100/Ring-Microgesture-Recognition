"""Diagnostic HDC encoder over the software-tree feature representation.

This module is deliberately Python-only. It reuses the exact 40-value feature
vector consumed by ``ml/train_mlc/tree.py`` and fits quantization bounds on the
training fold only. Bounds remain float32: rounding features such as acceleration
means and variances to integers would collapse useful near-zero distinctions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_hdc.encode import HDC_LEVEL_COUNT, majority_bits
from train_mlc.features import feature_vector, featurize

FEATURE_ENCODER_SEED_OFFSET = 40001


@dataclass
class FeatureCodebooks:
    dim: int
    levels: np.ndarray
    feature_ids: np.ndarray
    bundle_tie: np.ndarray
    level_min: np.ndarray
    level_max: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def words(self) -> int:
        return self.dim // 32


def fit_feature_bounds(train_windows) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Fit 1st/99th percentile bounds using training windows only."""
    features, _, names = featurize(train_windows)
    if not len(features):
        raise ValueError("feature HDC requires non-empty training windows")
    lo = np.percentile(features, 1, axis=0).astype(np.float32)
    hi = np.percentile(features, 99, axis=0).astype(np.float32)
    minimum_span = np.maximum(1e-6, np.maximum(np.abs(lo), np.abs(hi)) * 1e-6)
    hi = np.where(hi > lo, hi, lo + minimum_span).astype(np.float32)
    return lo, hi, tuple(names)


def make_feature_codebooks(
    feature_names: tuple[str, ...],
    level_min: np.ndarray,
    level_max: np.ndarray,
    dim: int = 2048,
    level_count: int = HDC_LEVEL_COUNT,
    seed: int = 20260706,
) -> FeatureCodebooks:
    if dim % 32 != 0:
        raise ValueError("feature-HDC dimension must be a multiple of 32")
    if len(feature_names) != len(level_min) or len(feature_names) != len(level_max):
        raise ValueError("feature names and quantization bounds differ in length")
    rng = np.random.default_rng(seed + dim + FEATURE_ENCODER_SEED_OFFSET)
    base = rng.integers(0, 2, size=dim, dtype=np.uint8).astype(np.bool_)
    order = rng.permutation(dim)
    levels = np.empty((level_count, dim), dtype=np.bool_)
    for level in range(level_count):
        flips = int(round((level / max(1, level_count - 1)) * dim))
        vector = base.copy()
        vector[order[:flips]] = ~vector[order[:flips]]
        levels[level] = vector
    feature_ids = rng.integers(
        0,
        2,
        size=(len(feature_names), dim),
        dtype=np.uint8,
    ).astype(np.bool_)
    bundle_tie = rng.integers(0, 2, size=dim, dtype=np.uint8).astype(np.bool_)
    return FeatureCodebooks(
        dim=dim,
        levels=levels,
        feature_ids=feature_ids,
        bundle_tie=bundle_tie,
        level_min=np.asarray(level_min, dtype=np.float32),
        level_max=np.asarray(level_max, dtype=np.float32),
        feature_names=feature_names,
    )


def fit_feature_codebooks(
    train_windows,
    dim: int = 2048,
    level_count: int = HDC_LEVEL_COUNT,
    seed: int = 20260706,
) -> FeatureCodebooks:
    lo, hi, names = fit_feature_bounds(train_windows)
    return make_feature_codebooks(names, lo, hi, dim=dim, level_count=level_count, seed=seed)


def feature_level_indices(features: np.ndarray, codebooks: FeatureCodebooks) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape != codebooks.level_min.shape:
        raise ValueError(
            f"feature shape {features.shape} != expected {codebooks.level_min.shape}"
        )
    span = np.maximum(codebooks.level_max - codebooks.level_min, 1e-12)
    scaled = (features - codebooks.level_min) / span
    indices = np.floor(scaled * len(codebooks.levels))
    return np.clip(indices, 0, len(codebooks.levels) - 1).astype(np.int16)


def encode_feature_vector(features: np.ndarray, codebooks: FeatureCodebooks) -> np.ndarray:
    indices = feature_level_indices(features, codebooks)
    bound = np.logical_xor(codebooks.feature_ids, codebooks.levels[indices])
    counts = np.count_nonzero(bound, axis=0)
    return majority_bits(counts, len(indices), codebooks.bundle_tie)


def encode_feature_window(raw: np.ndarray, codebooks: FeatureCodebooks) -> np.ndarray:
    features, names = feature_vector(raw)
    if tuple(names) != codebooks.feature_names:
        raise ValueError("tree feature order changed after fitting feature-HDC codebooks")
    return encode_feature_vector(features, codebooks)
