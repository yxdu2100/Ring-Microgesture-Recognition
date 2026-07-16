"""Small regression checks for MEMS Studio feature-name compatibility."""

from __future__ import annotations

import numpy as np

from train_mlc.features import mlc_feature_value


def test_abs_energy_matches_energy() -> None:
    raw = np.zeros((128, 6), dtype=np.int16)
    raw[:, 3] = np.arange(-64, 64, dtype=np.int16)
    energy = mlc_feature_value(raw, "ENERGY", "GYR_X")
    abs_energy = mlc_feature_value(raw, "ABS_ENERGY", "GYR_X")
    assert energy == abs_energy
