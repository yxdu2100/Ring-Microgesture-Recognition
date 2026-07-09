"""Feature set mirrored for MEMS Studio and the software tree baseline."""

from __future__ import annotations

import numpy as np

from ringdata.convert import raw_to_physical
from ringdata.segment import Window

CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]


def _zero_crossings(x: np.ndarray) -> float:
    centered = x - np.mean(x)
    signs = np.signbit(centered)
    return float(np.count_nonzero(signs[1:] != signs[:-1]))


def feature_vector(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = raw_to_physical(raw)
    accel_norm = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_norm = np.linalg.norm(x[:, 3:6], axis=1)
    series = {name: x[:, idx] for idx, name in enumerate(CHANNEL_NAMES)}
    series["accel_norm"] = accel_norm
    series["gyro_norm"] = gyro_norm

    values: list[float] = []
    names: list[str] = []
    for name, y in series.items():
        values.extend(
            [
                float(np.mean(y)),
                float(np.var(y)),
                float(np.mean(y * y)),
                float(np.ptp(y)),
                _zero_crossings(y),
            ]
        )
        names.extend(
            [
                f"{name}_mean",
                f"{name}_variance",
                f"{name}_energy",
                f"{name}_peak_to_peak",
                f"{name}_zero_crossings",
            ]
        )
    return np.array(values, dtype=np.float32), names


def featurize(windows: list[Window]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    labels = []
    names: list[str] | None = None
    for window in windows:
        row, row_names = feature_vector(window.raw)
        rows.append(row)
        labels.append(window.class_id)
        if names is None:
            names = row_names
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), names or []
    return np.vstack(rows), np.array(labels, dtype=np.int64), names or []
