"""Convert LSM6DSV16X raw int16 samples to physical units."""

from __future__ import annotations

import numpy as np

ACCEL_G_PER_LSB = 8.0 / 32768.0
GYRO_DPS_PER_LSB = 2000.0 / 32768.0


def raw_to_physical(raw: np.ndarray) -> np.ndarray:
    """Return float samples as [ax, ay, az] in g and [gx, gy, gz] in dps."""
    arr = np.asarray(raw, dtype=np.float32)
    out = arr.astype(np.float32, copy=True)
    out[..., 0:3] *= ACCEL_G_PER_LSB
    out[..., 3:6] *= GYRO_DPS_PER_LSB
    return out


def physical_to_raw(physical: np.ndarray) -> np.ndarray:
    """Convert physical units back to saturated int16 raw samples."""
    arr = np.asarray(physical, dtype=np.float32)
    out = arr.astype(np.float32, copy=True)
    out[..., 0:3] /= ACCEL_G_PER_LSB
    out[..., 3:6] /= GYRO_DPS_PER_LSB
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
