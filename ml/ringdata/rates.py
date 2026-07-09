"""Rate conversion for fixed windows."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .segment import Window, clone_window_with_raw


def decimate_window(raw: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    """Decimate 120 Hz raw int16 windows to 60 or 30 Hz with anti-aliasing."""
    from scipy.signal import decimate

    if source_hz == target_hz:
        return np.asarray(raw, dtype=np.int16).copy()
    if source_hz != 120 or target_hz not in {60, 30}:
        raise ValueError(f"unsupported decimation {source_hz} -> {target_hz}")
    x = np.asarray(raw, dtype=np.float32)
    if target_hz == 60:
        y = decimate(x, q=2, axis=0, ftype="iir", zero_phase=True)
    else:
        y60 = decimate(x, q=2, axis=0, ftype="iir", zero_phase=True)
        y = decimate(y60, q=2, axis=0, ftype="iir", zero_phase=True)
    expected = {60: 64, 30: 32}[target_hz]
    if y.shape[0] != expected:
        y = y[:expected]
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def resample_windows(windows: Iterable[Window], target_hz: int) -> list[Window]:
    out: list[Window] = []
    for window in windows:
        out.append(clone_window_with_raw(window, decimate_window(window.raw, window.rate_hz, target_hz), target_hz))
    return out
