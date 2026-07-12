"""Feature set mirrored for MEMS Studio and the software tree baseline."""

from __future__ import annotations

import numpy as np

from ringdata.convert import raw_to_physical
from ringdata.segment import Window

CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]
MLC_INPUT_NAMES = {
    "ACC_X": 0,
    "ACC_Y": 1,
    "ACC_Z": 2,
    "GYR_X": 3,
    "GYR_Y": 4,
    "GYR_Z": 5,
}
ACC_SENS_G_PER_LSB = {
    2: 0.000061,
    4: 0.000122,
    8: 0.000244,
    16: 0.000488,
}
GYR_SENS_DPS_PER_LSB = {
    125: 0.004375,
    250: 0.00875,
    500: 0.0175,
    1000: 0.035,
    2000: 0.07,
    4000: 0.14,
}
MEMS_STUDIO_GYRO_INTERNAL_PER_LSB = 0.001220703125


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


def raw_to_mlc_physical(
    raw: np.ndarray,
    accel_fs_g: int = 8,
    gyro_fs_dps: int = 2000,
    precision: str = "fp16",
    accel_lsb_scale: float | None = None,
    gyro_lsb_scale: float | None = None,
) -> np.ndarray:
    """Return MLC physical units: accel in g, gyro in dps.

    ST's MEMS Studio thresholds are in physical units. The default sensitivities
    follow the LSM6DSV16X data-sheet style values used by MEMS Studio for
    +/-8 g and +/-2000 dps: 0.244 mg/LSB and 70 mdps/LSB.
    """
    if accel_lsb_scale is None and accel_fs_g not in ACC_SENS_G_PER_LSB:
        raise ValueError(f"unsupported accel full-scale: {accel_fs_g} g")
    if gyro_lsb_scale is None and gyro_fs_dps not in GYR_SENS_DPS_PER_LSB:
        raise ValueError(f"unsupported gyro full-scale: {gyro_fs_dps} dps")
    if precision not in {"fp16", "fp64"}:
        raise ValueError(f"unknown precision: {precision}")
    dtype = np.float16 if precision == "fp16" else np.float64
    arr = np.asarray(raw, dtype=dtype)
    out = arr.astype(dtype, copy=True)
    acc_scale = ACC_SENS_G_PER_LSB[accel_fs_g] if accel_lsb_scale is None else accel_lsb_scale
    gyr_scale = GYR_SENS_DPS_PER_LSB[gyro_fs_dps] if gyro_lsb_scale is None else gyro_lsb_scale
    out[..., 0:3] = (out[..., 0:3] * dtype(acc_scale)).astype(dtype)
    out[..., 3:6] = (out[..., 3:6] * dtype(gyr_scale)).astype(dtype)
    return out


def mlc_series(
    raw: np.ndarray,
    input_name: str,
    accel_fs_g: int = 8,
    gyro_fs_dps: int = 2000,
    precision: str = "fp16",
    accel_lsb_scale: float | None = None,
    gyro_lsb_scale: float | None = None,
) -> np.ndarray:
    x = raw_to_mlc_physical(
        raw,
        accel_fs_g=accel_fs_g,
        gyro_fs_dps=gyro_fs_dps,
        precision=precision,
        accel_lsb_scale=accel_lsb_scale,
        gyro_lsb_scale=gyro_lsb_scale,
    )
    if input_name in MLC_INPUT_NAMES:
        return x[:, MLC_INPUT_NAMES[input_name]]
    if input_name == "ACC_V":
        return np.linalg.norm(x[:, 0:3].astype(np.float32), axis=1).astype(x.dtype)
    if input_name == "GYR_V":
        return np.linalg.norm(x[:, 3:6].astype(np.float32), axis=1).astype(x.dtype)
    raise ValueError(f"unsupported MLC input: {input_name}")


def mlc_feature_value(
    raw: np.ndarray,
    feature_name: str,
    input_name: str,
    accel_fs_g: int = 8,
    gyro_fs_dps: int = 2000,
    precision: str = "fp16",
    accel_lsb_scale: float | None = None,
    gyro_lsb_scale: float | None = None,
) -> np.float16 | np.float64:
    """Compute one MEMS Studio-style feature.

    Assumptions for features not exercised by the current ST export:
    variance is population variance and zero-crossing counts sign changes
    around the window mean.
    """
    if precision not in {"fp16", "fp64"}:
        raise ValueError(f"unknown precision: {precision}")
    dtype = np.float16 if precision == "fp16" else np.float64
    y = mlc_series(
        raw,
        input_name,
        accel_fs_g=accel_fs_g,
        gyro_fs_dps=gyro_fs_dps,
        precision=precision,
        accel_lsb_scale=accel_lsb_scale,
        gyro_lsb_scale=gyro_lsb_scale,
    )
    yf = y.astype(np.float32 if precision == "fp16" else np.float64)
    if feature_name == "MEAN":
        value = np.mean(yf)
    elif feature_name == "ABS_MEAN":
        value = np.mean(np.abs(yf))
    elif feature_name == "VARIANCE":
        value = np.var(yf)
    elif feature_name == "ENERGY":
        value = np.sum(yf * yf)
    elif feature_name == "PEAK_TO_PEAK":
        value = np.ptp(yf)
    elif feature_name == "MINIMUM":
        value = np.min(yf)
    elif feature_name == "MAXIMUM":
        value = np.max(yf)
    elif feature_name == "ABS_MAXIMUM":
        value = np.max(np.abs(yf))
    elif feature_name == "ZERO_CROSSING":
        centered = yf - np.mean(yf)
        signs = np.signbit(centered)
        value = np.count_nonzero(signs[1:] != signs[:-1])
    else:
        raise ValueError(f"unsupported MLC feature: {feature_name}")
    with np.errstate(over="ignore", invalid="ignore"):
        return dtype(value)


def mlc_feature_dict(
    raw: np.ndarray,
    feature_tokens: list[str],
    accel_fs_g: int = 8,
    gyro_fs_dps: int = 2000,
    precision: str = "fp16",
    accel_lsb_scale: float | None = None,
    gyro_lsb_scale: float | None = None,
) -> dict[str, np.float16 | np.float64]:
    return {
        token: mlc_feature_value(
            raw,
            *parse_mlc_feature_token(token),
            accel_fs_g=accel_fs_g,
            gyro_fs_dps=gyro_fs_dps,
            precision=precision,
            accel_lsb_scale=accel_lsb_scale,
            gyro_lsb_scale=gyro_lsb_scale,
        )
        for token in feature_tokens
    }


def parse_mlc_feature_token(token: str) -> tuple[str, str]:
    parts = token.split("_")
    if len(parts) < 3 or not parts[0].startswith("F"):
        raise ValueError(f"invalid MLC feature token: {token}")
    axis = "_".join(parts[-2:])
    feature = "_".join(parts[1:-2])
    return feature, axis


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
