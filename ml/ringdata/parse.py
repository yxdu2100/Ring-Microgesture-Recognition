"""Parse RingCollector session exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

HARDWARE_FLAG = 1 << 0
INTERPOLATED_FLAG = 1 << 1
FALLBACK_FLAG = 1 << 2
FIFO_OVERRUN_FLAG = 1 << 3
NONMONOTONIC_FLAG = 1 << 4

FLAG_BITS = {
    "hardware": HARDWARE_FLAG,
    "interpolated": INTERPOLATED_FLAG,
    "fallback": FALLBACK_FLAG,
    "fifo_overrun": FIFO_OVERRUN_FLAG,
    "nonmonotonic": NONMONOTONIC_FLAG,
}


@dataclass(frozen=True)
class Marker:
    event_type: str
    label: str
    cue_unwrapped_sample_id: int
    invalidated_cue_unwrapped_sample_id: int | None = None
    phone_wallclock_iso: str = ""


@dataclass
class Session:
    session_id: str
    folder: Path
    meta: dict
    sample_ids: np.ndarray
    timestamp_us: np.ndarray
    timestamp_ticks: np.ndarray
    timestamp_flags: np.ndarray
    raw: np.ndarray
    markers: list[Marker]
    gap_count: int
    missing_sample_count: int
    flag_counts: dict[str, int]
    hardware_flag_percentage: float

    @property
    def mode(self) -> str:
        return str(self.meta.get("mode", "guided"))

    @property
    def sample_rate_hz(self) -> int:
        config = str(self.meta.get("imuConfig", "")).lower()
        if "120" in config:
            return 120
        if "60" in config:
            return 60
        if len(self.timestamp_us) > 1:
            dt = float(np.median(np.diff(self.timestamp_us))) / 1_000_000.0
            if dt > 0:
                return int(round(1.0 / dt))
        return 120


def _read_markers(path: Path) -> list[Marker]:
    if not path.exists():
        return []
    markers: list[Marker] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            invalidated = row.get("invalidated_cue_unwrapped_sample_id", "").strip()
            markers.append(
                Marker(
                    event_type=row.get("event_type", "").strip(),
                    label=row.get("label", "").strip(),
                    cue_unwrapped_sample_id=int(row.get("cue_unwrapped_sample_id", "0")),
                    invalidated_cue_unwrapped_sample_id=int(invalidated) if invalidated else None,
                    phone_wallclock_iso=row.get("phone_wallclock_iso", "").strip(),
                )
            )
    return markers


def _flag_counts(flags: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero((flags & bit) != 0)) for name, bit in FLAG_BITS.items()}


def load_session(folder: str | Path, min_hardware_pct: float = 99.0) -> Session:
    """Load and validate one RingCollector export folder."""
    folder = Path(folder)
    meta_path = folder / "meta.json"
    imu_path = folder / "imu.csv"
    marker_path = folder / "markers.csv"
    missing = [str(p) for p in (meta_path, imu_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{folder}: missing required export file(s): {', '.join(missing)}")

    with meta_path.open() as f:
        meta = json.load(f)

    rows = []
    with imu_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "unwrapped_sample_id",
            "timestamp_us",
            "timestamp_ticks",
            "timestamp_flags",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        }
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{imu_path}: missing columns: {sorted(missing_cols)}")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"{imu_path}: no IMU samples")

    sample_ids = np.array([int(r["unwrapped_sample_id"]) for r in rows], dtype=np.int64)
    timestamp_us = np.array([int(r["timestamp_us"]) for r in rows], dtype=np.int64)
    timestamp_ticks = np.array([int(r["timestamp_ticks"]) for r in rows], dtype=np.int64)
    timestamp_flags = np.array([int(r["timestamp_flags"]) for r in rows], dtype=np.uint8)
    raw = np.array(
        [[int(r[c]) for c in ("ax", "ay", "az", "gx", "gy", "gz")] for r in rows],
        dtype=np.int16,
    )

    diffs = np.diff(sample_ids)
    if np.any(diffs <= 0):
        bad = int(np.nonzero(diffs <= 0)[0][0])
        raise ValueError(
            f"{folder}: unwrapped_sample_id must be strictly monotonic; "
            f"{sample_ids[bad]} -> {sample_ids[bad + 1]} at row {bad + 2}"
        )
    gap_count = int(np.count_nonzero(diffs != 1))
    missing_sample_count = int(np.sum(np.maximum(diffs - 1, 0))) if len(diffs) else 0
    flags = _flag_counts(timestamp_flags)
    hardware_pct = 100.0 * flags["hardware"] / float(len(timestamp_flags))
    if hardware_pct < min_hardware_pct:
        raise ValueError(
            f"{folder}: HARDWARE timestamp percentage {hardware_pct:.2f}% is below "
            f"required {min_hardware_pct:.2f}%"
        )

    session_id = str(meta.get("sessionID") or folder.name)
    return Session(
        session_id=session_id,
        folder=folder,
        meta=meta,
        sample_ids=sample_ids,
        timestamp_us=timestamp_us,
        timestamp_ticks=timestamp_ticks,
        timestamp_flags=timestamp_flags,
        raw=raw,
        markers=_read_markers(marker_path),
        gap_count=gap_count,
        missing_sample_count=missing_sample_count,
        flag_counts=flags,
        hardware_flag_percentage=hardware_pct,
    )


def load_sessions(data_dir: str | Path, min_hardware_pct: float = 99.0) -> list[Session]:
    """Load every direct child session folder in a data directory."""
    data_dir = Path(data_dir)
    folders: Iterable[Path]
    if (data_dir / "imu.csv").exists():
        folders = [data_dir]
    else:
        folders = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "imu.csv").exists())
    sessions = [load_session(p, min_hardware_pct=min_hardware_pct) for p in folders]
    if not sessions:
        raise ValueError(f"{data_dir}: no RingCollector session folders found")
    return sessions
