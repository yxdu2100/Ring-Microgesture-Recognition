"""Create synthetic RingCollector exports for plumbing tests only.

These sessions are not valid study data and must never be used for reported
model results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ringdata.segment import CLASS_NAMES

SEED = 20260706
RATE_HZ = 120
DT_US = 8352


def _gesture_wave(label: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    x = np.zeros((n, 6), dtype=np.float32)
    if label == "double_side_tap":
        pulses = np.exp(-((t - 0.18) / 0.035) ** 2) + np.exp(-((t - 0.42) / 0.035) ** 2)
        x[:, 0] += 2600 * pulses
        x[:, 4] += 1200 * np.sin(18 * np.pi * t) * pulses
    elif label == "double_pinch":
        pulses = np.exp(-((t - 0.22) / 0.05) ** 2) + np.exp(-((t - 0.52) / 0.05) ** 2)
        x[:, 1] += 1700 * pulses
        x[:, 3] += 900 * np.sin(12 * np.pi * t) * pulses
    elif label == "pinch_hold":
        ramp = 1 / (1 + np.exp(-(t - 0.22) * 35))
        hold = ramp * (1 - 1 / (1 + np.exp(-(t - 0.78) * 35)))
        x[:, 2] += 1300 * hold
        x[:, 5] += 350 * np.sin(4 * np.pi * t) * hold
    elif label == "double_flick":
        x[:, 3] += 4300 * np.sin(12 * np.pi * t) * np.exp(-((t - 0.32) / 0.22) ** 2)
        x[:, 4] += 2600 * np.sin(14 * np.pi * t + 0.4) * np.exp(-((t - 0.34) / 0.25) ** 2)
        x[:, 0] += 900 * np.sin(8 * np.pi * t)
    x += rng.normal(0, 80, size=x.shape)
    return x


def _write_session(root: Path, folder_name: str, mode: str, rng: np.random.Generator, group_id: str) -> None:
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    markers = []
    sample_id = 0
    total = 0

    def add_samples(raw: np.ndarray) -> None:
        nonlocal sample_id, total
        for row in raw:
            rows.append(
                [
                    sample_id,
                    sample_id * DT_US,
                    10_000_000 + sample_id * 384,
                    1,
                    *np.clip(np.rint(row), -32768, 32767).astype(np.int16).tolist(),
                ]
            )
            sample_id += 1
            total += 1

    if mode == "guided":
        rest = np.array([0, 0, -4096, 0, 0, 0], dtype=np.float32)
        add_samples(rest + rng.normal(0, 35, size=(240, 6)))
        for label in CLASS_NAMES[:-1]:
            markers.append(["block_start", label, sample_id, "2026-07-06T00:00:00Z"])
            for _ in range(12):
                add_samples(rest + rng.normal(0, 35, size=(160, 6)))
                cue = sample_id
                markers.append(["go", label, cue, "2026-07-06T00:00:00Z"])
                lead = rest + rng.normal(0, 35, size=(35, 6))
                gesture = rest + _gesture_wave(label, 128, rng)
                tail = rest + rng.normal(0, 35, size=(120, 6))
                add_samples(np.vstack([lead, gesture, tail]))
            markers.append(["block_end", label, sample_id, "2026-07-06T00:00:00Z"])
    else:
        rest = np.array([0, 0, -4096, 0, 0, 0], dtype=np.float32)
        drift = np.cumsum(rng.normal(0, 3, size=(3600, 6)), axis=0)
        add_samples(rest + drift + rng.normal(0, 45, size=(3600, 6)))

    with (folder / "imu.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unwrapped_sample_id", "timestamp_us", "timestamp_ticks", "timestamp_flags", "ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(rows)
    with (folder / "markers.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_type", "label", "cue_unwrapped_sample_id", "phone_wallclock_iso"])
        writer.writerows(markers)
    meta = {
        "sessionID": group_id,
        "participantID": group_id[-2:],
        "mode": mode,
        "gestureSetVersion": "synthetic_plumbing_only",
        "imuConfig": "120hz_8g_2000dps",
        "startWallclock": "2026-07-06T00:00:00Z",
        "endWallclock": "2026-07-06T00:05:00Z",
        "totalSamples": total,
        "droppedSamples": 0,
        "hardwareTimestampCount": total,
        "interpolatedCount": 0,
        "fallbackCount": 0,
        "fifoOverrunCount": 0,
        "nonmonotonicCount": 0,
        "disconnectCount": 0,
        "notes": "SYNTHETIC PLUMBING TEST ONLY - never report as results",
    }
    if mode == "null":
        meta["label"] = "null"
    with (folder / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="ml/synthetic_data")
    parser.add_argument("--sessions", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.out_dir)
    rng = np.random.default_rng(SEED)
    for idx in range(1, args.sessions + 1):
        group_id = f"synthetic_{idx:02d}"
        _write_session(root, f"{group_id}_guided", "guided", rng, group_id)
        _write_session(root, f"{group_id}_null", "null", rng, group_id)
    print(f"wrote synthetic plumbing data to {root}")


if __name__ == "__main__":
    main()
