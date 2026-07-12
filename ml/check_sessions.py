"""Fast post-collection quality report for RingCollector exports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, apply_manifest, load_sessions, segment_sessions


def check_sessions(sessions, expected_reps: int) -> tuple[list[dict], bool]:
    rows = []
    any_failure = False
    gesture_names = [name for name in CLASS_NAMES if name != "null"]
    for session in sessions:
        windows = segment_sessions([session], enforce_perform_window=False)
        guided = [window for window in windows if window.source == "guided"]
        counts = Counter(window.label for window in guided)
        dt = np.diff(session.timestamp_us)
        median_dt = float(np.median(dt)) if len(dt) else 0.0
        measured_hz = 1_000_000.0 / median_dt if median_dt > 0 else 0.0
        reanchor_pct = (
            100.0 * sum(window.reanchored for window in guided) / len(guided)
            if guided else 0.0
        )
        failures = []
        warnings = []
        if session.gap_count:
            failures.append(f"{session.gap_count} sample-id gaps")
        if session.hardware_flag_percentage < 99.0:
            failures.append(f"hardware timestamps {session.hardware_flag_percentage:.2f}%")
        if abs(measured_hz - session.sample_rate_hz) > 0.05 * session.sample_rate_hz:
            failures.append(f"measured rate {measured_hz:.2f} Hz")
        if guided:
            for gesture in gesture_names:
                if counts[gesture] != expected_reps:
                    failures.append(f"{gesture}={counts[gesture]} (expected {expected_reps})")
            if reanchor_pct > 30.0:
                failures.append(f"reanchor rate {reanchor_pct:.1f}%")
            overruns = sum(window.perform_window_overrun_samples > 0 for window in guided)
            if overruns:
                warnings.append(f"{overruns} windows exceed guided perform interval")

        status = "FAIL" if failures else ("WARN" if warnings else "PASS")
        any_failure |= bool(failures)
        row = {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "role": session.data_role,
            "usage": session.usage,
            "status": status,
            "duration_minutes": len(session.raw) / session.sample_rate_hz / 60.0,
            "samples": len(session.raw),
            "measured_hz": measured_hz,
            "gap_count": session.gap_count,
            "missing_samples": session.missing_sample_count,
            "hardware_pct": session.hardware_flag_percentage,
            "reanchor_pct": reanchor_pct,
            "failures": "; ".join(failures),
            "warnings": "; ".join(warnings),
        }
        row.update({f"{gesture}_windows": counts[gesture] for gesture in gesture_names})
        rows.append(row)
    return rows, any_failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="ml/dataset_manifest.csv")
    parser.add_argument("--session", action="append", default=[],
                        help="Session ID to check; repeat for multiple (default: all)")
    parser.add_argument("--expected-reps-per-gesture", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sessions, warnings = apply_manifest(load_sessions(args.data_dir), args.manifest)
    for warning in warnings:
        print(f"warning: {warning}")
    if args.session:
        selected = set(args.session)
        sessions = [session for session in sessions if session.session_id in selected]
        missing = selected - {session.session_id for session in sessions}
        if missing:
            raise ValueError(f"unknown session(s): {', '.join(sorted(missing))}")
    rows, failed = check_sessions(sessions, args.expected_reps_per_gesture)
    if not rows:
        raise ValueError("no sessions selected")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        detail = row["failures"] or row["warnings"]
        print(f"{row['status']:4s} {row['session_id']} {row['role']}: {detail}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
