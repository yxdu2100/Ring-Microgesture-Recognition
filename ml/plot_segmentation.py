"""Plot segmentation diagnostics for quick onset sanity checks."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from ringdata.convert import raw_to_physical
from ringdata.parse import load_sessions
from ringdata.segment import CLASS_NAMES, segment_sessions


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n, dtype=np.float32) / float(n), mode="same")


def _session_energy(session):
    physical = raw_to_physical(session.raw)
    accel_norm = np.linalg.norm(physical[:, 0:3], axis=1)
    return _moving_average(np.abs(accel_norm - 1.0), int(round(0.100 * session.sample_rate_hz)))


def write_segmentation_report(sessions, windows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_id",
                "session_id",
                "label",
                "cue_sample_id",
                "onset_sample_id",
                "initial_onset_sample_id",
                "onset_offset_samples",
                "initial_onset_offset_samples",
                "window_end_offset_samples",
                "perform_window_overrun_samples",
                "energy_fraction_initial",
                "energy_fraction_final",
                "reanchored",
                "reanchor_reason",
                "baseline",
                "threshold",
                "peak_energy",
            ]
        )
        for w in windows:
            onset_offset = "" if w.onset_sample_id is None or w.cue_sample_id is None else w.onset_sample_id - w.cue_sample_id
            initial_onset_offset = (
                ""
                if w.initial_onset_sample_id is None or w.cue_sample_id is None
                else w.initial_onset_sample_id - w.cue_sample_id
            )
            writer.writerow(
                [
                    w.window_id,
                    w.session_id,
                    w.label,
                    w.cue_sample_id,
                    w.onset_sample_id,
                    w.initial_onset_sample_id,
                    onset_offset,
                    initial_onset_offset,
                    w.cue_to_window_end_samples,
                    w.perform_window_overrun_samples,
                    w.energy_fraction_initial,
                    w.energy_fraction_final,
                    w.reanchored,
                    w.reanchor_reason,
                    w.baseline,
                    w.threshold,
                    w.peak_energy,
                ]
            )


def plot_windows(sessions, windows, out_png: Path, examples_per_class: int = 5) -> None:
    sessions_by_id = {s.session_id: s for s in sessions}
    energy_by_id = {s.session_id: _session_energy(s) for s in sessions}
    by_class = defaultdict(list)
    for window in windows:
        if window.label != "null":
            by_class[window.label].append(window)

    labels = [label for label in CLASS_NAMES if label != "null" and by_class[label]]
    fig, axes = plt.subplots(len(labels), examples_per_class, figsize=(2.8 * examples_per_class, 2.15 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = np.array([axes])

    for row, label in enumerate(labels):
        for col in range(examples_per_class):
            ax = axes[row, col]
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            if col >= len(by_class[label]):
                ax.axis("off")
                continue
            window = by_class[label][col]
            session = sessions_by_id[window.session_id]
            energy = energy_by_id[window.session_id]
            cue_idx = int(np.nonzero(session.sample_ids == window.cue_sample_id)[0][0])
            start = max(0, cue_idx - 60)
            end = min(len(energy), cue_idx + 300)
            x = np.arange(start, end) - cue_idx
            ax.plot(x, energy[start:end], color="#1f77b4", linewidth=1.2)
            if window.threshold is not None:
                ax.axhline(window.threshold, color="#888888", linewidth=0.8, linestyle=":")
            ax.axvline(0, color="#222222", linewidth=0.9, linestyle="--")
            if window.initial_onset_sample_id is not None and window.initial_onset_sample_id != window.onset_sample_id:
                ax.axvline(
                    window.initial_onset_sample_id - window.cue_sample_id,
                    color="#ff9896",
                    linewidth=0.9,
                    linestyle=":",
                )
            ax.axvline(window.onset_sample_id - window.cue_sample_id, color="#d62728", linewidth=1.0)
            ax.axvspan(
                window.start_sample_id - window.cue_sample_id,
                window.end_sample_id - window.cue_sample_id,
                color="#d62728",
                alpha=0.08,
            )
            suffix = " *" if window.reanchored else ""
            ax.set_title(f"{label} #{col + 1}{suffix}", fontsize=9)
            if col == 0:
                ax.set_ylabel("|norm(a)-1g|")
            if row == len(labels) - 1:
                ax.set_xlabel("samples from cue")
    fig.suptitle("Segmentation diagnostic: dashed=cue, red=final onset, dotted pink=initial onset, shaded=window", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="ml/results/segmentation_diagnostics")
    parser.add_argument("--examples-per-class", type=int, default=5)
    parser.add_argument("--sustained-crossing-samples", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions, sustained_crossing_samples=args.sustained_crossing_samples)
    write_segmentation_report(sessions, windows, out_dir / "segmentation_report.csv")
    plot_windows(sessions, windows, out_dir / "segmentation_examples.png", args.examples_per_class)
    print(f"wrote {out_dir / 'segmentation_report.csv'}")
    print(f"wrote {out_dir / 'segmentation_examples.png'}")


if __name__ == "__main__":
    main()
