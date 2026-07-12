"""Window segmentation for guided and null RingCollector sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .convert import raw_to_physical
from .parse import Marker, Session

CLASS_NAMES = ["double_side_tap", "double_pinch", "pinch_hold", "double_flick", "null"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class Window:
    window_id: str
    session_id: str
    label: str
    class_id: int
    rate_hz: int
    raw: np.ndarray
    start_sample_id: int
    end_sample_id: int
    cue_sample_id: int | None = None
    onset_sample_id: int | None = None
    initial_onset_sample_id: int | None = None
    cue_to_window_end_samples: int | None = None
    perform_window_overrun_samples: int = 0
    energy_fraction_initial: float | None = None
    energy_fraction_final: float | None = None
    reanchored: bool = False
    reanchor_reason: str = ""
    threshold: float | None = None
    baseline: float | None = None
    peak_energy: float | None = None
    source: str = "guided"
    participant_id: str = "unknown"
    data_role: str = "unknown"
    guided_protocol: str = "normal"
    usage: str = "auto"


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return np.convolve(x, np.ones(n, dtype=np.float32) / float(n), mode="same")


def _redo_targets(markers: list[Marker]) -> set[tuple[str, int]]:
    """Return (label, cue_sample_id) pairs discarded by redo/retry markers."""
    targets: set[tuple[str, int]] = set()
    gos_by_label: dict[str, list[Marker]] = {}
    for marker in markers:
        if marker.event_type == "go":
            gos_by_label.setdefault(marker.label, []).append(marker)
        elif marker.event_type in {"redo", "retry"}:
            if marker.invalidated_cue_unwrapped_sample_id is not None:
                targets.add((marker.label, marker.invalidated_cue_unwrapped_sample_id))
                continue
            prior = [
                m
                for m in gos_by_label.get(marker.label, [])
                if m.cue_unwrapped_sample_id <= marker.cue_unwrapped_sample_id
            ]
            if prior:
                target = prior[-1]
                targets.add((target.label, target.cue_unwrapped_sample_id))
    return targets


def _first_onset(
    energy: np.ndarray,
    cue_index: int,
    sample_rate_hz: int,
    k: float,
    prefer_after_cue: bool,
    sustain_samples: int,
) -> tuple[int, float, float]:
    region_radius = int(round(1.5 * sample_rate_hz))
    region_start = max(0, cue_index - region_radius)
    region_end = min(len(energy), cue_index + region_radius + 1)

    baseline_start = max(region_start, cue_index - int(round(1.0 * sample_rate_hz)))
    baseline_end = max(baseline_start + 1, cue_index - int(round(0.2 * sample_rate_hz)))
    if baseline_end <= baseline_start or baseline_end > len(energy):
        fallback_end = region_start + max(1, (region_end - region_start) // 3)
        baseline_slice = energy[region_start:fallback_end]
    else:
        baseline_slice = energy[baseline_start:baseline_end]

    baseline = float(np.median(baseline_slice))
    mad = float(np.median(np.abs(baseline_slice - baseline)))
    threshold = baseline + k * max(mad, 1e-6)

    search_ranges: list[tuple[int, int]]
    if prefer_after_cue:
        search_ranges = [(cue_index, region_end), (region_start, region_end)]
    else:
        search_ranges = [(region_start, region_end)]

    sustain_samples = max(1, sustain_samples)
    for start, end in search_ranges:
        above = energy[start:end] >= threshold
        if sustain_samples == 1:
            hits = np.nonzero(above)[0]
            if len(hits):
                return start + int(hits[0]), baseline, threshold
            continue
        run = np.convolve(above.astype(np.int16), np.ones(sustain_samples, dtype=np.int16), mode="valid")
        hits = np.nonzero(run >= sustain_samples)[0]
        if len(hits):
            return start + int(hits[0]), baseline, threshold
    return cue_index, baseline, threshold


def _window_energy_fraction(
    energy: np.ndarray,
    window_start: int,
    window_end: int,
    validation_start: int,
    validation_end: int,
) -> float:
    validation_start = max(0, validation_start)
    validation_end = min(len(energy), validation_end)
    if validation_end <= validation_start:
        return 1.0

    total = float(np.sum(energy[validation_start:validation_end]))
    if total <= 1e-12:
        return 1.0

    overlap_start = max(window_start, validation_start)
    overlap_end = min(window_end, validation_end)
    if overlap_end <= overlap_start:
        return 0.0
    return float(np.sum(energy[overlap_start:overlap_end]) / total)


def _sustained_run_starts(
    energy: np.ndarray,
    threshold: float,
    start: int,
    end: int,
    sustain_samples: int,
) -> list[int]:
    start = max(0, start)
    end = min(len(energy), end)
    sustain_samples = max(1, sustain_samples)
    if end <= start:
        return []

    above = energy[start:end] >= threshold
    starts: list[int] = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < len(above) and above[j]:
            j += 1
        if (j - i) >= sustain_samples:
            starts.append(start + i)
        i = j
    return starts


def _best_reanchor_start(
    energy: np.ndarray,
    cue_index: int,
    threshold: float,
    window_samples: int,
    validation_samples: int,
    sustain_samples: int,
) -> int | None:
    validation_start = cue_index
    validation_end = min(len(energy), cue_index + validation_samples + 1)
    starts = _sustained_run_starts(energy, threshold, validation_start, validation_end, sustain_samples)
    if not starts:
        return None

    best_start = starts[0]
    best_fraction = -1.0
    best_run_energy = -1.0
    for start in starts:
        end = min(len(energy), start + window_samples)
        fraction = _window_energy_fraction(energy, start, end, validation_start, validation_end)
        run_energy = float(np.sum(energy[start:end]))
        if (fraction, run_energy, -start) > (best_fraction, best_run_energy, -best_start):
            best_start = start
            best_fraction = fraction
            best_run_energy = run_energy
    return best_start


def segment_session(
    session: Session,
    window_samples: int = 128,
    k: float = 4.0,
    prefer_after_cue: bool = True,
    sustained_crossing_samples: int = 5,
    perform_window_samples: int = 300,
    enforce_perform_window: bool = True,
    energy_validation_samples: int = 250,
    min_window_energy_fraction: float = 0.80,
) -> list[Window]:
    """Segment one session into fixed-length windows.

    Guided sessions use go markers. Because current RingCollector markers record
    the cue time rather than measured movement start, onset search defaults to
    cue-forward and falls back to the full +-1.5 s region only if needed.
    """
    sample_rate_hz = session.sample_rate_hz
    if sample_rate_hz != 120:
        raise ValueError(f"{session.session_id}: expected 120 Hz source data, got {sample_rate_hz} Hz")
    sustain_samples = max(1, sustained_crossing_samples)

    if session.mode == "null":
        stride = window_samples // 2
        label = "null"
        windows = []
        for start in range(0, len(session.raw) - window_samples + 1, stride):
            ids = session.sample_ids[start : start + window_samples]
            if int(ids[-1] - ids[0]) != window_samples - 1:
                continue
            start_id = int(session.sample_ids[start])
            windows.append(
                Window(
                    window_id=f"{session.session_id}:null:{start_id}",
                    session_id=session.session_id,
                    label=label,
                    class_id=CLASS_TO_ID[label],
                    rate_hz=sample_rate_hz,
                    raw=session.raw[start : start + window_samples].copy(),
                    start_sample_id=start_id,
                    end_sample_id=int(session.sample_ids[start + window_samples - 1]),
                    source="null",
                    participant_id=session.participant_id,
                    data_role=session.data_role,
                    guided_protocol=session.guided_protocol,
                    usage=session.usage,
                )
            )
        return windows

    physical = raw_to_physical(session.raw)
    accel_norm = np.linalg.norm(physical[:, 0:3], axis=1)
    energy = _moving_average(np.abs(accel_norm - 1.0), int(round(0.100 * sample_rate_hz)))
    discard = _redo_targets(session.markers)
    windows: list[Window] = []

    for marker in session.markers:
        if marker.event_type != "go":
            continue
        if marker.label not in CLASS_TO_ID:
            raise ValueError(f"{session.session_id}: unknown gesture label {marker.label!r}")
        if (marker.label, marker.cue_unwrapped_sample_id) in discard:
            continue
        cue_index_arr = np.nonzero(session.sample_ids == marker.cue_unwrapped_sample_id)[0]
        if len(cue_index_arr) == 0:
            raise ValueError(
                f"{session.session_id}: marker cue sample {marker.cue_unwrapped_sample_id} "
                "does not exist in imu.csv"
            )
        cue_index = int(cue_index_arr[0])
        onset, baseline, threshold = _first_onset(
            energy,
            cue_index,
            sample_rate_hz,
            k=k,
            prefer_after_cue=prefer_after_cue,
            sustain_samples=sustain_samples,
        )
        initial_onset = onset
        start = onset
        end = start + window_samples
        if end > len(session.raw):
            start = len(session.raw) - window_samples
            end = len(session.raw)
        if start < 0:
            continue
        validation_start = cue_index
        validation_end = min(len(energy), cue_index + energy_validation_samples + 1)
        initial_fraction = _window_energy_fraction(energy, start, end, validation_start, validation_end)
        final_fraction = initial_fraction
        reanchored = False
        reanchor_reason = ""

        if initial_fraction < min_window_energy_fraction:
            reanchor_start = _best_reanchor_start(
                energy,
                cue_index,
                threshold,
                window_samples,
                energy_validation_samples,
                sustain_samples,
            )
            if reanchor_start is not None and reanchor_start != start:
                candidate_end = min(len(session.raw), reanchor_start + window_samples)
                candidate_start = max(0, candidate_end - window_samples)
                candidate_fraction = _window_energy_fraction(
                    energy,
                    candidate_start,
                    candidate_end,
                    validation_start,
                    validation_end,
                )
                if candidate_fraction > initial_fraction:
                    start = candidate_start
                    end = candidate_end
                    onset = reanchor_start
                    final_fraction = candidate_fraction
                    reanchored = True
                    reanchor_reason = (
                        f"energy_fraction {initial_fraction:.3f} < "
                        f"{min_window_energy_fraction:.3f}; reanchored_to_largest_sustained_burst"
                    )
        cue_to_window_end = (end - 1) - cue_index
        overrun = max(0, cue_to_window_end - perform_window_samples)
        if enforce_perform_window and overrun > 0:
            raise ValueError(
                f"{session.session_id}: segmented window exceeds perform window for "
                f"{marker.label} cue={marker.cue_unwrapped_sample_id}: "
                f"onset_offset={onset - cue_index} samples, "
                f"window_end_offset={cue_to_window_end} samples, "
                f"perform_window={perform_window_samples} samples, overrun={overrun}"
            )
        start_id = int(session.sample_ids[start])
        region_radius = int(round(1.5 * sample_rate_hz))
        region = energy[max(0, cue_index - region_radius) : min(len(energy), cue_index + region_radius + 1)]
        windows.append(
            Window(
                window_id=f"{session.session_id}:{marker.label}:{marker.cue_unwrapped_sample_id}",
                session_id=session.session_id,
                label=marker.label,
                class_id=CLASS_TO_ID[marker.label],
                rate_hz=sample_rate_hz,
                raw=session.raw[start:end].copy(),
                start_sample_id=start_id,
                end_sample_id=int(session.sample_ids[end - 1]),
                cue_sample_id=marker.cue_unwrapped_sample_id,
                onset_sample_id=int(session.sample_ids[onset]),
                initial_onset_sample_id=int(session.sample_ids[initial_onset]),
                cue_to_window_end_samples=cue_to_window_end,
                perform_window_overrun_samples=overrun,
                energy_fraction_initial=initial_fraction,
                energy_fraction_final=final_fraction,
                reanchored=reanchored,
                reanchor_reason=reanchor_reason,
                threshold=threshold,
                baseline=baseline,
                peak_energy=float(np.max(region)) if len(region) else None,
                participant_id=session.participant_id,
                data_role=session.data_role,
                guided_protocol=session.guided_protocol,
                usage=session.usage,
            )
        )
    return windows


def segment_sessions(sessions: Iterable[Session], **kwargs) -> list[Window]:
    windows: list[Window] = []
    for session in sessions:
        windows.extend(segment_session(session, **kwargs))
    return windows


def clone_window_with_raw(window: Window, raw: np.ndarray, rate_hz: int) -> Window:
    return replace(window, raw=raw.astype(np.int16, copy=False), rate_hz=rate_hz)
