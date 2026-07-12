"""Continuous sliding-window and activation-event evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .parse import Session
from .segment import CLASS_NAMES, CLASS_TO_ID, Window

NULL_CLASS_ID = CLASS_TO_ID["null"]


@dataclass(frozen=True)
class ActivationEvent:
    session_id: str
    class_id: int
    class_name: str
    start_sample_id: int
    end_sample_id: int
    confirming_windows: int
    score: float | None = None


def stream_windows(
    sessions: Iterable[Session],
    window_samples: int = 128,
    hop_samples: int = 64,
) -> list[Window]:
    """Create chronological deployment windows from complete recordings."""
    windows: list[Window] = []
    for session in sessions:
        for start in range(0, len(session.raw) - window_samples + 1, hop_samples):
            ids = session.sample_ids[start : start + window_samples]
            if int(ids[-1] - ids[0]) != window_samples - 1:
                continue
            start_id = int(ids[0])
            windows.append(
                Window(
                    window_id=f"{session.session_id}:stream:{start_id}",
                    session_id=session.session_id,
                    label="null",
                    class_id=NULL_CLASS_ID,
                    rate_hz=session.sample_rate_hz,
                    raw=session.raw[start : start + window_samples].copy(),
                    start_sample_id=start_id,
                    end_sample_id=int(ids[-1]),
                    source="continuous",
                    participant_id=session.participant_id,
                    data_role=session.data_role,
                    guided_protocol=session.guided_protocol,
                    usage=session.usage,
                )
            )
    return windows


def confirm_consecutive_predictions(
    windows: list[Window],
    predictions,
    scores=None,
    consecutive: int = 2,
    refractory_samples: int = 120,
) -> list[ActivationEvent]:
    """Convert chronological window predictions into debounced activations.

    One event is emitted per contiguous non-null run after ``consecutive``
    agreeing windows. A session boundary, sample gap, null prediction, or class
    change resets confirmation.
    """
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(windows) != len(predictions):
        raise ValueError("window/prediction length mismatch")
    if scores is None:
        score_values: list[float | None] = [None] * len(windows)
    else:
        score_array = np.asarray(scores, dtype=np.float64)
        if len(score_array) != len(windows):
            raise ValueError("window/score length mismatch")
        score_values = [float(value) for value in score_array]

    events: list[ActivationEvent] = []
    run_class = NULL_CLASS_ID
    run_count = 0
    run_start = 0
    emitted = False
    previous: Window | None = None
    refractory_until: dict[tuple[str, int], int] = {}

    for window, prediction, score in zip(windows, predictions, score_values):
        contiguous = (
            previous is not None
            and previous.session_id == window.session_id
            and window.start_sample_id > previous.start_sample_id
            and window.start_sample_id <= previous.end_sample_id + 1
        )
        if not contiguous:
            run_class = NULL_CLASS_ID
            run_count = 0
            emitted = False

        prediction = int(prediction)
        if prediction == NULL_CLASS_ID:
            run_class = NULL_CLASS_ID
            run_count = 0
            emitted = False
        elif prediction == run_class:
            run_count += 1
        else:
            run_class = prediction
            run_count = 1
            run_start = window.start_sample_id
            emitted = False

        key = (window.session_id, prediction)
        if (
            prediction != NULL_CLASS_ID
            and run_count >= consecutive
            and not emitted
            and window.end_sample_id >= refractory_until.get(key, -1)
        ):
            events.append(
                ActivationEvent(
                    session_id=window.session_id,
                    class_id=prediction,
                    class_name=CLASS_NAMES[prediction],
                    start_sample_id=run_start,
                    end_sample_id=window.end_sample_id,
                    confirming_windows=consecutive,
                    score=score,
                )
            )
            emitted = True
            refractory_until[key] = window.end_sample_id + refractory_samples
        previous = window
    return events


def match_events_to_gestures(
    events: list[ActivationEvent],
    gesture_windows: list[Window],
    grace_samples: int = 64,
    sample_rate_hz: int = 120,
) -> tuple[dict, list[dict]]:
    """Match at most one activation to each segmented gesture instance."""
    references = sorted(
        [window for window in gesture_windows if window.label != "null"],
        key=lambda window: (window.session_id, window.onset_sample_id or window.start_sample_id),
    )
    ordered_events = sorted(events, key=lambda event: (event.session_id, event.end_sample_id))
    used_events: set[int] = set()
    rows: list[dict] = []
    correct = 0
    wrong = 0
    missed = 0
    latencies: list[float] = []

    for reference in references:
        onset = reference.onset_sample_id or reference.start_sample_id
        latest = reference.end_sample_id + grace_samples
        candidates = [
            (idx, event)
            for idx, event in enumerate(ordered_events)
            if idx not in used_events
            and event.session_id == reference.session_id
            and onset <= event.end_sample_id <= latest
        ]
        if not candidates:
            missed += 1
            rows.append({
                "window_id": reference.window_id,
                "session_id": reference.session_id,
                "true_label": reference.label,
                "predicted_label": "",
                "outcome": "missed",
                "latency_ms": "",
            })
            continue
        idx, event = min(candidates, key=lambda item: item[1].end_sample_id)
        used_events.add(idx)
        latency_ms = 1000.0 * (event.end_sample_id - onset) / float(sample_rate_hz)
        if event.class_id == reference.class_id:
            correct += 1
            latencies.append(latency_ms)
            outcome = "correct"
        else:
            wrong += 1
            outcome = "wrong_class"
        rows.append({
            "window_id": reference.window_id,
            "session_id": reference.session_id,
            "true_label": reference.label,
            "predicted_label": event.class_name,
            "outcome": outcome,
            "latency_ms": latency_ms,
        })

    unmatched = [event for idx, event in enumerate(ordered_events) if idx not in used_events]
    total = len(references)
    metrics = {
        "gesture_events": total,
        "correct_events": correct,
        "wrong_class_events": wrong,
        "missed_events": missed,
        "unmatched_activation_events": len(unmatched),
        "event_recall": correct / total if total else float("nan"),
        "latency_ms_median": float(np.median(latencies)) if latencies else float("nan"),
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else float("nan"),
    }
    return metrics, rows


def recorded_hours(sessions: Iterable[Session]) -> float:
    """Return actual sampled exposure, excluding missing/disconnected spans."""
    return sum(len(session.raw) / float(session.sample_rate_hz) for session in sessions) / 3600.0

