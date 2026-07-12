"""Shared data-loading utilities for ring microgesture experiments."""

from .convert import raw_to_physical, physical_to_raw
from .continuous import (
    ActivationEvent,
    confirm_consecutive_predictions,
    correct_activation_survival_fraction,
    match_events_to_gestures,
    recorded_hours,
    stream_windows,
)
from .manifest import ManifestEntry, apply_manifest, load_manifest
from .parse import FLAG_BITS, HARDWARE_FLAG, Marker, Session, load_session, load_sessions
from .rates import decimate_window, resample_windows
from .segment import CLASS_NAMES, Window, segment_session, segment_sessions
from .splits import build_or_load_splits, load_splits

__all__ = [
    "CLASS_NAMES",
    "ActivationEvent",
    "FLAG_BITS",
    "HARDWARE_FLAG",
    "Marker",
    "ManifestEntry",
    "Session",
    "Window",
    "apply_manifest",
    "confirm_consecutive_predictions",
    "correct_activation_survival_fraction",
    "build_or_load_splits",
    "decimate_window",
    "load_session",
    "load_sessions",
    "load_manifest",
    "match_events_to_gestures",
    "load_splits",
    "physical_to_raw",
    "raw_to_physical",
    "recorded_hours",
    "resample_windows",
    "segment_session",
    "segment_sessions",
    "stream_windows",
]
