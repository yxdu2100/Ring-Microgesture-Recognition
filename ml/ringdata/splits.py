"""Frozen cross-session and within-session split handling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .segment import CLASS_NAMES, Window

DEFAULT_SEED = 20260706
SPLIT_VERSION = 2
EXCLUDED_SESSIONS = {"20260706_002"}
WINDOW_SAMPLES = 128
NULL_STEP_SAMPLES = 64


def _ids_by_session(windows: Iterable[Window]) -> dict[str, list[str]]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for window in windows:
        by_session[window.session_id].append(window.window_id)
    return dict(by_session)


def _labels_by_session(windows: Iterable[Window]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        labels[window.session_id].add(window.label)
    return dict(labels)


def _flatten(session_ids: list[str], by_session: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for session_id in session_ids:
        out.extend(sorted(by_session[session_id]))
    return out


def _filter_supported_windows(windows: list[Window]) -> list[Window]:
    return [w for w in windows if w.session_id not in EXCLUDED_SESSIONS]


def _build_cross_session(windows: list[Window], seed: int) -> dict:
    by_session = _ids_by_session(windows)
    labels_by_session = _labels_by_session(windows)
    sessions = sorted(by_session)
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)

    guided_sessions = [s for s in shuffled if labels_by_session[s] - {"null"}]
    null_sessions = [s for s in shuffled if labels_by_session[s] == {"null"}]

    if len(guided_sessions) >= 2 and len(null_sessions) >= 2:
        test_sessions = sorted([guided_sessions[0], null_sessions[0]])
        val_sessions = sorted([null_sessions[1]]) if len(null_sessions) >= 3 else []
        held = set(test_sessions) | set(val_sessions)
        train_sessions = sorted(s for s in sessions if s not in held)
    elif len(guided_sessions) >= 2 and len(null_sessions) >= 1:
        test_sessions = sorted([guided_sessions[0], null_sessions[0]])
        val_sessions = []
        held = set(test_sessions)
        train_sessions = sorted(s for s in sessions if s not in held)
    else:
        rng.shuffle(sessions)

        if len(sessions) >= 3:
            n_test = max(1, round(0.2 * len(sessions)))
            n_val = max(1, round(0.2 * len(sessions)))
            test_sessions = sorted(sessions[:n_test])
            val_sessions = sorted(sessions[n_test : n_test + n_val])
            train_sessions = sorted(sessions[n_test + n_val :])
        elif len(sessions) == 2:
            train_sessions = sorted([sessions[0]])
            val_sessions = []
            test_sessions = sorted([sessions[1]])
        else:
            train_sessions = sessions
            val_sessions = []
            test_sessions = []

    return {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
        "train": _flatten(train_sessions, by_session),
        "val": _flatten(val_sessions, by_session),
        "test": _flatten(test_sessions, by_session),
    }


def _split_sorted_groups(groups: list[tuple[int, int, list[str]]]) -> dict[str, list[str]]:
    """Split contiguous timeline groups 70/15/15 without breaking a group."""
    if len(groups) < 3:
        return {"train": [i for *_unused, ids in groups for i in ids], "val": [], "test": []}

    groups = sorted(groups, key=lambda item: (item[0], item[1], item[2][0]))
    n = len(groups)
    n_train = max(1, round(0.70 * n))
    n_val = max(1, round(0.15 * n))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    chunks = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }
    return {
        split: [window_id for _start, _end, ids in chunk for window_id in ids]
        for split, chunk in chunks.items()
    }


def _build_null_timeline_groups(session_windows: list[Window]) -> list[tuple[int, int, list[str]]]:
    ordered = sorted(session_windows, key=lambda w: (w.start_sample_id, w.end_sample_id, w.window_id))
    if len(ordered) < 3:
        return [(w.start_sample_id, w.end_sample_id, [w.window_id]) for w in ordered]

    start = min(w.start_sample_id for w in ordered)
    end = max(w.end_sample_id for w in ordered) + 1
    span = max(1, end - start)
    b1 = start + round(0.70 * span)
    b2 = start + round(0.85 * span)
    blocks = [(start, b1), (b1, b2), (b2, end)]
    out: list[tuple[int, int, list[str]]] = []

    for block_start, block_end in blocks:
        ids = [
            w.window_id
            for w in ordered
            if w.start_sample_id >= block_start and w.end_sample_id < block_end
        ]
        if ids:
            out.append((block_start, block_end - 1, ids))
    return out


def _build_within_session(windows: list[Window], seed: int) -> dict:
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    by_session: dict[str, list[Window]] = defaultdict(list)
    for window in windows:
        by_session[window.session_id].append(window)

    for session_windows in by_session.values():
        labels = {w.label for w in session_windows}
        if labels == {"null"}:
            split_ids = _split_sorted_groups(_build_null_timeline_groups(session_windows))
        else:
            by_instance: dict[tuple[str, int], list[Window]] = defaultdict(list)
            for window in session_windows:
                instance_key = (window.label, window.cue_sample_id or window.start_sample_id)
                by_instance[instance_key].append(window)
            groups = []
            for instance_windows in by_instance.values():
                groups.append(
                    (
                        min(w.start_sample_id for w in instance_windows),
                        max(w.end_sample_id for w in instance_windows),
                        sorted(w.window_id for w in instance_windows),
                    )
                )
            split_ids = _split_sorted_groups(groups)

        train.extend(split_ids["train"])
        val.extend(split_ids["val"])
        test.extend(split_ids["test"])

    return {
        "kind": "time_contiguous_grouped",
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def _build_cross_session_loso(windows: list[Window], seed: int) -> list[dict]:
    by_session = _ids_by_session(windows)
    labels_by_session = _labels_by_session(windows)
    sessions = sorted(by_session)
    gesture_sessions = sorted(s for s in sessions if labels_by_session[s] - {"null"})
    null_sessions = sorted(s for s in sessions if labels_by_session[s] == {"null"})

    if len(gesture_sessions) < 2:
        return []

    folds = []
    for idx, test_gesture in enumerate(gesture_sessions):
        test_null = null_sessions[idx % len(null_sessions)] if null_sessions else None
        val_gesture_candidates = [s for s in gesture_sessions if s != test_gesture]
        val_gesture = val_gesture_candidates[idx % len(val_gesture_candidates)]
        val_null_candidates = [s for s in null_sessions if s != test_null]
        val_null = val_null_candidates[idx % len(val_null_candidates)] if val_null_candidates else None

        test_sessions = [test_gesture] + ([test_null] if test_null else [])
        val_sessions = [val_gesture] + ([val_null] if val_null else [])
        held = set(test_sessions) | set(val_sessions)
        train_sessions = sorted(s for s in sessions if s not in held)

        folds.append(
            {
                "fold_id": f"loso_{idx + 1:02d}_{test_gesture}",
                "test_gesture_session": test_gesture,
                "test_null_session": test_null or "",
                "val_gesture_session": val_gesture,
                "val_null_session": val_null or "",
                "train_sessions": train_sessions,
                "val_sessions": sorted(val_sessions),
                "test_sessions": sorted(test_sessions),
                "train": _flatten(train_sessions, by_session),
                "val": _flatten(sorted(val_sessions), by_session),
                "test": _flatten(sorted(test_sessions), by_session),
            }
        )
    return folds


def _first_loso_as_cross_session(folds: list[dict]) -> dict:
    if not folds:
        return {"train_sessions": [], "val_sessions": [], "test_sessions": [], "train": [], "val": [], "test": []}
    return {k: v for k, v in folds[0].items() if k != "fold_id"}


def build_splits(windows: list[Window], seed: int = DEFAULT_SEED) -> dict:
    windows = _filter_supported_windows(windows)
    loso = _build_cross_session_loso(windows, seed)
    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "classes": CLASS_NAMES,
        "excluded_sessions": sorted(EXCLUDED_SESSIONS),
        "cross_session": _first_loso_as_cross_session(loso) if loso else _build_cross_session(windows, seed),
        "cross_session_loso": loso,
        "within_session": _build_within_session(windows, seed),
    }


def load_splits(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def build_or_load_splits(
    windows: list[Window],
    path: str | Path = "ml/splits.json",
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False,
) -> dict:
    path = Path(path)
    if path.exists() and not force_rebuild:
        splits = load_splits(path)
        if splits.get("version") == SPLIT_VERSION:
            return splits
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = build_splits(windows, seed=seed)
    with path.open("w") as f:
        json.dump(splits, f, indent=2, sort_keys=True)
        f.write("\n")
    return splits


def select_windows(windows: list[Window], ids: list[str]) -> list[Window]:
    by_id = {w.window_id: w for w in windows}
    return [by_id[i] for i in ids if i in by_id]


def assert_no_cross_session_leakage(splits: dict) -> None:
    fold_like = splits.get("cross_session_loso") or [splits["cross_session"]]
    for fold in fold_like:
        train = set(fold.get("train_sessions", []))
        val = set(fold.get("val_sessions", []))
        test = set(fold.get("test_sessions", []))
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            fold_id = fold.get("fold_id", "cross_session")
            raise ValueError(f"{fold_id}: cross-session split leakage: {sorted(overlap)}")
