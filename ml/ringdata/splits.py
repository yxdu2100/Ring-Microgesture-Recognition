"""Frozen cross-session and within-session split handling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .segment import CLASS_NAMES, Window

DEFAULT_SEED = 20260706


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


def _build_within_session(windows: list[Window], seed: int) -> dict:
    rng = random.Random(seed)
    by_session = _ids_by_session(windows)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for ids in by_session.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        if len(ids) < 3:
            train.extend(ids)
            continue
        n_test = max(1, round(0.2 * len(ids)))
        n_val = max(1, round(0.2 * len(ids)))
        test.extend(ids[:n_test])
        val.extend(ids[n_test : n_test + n_val])
        train.extend(ids[n_test + n_val :])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_splits(windows: list[Window], seed: int = DEFAULT_SEED) -> dict:
    return {
        "version": 1,
        "seed": seed,
        "classes": CLASS_NAMES,
        "cross_session": _build_cross_session(windows, seed),
        "within_session": _build_within_session(windows, seed),
    }


def load_splits(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def build_or_load_splits(
    windows: list[Window],
    path: str | Path = "ml/splits.json",
    seed: int = DEFAULT_SEED,
) -> dict:
    path = Path(path)
    if path.exists():
        return load_splits(path)
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
    train = set(splits["cross_session"].get("train_sessions", []))
    val = set(splits["cross_session"].get("val_sessions", []))
    test = set(splits["cross_session"].get("test_sessions", []))
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise ValueError(f"cross-session split leakage: {sorted(overlap)}")
