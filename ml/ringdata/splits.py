"""Frozen participant/session-grouped split handling."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .segment import CLASS_NAMES, Window

DEFAULT_SEED = 20260711
SPLIT_VERSION = 3
EXCLUDED_SESSIONS: set[str] = set()  # compatibility; exclusions belong in the manifest
PRIMARY_FOLDS = 5
VALIDATION_GESTURE_SESSIONS = 2


def _ids_by_session(windows: Iterable[Window]) -> dict[str, list[str]]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for window in windows:
        by_session[window.session_id].append(window.window_id)
    return {key: sorted(value) for key, value in by_session.items()}


def _session_metadata(windows: Iterable[Window]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for window in windows:
        row = {
            "participant_id": window.participant_id,
            "data_role": window.data_role,
            "usage": window.usage,
            "guided_protocol": window.guided_protocol,
            "collection_date": window.session_id[:8],
        }
        previous = out.setdefault(window.session_id, row)
        if previous != row:
            raise ValueError(f"inconsistent metadata within session {window.session_id}")
    return out


def _dataset_hash(metadata: dict[str, dict], by_session: dict[str, list[str]]) -> str:
    payload = [
        {
            "session_id": session_id,
            **metadata[session_id],
            "window_ids": by_session[session_id],
        }
        for session_id in sorted(by_session)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _primary_participant(metadata: dict[str, dict]) -> str:
    counts = Counter(
        row["participant_id"]
        for row in metadata.values()
        if row["data_role"] == "gesture" and row["usage"] != "exclude"
    )
    if not counts:
        raise ValueError("no active gesture sessions in dataset")
    return sorted(counts, key=lambda participant: (-counts[participant], participant))[0]


def _balanced_test_groups(
    sessions: list[str],
    metadata: dict[str, dict],
    folds: int,
    seed: int,
) -> list[list[str]]:
    if len(sessions) < folds:
        raise ValueError(f"need at least {folds} gesture sessions; found {len(sessions)}")
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for session_id in sessions:
        row = metadata[session_id]
        strata[(row["collection_date"], row["guided_protocol"])].append(session_id)
    groups: list[list[str]] = [[] for _ in range(folds)]
    stratum_counts: list[Counter] = [Counter() for _ in range(folds)]
    for stratum, members in sorted(strata.items(), key=lambda item: (-len(item[1]), item[0])):
        members = sorted(members)
        rng.shuffle(members)
        for member in members:
            target = min(
                range(folds),
                key=lambda idx: (len(groups[idx]), stratum_counts[idx][stratum], idx),
            )
            groups[target].append(member)
            stratum_counts[target][stratum] += 1
    return [sorted(group) for group in groups]


def _choose_validation_sessions(
    candidates: list[str],
    count: int,
    seed: int,
) -> list[str]:
    ordered = sorted(candidates)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[: min(count, len(ordered))])


def _flatten(session_ids: Iterable[str], by_session: dict[str, list[str]]) -> list[str]:
    return [window_id for session_id in sorted(session_ids) for window_id in by_session[session_id]]


def _build_within_user_folds(
    by_session: dict[str, list[str]],
    metadata: dict[str, dict],
    participant_id: str,
    seed: int,
    fold_count: int = PRIMARY_FOLDS,
) -> list[dict]:
    gesture_sessions = sorted(
        session_id
        for session_id, row in metadata.items()
        if row["participant_id"] == participant_id
        and row["data_role"] == "gesture"
        and row["usage"] in {"fold", "auto"}
    )
    structured_train = sorted(
        session_id
        for session_id, row in metadata.items()
        if row["data_role"] == "structured_null" and row["usage"] in {"train", "auto"}
    )
    structured_val = sorted(
        session_id
        for session_id, row in metadata.items()
        if row["data_role"] == "structured_null" and row["usage"] == "validation"
    )
    development_null = sorted(
        session_id
        for session_id, row in metadata.items()
        if row["data_role"] == "free_living_null" and row["usage"] == "development"
    )
    final_test_null = sorted(
        session_id
        for session_id, row in metadata.items()
        if row["data_role"] == "free_living_null" and row["usage"] == "final_test"
    )
    test_groups = _balanced_test_groups(gesture_sessions, metadata, fold_count, seed)
    folds: list[dict] = []
    for index, test_gesture_sessions in enumerate(test_groups):
        remaining = sorted(set(gesture_sessions) - set(test_gesture_sessions))
        val_gesture_sessions = _choose_validation_sessions(
            remaining,
            VALIDATION_GESTURE_SESSIONS,
            seed + 1009 * (index + 1),
        )
        train_gesture_sessions = sorted(set(remaining) - set(val_gesture_sessions))
        train_sessions = sorted(train_gesture_sessions + structured_train)
        val_sessions = sorted(val_gesture_sessions + structured_val)
        folds.append(
            {
                "fold_id": f"within_user_{index + 1:02d}",
                "participant_id": participant_id,
                "train_gesture_sessions": train_gesture_sessions,
                "val_gesture_sessions": val_gesture_sessions,
                "test_gesture_sessions": test_gesture_sessions,
                "train_structured_null_sessions": structured_train,
                "val_structured_null_sessions": structured_val,
                "development_free_living_sessions": development_null,
                "final_test_free_living_sessions": final_test_null,
                "train_sessions": train_sessions,
                "val_sessions": val_sessions,
                "test_sessions": test_gesture_sessions,
                "train": _flatten(train_sessions, by_session),
                "val": _flatten(val_sessions, by_session),
                "test": _flatten(test_gesture_sessions, by_session),
                "development_null": _flatten(development_null, by_session),
                "final_test_null": _flatten(final_test_null, by_session),
            }
        )
    return folds


def _build_lopo_folds(
    by_session: dict[str, list[str]],
    metadata: dict[str, dict],
    seed: int,
) -> list[dict]:
    participants = sorted(
        {
            row["participant_id"]
            for row in metadata.values()
            if row["data_role"] == "gesture" and row["usage"] != "exclude"
        }
    )
    if len(participants) < 2:
        return []
    folds: list[dict] = []
    for index, held_participant in enumerate(participants):
        test_gesture = sorted(
            session_id
            for session_id, row in metadata.items()
            if row["participant_id"] == held_participant and row["data_role"] == "gesture"
        )
        train_gesture = sorted(
            session_id
            for session_id, row in metadata.items()
            if row["participant_id"] != held_participant and row["data_role"] == "gesture"
        )
        train_null = sorted(
            session_id
            for session_id, row in metadata.items()
            if row["participant_id"] != held_participant
            and row["data_role"] == "structured_null"
            and row["usage"] != "exclude"
        )
        val_gesture = _choose_validation_sessions(train_gesture, 1, seed + index)
        actual_train_gesture = sorted(set(train_gesture) - set(val_gesture))
        train_sessions = sorted(actual_train_gesture + train_null)
        folds.append(
            {
                "fold_id": f"lopo_{held_participant}",
                "held_participant_id": held_participant,
                "train_sessions": train_sessions,
                "val_sessions": val_gesture,
                "test_sessions": test_gesture,
                "train": _flatten(train_sessions, by_session),
                "val": _flatten(val_gesture, by_session),
                "test": _flatten(test_gesture, by_session),
            }
        )
    return folds


def build_splits(windows: list[Window], seed: int = DEFAULT_SEED) -> dict:
    by_session = _ids_by_session(windows)
    metadata = _session_metadata(windows)
    participant_id = _primary_participant(metadata)
    within_user = _build_within_user_folds(by_session, metadata, participant_id, seed)
    if not within_user:
        raise ValueError("could not build within-user session folds")
    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "classes": CLASS_NAMES,
        "dataset_hash": _dataset_hash(metadata, by_session),
        "primary_participant_id": participant_id,
        "session_metadata": metadata,
        "within_user_session_folds": within_user,
        "lopo_folds": _build_lopo_folds(by_session, metadata, seed),
        # Compatibility for single-split trainer/export entry points.
        "cross_session": {key: value for key, value in within_user[0].items() if key != "fold_id"},
    }


def load_splits(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def build_or_load_splits(
    windows: list[Window],
    path: str | Path = "ml/splits_within_user.json",
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False,
) -> dict:
    path = Path(path)
    current = build_splits(windows, seed=seed)
    if path.exists() and not force_rebuild:
        splits = load_splits(path)
        if splits.get("version") != SPLIT_VERSION:
            raise ValueError(f"{path}: obsolete split version; rebuild explicitly")
        if splits.get("seed") != seed:
            raise ValueError(
                f"{path}: split seed {splits.get('seed')} != requested {seed}; "
                "use the study split seed or rebuild explicitly"
            )
        if splits.get("dataset_hash") != current["dataset_hash"]:
            raise ValueError(f"{path}: dataset changed; rebuild splits explicitly")
        return splits
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(current, f, indent=2, sort_keys=True)
        f.write("\n")
    return current


def select_windows(windows: list[Window], ids: list[str], *, strict: bool = True) -> list[Window]:
    by_id = {window.window_id: window for window in windows}
    missing = [window_id for window_id in ids if window_id not in by_id]
    if strict and missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"split references {len(missing)} missing windows: {preview}")
    return [by_id[window_id] for window_id in ids if window_id in by_id]


def assert_no_cross_session_leakage(splits: dict) -> None:
    folds = splits.get("within_user_session_folds") or [splits["cross_session"]]
    for fold in folds:
        train = set(fold.get("train_sessions", []))
        val = set(fold.get("val_sessions", []))
        test = set(fold.get("test_sessions", []))
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            raise ValueError(f"{fold.get('fold_id', 'cross_session')}: session leakage {sorted(overlap)}")
        train_null = set(fold.get("train_structured_null_sessions", []))
        dev_null = set(fold.get("development_free_living_sessions", []))
        final_null = set(fold.get("final_test_free_living_sessions", []))
        if train_null & (dev_null | final_null):
            raise ValueError(f"{fold.get('fold_id')}: free-living null leaked into training")
