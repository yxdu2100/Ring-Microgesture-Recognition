"""Dataset-role manifest for reproducible study splits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .parse import Session


VALID_ROLES = {"gesture", "structured_null", "free_living_null"}
VALID_USAGE = {"fold", "train", "validation", "development", "final_test", "exclude", "auto"}


@dataclass(frozen=True)
class ManifestEntry:
    session_id: str
    participant_id: str
    role: str
    usage: str
    include: bool
    notes: str = ""


def load_manifest(path: str | Path) -> dict[str, ManifestEntry]:
    path = Path(path)
    if not path.exists():
        return {}
    entries: dict[str, ManifestEntry] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            session_id = row.get("session_id", "").strip()
            if not session_id:
                continue
            role = row.get("role", "").strip()
            usage = row.get("usage", "auto").strip() or "auto"
            if role not in VALID_ROLES:
                raise ValueError(f"{path}: {session_id} has invalid role {role!r}")
            if usage not in VALID_USAGE:
                raise ValueError(f"{path}: {session_id} has invalid usage {usage!r}")
            include = row.get("include", "true").strip().lower() not in {"0", "false", "no"}
            if session_id in entries:
                raise ValueError(f"{path}: duplicate session_id {session_id}")
            entries[session_id] = ManifestEntry(
                session_id=session_id,
                participant_id=row.get("participant_id", "").strip() or "unknown",
                role=role,
                usage=usage,
                include=include,
                notes=row.get("notes", "").strip(),
            )
    return entries


def apply_manifest(
    sessions: list[Session],
    path: str | Path = "ml/dataset_manifest.csv",
) -> tuple[list[Session], list[str]]:
    """Apply explicit roles and safely infer roles for newly collected data."""
    entries = load_manifest(path)
    warnings: list[str] = []
    included: list[Session] = []
    has_free_living_development = any(
        entry.role == "free_living_null" and entry.usage == "development"
        for entry in entries.values()
    )
    seen: set[str] = set()
    for session in sessions:
        if session.session_id in seen:
            raise ValueError(f"duplicate active session_id {session.session_id}")
        seen.add(session.session_id)
        entry = entries.get(session.session_id)
        if entry is not None:
            if entry.participant_id not in {"unknown", session.participant_id}:
                raise ValueError(
                    f"manifest participant mismatch for {session.session_id}: "
                    f"{entry.participant_id} != {session.participant_id}"
                )
            session.data_role = entry.role
            session.usage = entry.usage
            if entry.include and entry.usage != "exclude":
                included.append(session)
            continue

        if session.data_role == "gesture":
            session.usage = "fold"
        elif session.data_role == "structured_null":
            session.usage = "train"
        elif session.data_role == "free_living_null":
            session.usage = "final_test" if has_free_living_development else "development"
            has_free_living_development = True
        else:
            warnings.append(
                f"excluding unmanifested session {session.session_id} with role {session.data_role!r}"
            )
            continue
        warnings.append(
            f"session {session.session_id} is not in {path}; inferred "
            f"role={session.data_role} usage={session.usage}"
        )
        included.append(session)

    missing = sorted(set(entries) - seen)
    if missing:
        warnings.append(f"manifest entries not present on disk: {', '.join(missing)}")
    return included, warnings

