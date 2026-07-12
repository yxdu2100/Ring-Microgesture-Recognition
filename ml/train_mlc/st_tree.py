"""Parse and evaluate MEMS Studio exported decision-tree text files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ringdata.segment import CLASS_TO_ID
from train_mlc.features import MEMS_STUDIO_GYRO_INTERNAL_PER_LSB, mlc_feature_dict

ST_TO_PROJECT_CLASS = {
    "sidetap": "double_side_tap",
    "doublepinch": "double_pinch",
    "pinchhold": "pinch_hold",
    "flicker": "double_flick",
    "double_side_tap": "double_side_tap",
    "double_pinch": "double_pinch",
    "pinch_hold": "pinch_hold",
    "double_flick": "double_flick",
    "null": "null",
}
PROJECT_TO_ST_CLASS = {v: k for k, v in ST_TO_PROJECT_CLASS.items()}

COND_RE = re.compile(r"^(?P<feature>F\d+_[A-Z0-9_]+)\s+(?P<op><=|>)\s+(?P<threshold>-?\d+(?:\.\d+)?)")
LEAF_RE = re.compile(r":\s*(?P<label>[A-Za-z0-9_]+)\s*\((?P<count>\d+)(?:/(?P<errors>[\d.]+))?\)")


@dataclass(frozen=True)
class STCondition:
    feature: str
    op: str
    threshold: float

    def matches(self, features: dict[str, np.float16 | np.float64], precision: str) -> bool:
        value = features[self.feature]
        if precision == "fp16":
            lhs = np.float16(value)
            rhs = np.float16(self.threshold)
        elif precision == "fp64":
            lhs = np.float64(value)
            rhs = np.float64(self.threshold)
        else:
            raise ValueError(f"unknown precision: {precision}")
        if self.op == "<=":
            return bool(lhs <= rhs)
        if self.op == ">":
            return bool(lhs > rhs)
        raise ValueError(f"unsupported operator: {self.op}")


@dataclass(frozen=True)
class STRule:
    conditions: tuple[STCondition, ...]
    st_label: str
    count: int
    errors: float

    @property
    def project_label(self) -> str:
        try:
            return ST_TO_PROJECT_CLASS[self.st_label]
        except KeyError as exc:
            raise ValueError(f"unknown ST class label: {self.st_label}") from exc

    @property
    def class_id(self) -> int:
        return CLASS_TO_ID[self.project_label]


@dataclass
class STTrainingStats:
    class_order: list[str]
    confusion: np.ndarray | None
    total: int | None
    correct: int | None
    incorrect: int | None
    accuracy: float | None


class MLCTreeClassifier:
    """Fixed MEMS Studio tree with device-faithful fp16 comparisons."""

    def __init__(
        self,
        rules: list[STRule],
        feature_tokens: list[str],
        precision: str = "fp16",
        accel_fs_g: int = 8,
        gyro_fs_dps: int = 2000,
        accel_lsb_scale: float | None = None,
        gyro_lsb_scale: float | None = MEMS_STUDIO_GYRO_INTERNAL_PER_LSB,
    ):
        self.rules = rules
        self.feature_tokens = feature_tokens
        self.precision = precision
        self.accel_fs_g = accel_fs_g
        self.gyro_fs_dps = gyro_fs_dps
        self.accel_lsb_scale = accel_lsb_scale
        self.gyro_lsb_scale = gyro_lsb_scale

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        precision: str = "fp16",
        accel_fs_g: int = 8,
        gyro_fs_dps: int = 2000,
        accel_lsb_scale: float | None = None,
        gyro_lsb_scale: float | None = MEMS_STUDIO_GYRO_INTERNAL_PER_LSB,
    ) -> "MLCTreeClassifier":
        parsed = parse_st_tree(path)
        return cls(
            parsed["rules"],
            parsed["feature_tokens"],
            precision=precision,
            accel_fs_g=accel_fs_g,
            gyro_fs_dps=gyro_fs_dps,
            accel_lsb_scale=accel_lsb_scale,
            gyro_lsb_scale=gyro_lsb_scale,
        )

    def clf_init(self) -> None:
        return None

    def clf_process_window(self, window) -> int:
        return self.predict_raw(window.raw)

    def predict_raw(self, raw: np.ndarray) -> int:
        features = mlc_feature_dict(
            raw,
            self.feature_tokens,
            accel_fs_g=self.accel_fs_g,
            gyro_fs_dps=self.gyro_fs_dps,
            precision=self.precision,
            accel_lsb_scale=self.accel_lsb_scale,
            gyro_lsb_scale=self.gyro_lsb_scale,
        )
        for rule in self.rules:
            if all(condition.matches(features, self.precision) for condition in rule.conditions):
                return rule.class_id
        raise RuntimeError("ST tree evaluation reached no leaf; parser produced incomplete rules")

    def predict_windows(self, windows) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.array([w.class_id for w in windows], dtype=np.int64)
        y_pred = np.array([self.clf_process_window(w) for w in windows], dtype=np.int64)
        return y_true, y_pred


def _line_depth(line: str) -> tuple[int, str]:
    depth = 0
    rest = line
    while rest.startswith("|   "):
        depth += 1
        rest = rest[4:]
    return depth, rest.strip()


def _parse_condition(text: str) -> STCondition | None:
    match = COND_RE.search(text)
    if match is None:
        return None
    return STCondition(match.group("feature"), match.group("op"), float(match.group("threshold")))


def _parse_leaf(text: str) -> tuple[str, int, float] | None:
    match = LEAF_RE.search(text)
    if match is None:
        return None
    return match.group("label"), int(match.group("count")), float(match.group("errors") or 0.0)


def _parse_features(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if line.strip() == "Features:" and idx + 1 < len(lines):
            text = lines[idx + 1].strip()
            if text.startswith("=>"):
                return [item.strip() for item in text[2:].split(",") if item.strip()]
    tokens = []
    for line in lines:
        for match in re.finditer(r"F\d+_[A-Z0-9_]+", line):
            token = match.group(0)
            if token not in tokens:
                tokens.append(token)
    return tokens


def _parse_classes(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if line.strip() == "Classes:" and idx + 1 < len(lines):
            text = lines[idx + 1].strip()
            if text.startswith("=>"):
                return [item.strip() for item in text[2:].split(",") if item.strip()]
    return []


def _parse_training_stats(lines: list[str], class_order: list[str]) -> STTrainingStats:
    total = correct = incorrect = None
    accuracy = None
    confusion = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Total Number of Instances:"):
            total = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Correctly Classified Instances:"):
            correct = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Incorrectly Classified Instances:"):
            incorrect = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Accuracy:"):
            accuracy = float(stripped.split(":", 1)[1].replace("%", "").strip())
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Confusion Matrix:")
    except StopIteration:
        start = -1
    if start >= 0 and class_order:
        rows = []
        for line in lines[start + 2 : start + 2 + len(class_order)]:
            parts = line.split()
            if len(parts) >= len(class_order) + 1:
                rows.append([int(float(x)) for x in parts[1 : 1 + len(class_order)]])
        if len(rows) == len(class_order):
            confusion = np.array(rows, dtype=np.int64)
    return STTrainingStats(class_order, confusion, total, correct, incorrect, accuracy)


def parse_st_tree(path: str | Path) -> dict:
    lines = Path(path).read_text().splitlines()
    rules: list[STRule] = []
    stack: list[STCondition | None] = []
    for raw_line in lines:
        if not raw_line.strip() or raw_line.startswith("=") or raw_line.startswith("Number of Leaves:"):
            continue
        if raw_line.strip() in {"Classes:", "Features:", "Confusion Matrix:", "Report:"}:
            continue
        depth, text = _line_depth(raw_line)
        condition = _parse_condition(text)
        if condition is None:
            continue
        while len(stack) <= depth:
            stack.append(None)
        stack[depth] = condition
        del stack[depth + 1 :]
        leaf = _parse_leaf(text)
        if leaf is not None:
            label, count, errors = leaf
            conditions = tuple(c for c in stack[: depth + 1] if c is not None)
            rules.append(STRule(conditions, label, count, errors))
    class_order = _parse_classes(lines)
    return {
        "rules": rules,
        "feature_tokens": _parse_features(lines),
        "class_order": class_order,
        "training_stats": _parse_training_stats(lines, class_order),
    }
