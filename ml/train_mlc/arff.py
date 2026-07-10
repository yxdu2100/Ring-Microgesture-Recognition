"""Small ARFF reader for MEMS Studio feature exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ArffData:
    relation: str
    attributes: list[str]
    rows: np.ndarray
    labels: list[str]

    @property
    def feature_names(self) -> list[str]:
        return self.attributes[:-1]


def read_arff(path: str | Path) -> ArffData:
    relation = ""
    attributes: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lower = stripped.lower()
        if lower.startswith("@relation"):
            relation = stripped.split(None, 1)[1].strip().strip("'\"")
        elif lower.startswith("@attribute"):
            parts = stripped.split(None, 2)
            attributes.append(parts[1].strip().strip("'\""))
        elif lower.startswith("@data"):
            in_data = True
        elif in_data:
            data_rows.append([item.strip() for item in next(csv.reader([stripped]))])
    if not attributes:
        raise ValueError(f"{path}: no ARFF attributes found")
    if not data_rows:
        raise ValueError(f"{path}: no ARFF data rows found")
    values = np.array([[float(item) for item in row[:-1]] for row in data_rows], dtype=np.float64)
    labels = [row[-1] for row in data_rows]
    return ArffData(relation=relation, attributes=attributes, rows=values, labels=labels)
