"""Simulate on-device HDC enrollment from a later held-out session."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_utils import macro_f1_present_classes
from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from train_hdc.encode import encode_window, make_codebooks
from train_hdc.train import predict_hdc, train_hdc, _signed


def enrollment_curve(windows, out_csv: Path, dim: int = 2048) -> list[dict]:
    sessions = sorted({w.session_id for w in windows})
    if len(sessions) < 2:
        raise ValueError("enrollment simulation needs at least two sessions")
    codebooks = make_codebooks(dim=dim)
    rows = []
    for k in range(1, len(sessions)):
        base_sessions = set(sessions[:k])
        held_session = sessions[k]
        base_w = [w for w in windows if w.session_id in base_sessions]
        held_w = [w for w in windows if w.session_id == held_session]
        by_class = defaultdict(list)
        for w in held_w:
            by_class[w.class_id].append(w)
        for n in (1, 3, 5, 10):
            memories = train_hdc(base_w, codebooks)
            for cls, class_windows in by_class.items():
                for w in class_windows[:n]:
                    memories[cls] += _signed(encode_window(w.raw, codebooks))
            y_true, y_pred = predict_hdc(held_w, memories, codebooks)
            acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
            macro_f1, macro_f1_all, *_ = macro_f1_present_classes(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            rows.append(
                {
                    "base_sessions": k,
                    "held_session": held_session,
                    "examples_per_class": n,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "macro_f1_all_classes": macro_f1_all,
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="ml/results/hdc/enroll_curve.csv")
    args = parser.parse_args()
    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions)
    rows = enrollment_curve(windows, Path(args.out))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
