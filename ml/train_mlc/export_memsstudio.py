"""Export train-split windows as MEMS Studio-friendly tab-delimited text."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.splits import build_or_load_splits, select_windows


def export_windows(windows, splits: dict, out_dir: Path) -> int:
    train_ids = splits["cross_session"]["train"]
    selected = select_windows(windows, train_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
    for window in selected:
        path = out_dir / window.label / f"{window.window_id.replace(':', '_')}.txt"
        np.savetxt(path, window.raw, fmt="%d", delimiter="\t", header="ax\tay\taz\tgx\tgy\tgz", comments="")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--out-dir", default="ml/results/memsstudio_export")
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions)
    splits = build_or_load_splits(windows, args.splits)
    n = export_windows(windows, splits, Path(args.out_dir))
    print(f"exported {n} train windows to {args.out_dir}")


if __name__ == "__main__":
    main()
