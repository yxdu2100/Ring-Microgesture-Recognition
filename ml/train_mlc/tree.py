"""Decision-tree stand-in for the sensor MLC and software M33 baseline."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from eval_utils import prediction_report
from ringdata import CLASS_NAMES, load_sessions, segment_sessions
from ringdata.splits import assert_no_cross_session_leakage, build_or_load_splits, select_windows
from train_mlc.features import featurize

SEED = 20260706


def _git_hash(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def train_tree(windows, splits: dict, split_type: str = "cross_session"):
    split = splits[split_type]
    train_w = select_windows(windows, split["train"])
    test_w = select_windows(windows, split["test"])
    if not train_w or not test_w:
        raise ValueError(
            f"{split_type}: need non-empty train and test windows; "
            f"got train={len(train_w)} test={len(test_w)}"
        )
    x_train, y_train, names = featurize(train_w)
    x_test, y_test, _ = featurize(test_w)
    clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return clf, names, y_test, pred


def _c_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def export_tree_header(clf, feature_names: list[str], out_path: Path, data_hash: str, macro_f1: float) -> None:
    tree = clf.tree_

    def emit_node(node: int, indent: str) -> list[str]:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            values = tree.value[node][0]
            cls = int(np.argmax(values))
            return [f"{indent}return {cls};"]
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        lines = [f"{indent}if (f[{feature}] <= {threshold:.9g}f) {{"]
        lines.extend(emit_node(int(tree.children_left[node]), indent + "\t"))
        lines.append(f"{indent}}} else {{")
        lines.extend(emit_node(int(tree.children_right[node]), indent + "\t"))
        lines.append(f"{indent}}}")
        return lines

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("#ifndef GENERATED_TREE_SW_H\n#define GENERATED_TREE_SW_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// data_git_hash: {data_hash}\n")
        f.write(f"// cross_session_macro_f1: {macro_f1:.6f}\n")
        f.write(f"#define TREE_SW_FEATURE_COUNT {len(feature_names)}U\n")
        f.write("#define TREE_SW_MAX_DEPTH 6U\n\n")
        f.write("static const char *const tree_sw_feature_names[TREE_SW_FEATURE_COUNT] = {\n")
        for name in feature_names:
            f.write(f'\t"{_c_name(name)}",\n')
        f.write("};\n\n")
        f.write("static inline int tree_sw_predict(const float f[TREE_SW_FEATURE_COUNT])\n{\n")
        for line in emit_node(0, "\t"):
            f.write(line + "\n")
        f.write("}\n\n#endif\n")


def _write_metrics(results_dir: Path, report: dict, split_type: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"mlc_tree_{split_type}_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "rate_hz", "split_type", "macro_f1", "top_predicted_class", "top_predicted_fraction"])
        writer.writerow([
            "mlc_tree",
            120,
            split_type,
            f"{report['macro_f1']:.6f}",
            report["top_predicted_class"],
            f"{report['top_predicted_fraction']:.6f}",
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="ml/splits.json")
    parser.add_argument("--results-dir", default="ml/results")
    parser.add_argument("--header", default="firmware/src/classifiers/generated/tree_sw.h")
    parser.add_argument("--split-type", default="cross_session", choices=["cross_session", "within_session"])
    args = parser.parse_args()

    sessions = load_sessions(args.data_dir)
    windows = segment_sessions(sessions)
    splits = build_or_load_splits(windows, args.splits, seed=SEED)
    assert_no_cross_session_leakage(splits)
    clf, names, y_test, pred = train_tree(windows, splits, split_type=args.split_type)
    report = prediction_report(y_test, pred, "mlc_tree", 120, args.split_type, args.results_dir)
    _write_metrics(Path(args.results_dir), report, args.split_type)
    export_tree_header(clf, names, Path(args.header), _git_hash(Path(".")), report["macro_f1"])
    print(f"mlc_tree {args.split_type} macro_f1={report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
