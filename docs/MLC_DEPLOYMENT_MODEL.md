# Final MLC deployment model

This model is for live app feedback and is separate from every tree reported in
the paper. Do not replace the paper result files with this deployment tree.

## Prepared MEMS Studio datasets

Three deterministic candidate datasets are under:

```text
ml/results/memsstudio_deployment_candidates/
  null_1x/
  null_2x/
  null_4x/
```

Each candidate contains the same 370 windows for each target gesture, drawn
from all six participants. The candidates differ only in the null count:

| Candidate | Windows per gesture | Null windows | Intended use |
|---|---:|---:|---|
| `null_1x` | 370 | 370 | Accuracy/recall-oriented reference |
| `null_2x` | 370 | 740 | Recommended first deployment candidate |
| `null_4x` | 370 | 1,480 | Aggressive false-trigger reduction |

Null windows are distributed evenly across three P1 structured-null sessions
and the five P2--P6 free-living sessions. The P1 development and final-test
free-living recordings are not used for training, so they remain available for
candidate selection and confirmation.

## Train the three candidates

Create three copies of the same MEMS Studio MLC project. For each project,
import only the five CSV files from one matching candidate folder:

```text
double_side_tap.csv
double_pinch.csv
pinch_hold.csv
double_flick.csv
null.csv
```

Keep the settings used by the paper trees:

- LSM6DSV16X, 120 Hz;
- accelerometer +/-8 g and gyroscope +/-2000 dps;
- 128-sample, non-overlapping feature windows;
- the same sensor-supported feature menu;
- maximum tree depth 6;
- the same five class labels and output codes.

Do not enable automatic class resampling or class balancing: it would erase
the intended 1x/2x/4x null ratio. In the complete training report, verify that
the null support is approximately 1x, 2x, or 4x the support of each gesture.
MEMS Studio may discard one final feature window from each concatenated file,
so an off-by-one difference is expected.

Export the complete decision-tree reports as:

```text
ml/st_trees_deployment/null_1x.txt
ml/st_trees_deployment/null_2x.txt
ml/st_trees_deployment/null_4x.txt
```

The reports must include the tree rules and the `Classes:` and `Features:`
sections. Export the UCF only after selecting the candidate.

## Select and deploy

Use P1 development free-living session `20260711_019` to compare FP/hr for the
three exported trees. Select the lowest-FP candidate that still recognizes all
four gestures reliably. Confirm the selected tree once on final-test sessions
`20260712_001` and `20260712_002`, then export its UCF and deploy it to the
ring. A short live scripted check should include 15 repetitions per gesture
plus typing, walking, and object manipulation; do not choose `null_4x` solely
because it predicts null most often.

The default starting choice is `null_2x`. `null_4x` is preferable only if its
gesture recall remains acceptable in the live check.

## Reproduce the export

```sh
PYTHONPATH=ml python3 ml/train_mlc/export_deployment_candidates.py
```

Each candidate folder contains `export_manifest.json`; the root contains
`export_summary.json` with source counts and exclusions.
