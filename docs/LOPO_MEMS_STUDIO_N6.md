# Six-participant LOPO MLC workflow

The final six-participant analysis requires six newly trained MEMS Studio
trees. The earlier four-participant trees are not valid because adding P5 and
P6 changes the training data for folds 01--04.

| Fold | Train in MEMS Studio | Held-out test participant | Save report as |
|---|---|---|---|
| `lopo_01` | P2, P3, P4, P5, P6 | P1 | `lopo_01.txt` |
| `lopo_02` | P1, P3, P4, P5, P6 | P2 | `lopo_02.txt` |
| `lopo_03` | P1, P2, P4, P5, P6 | P3 | `lopo_03.txt` |
| `lopo_04` | P1, P2, P3, P5, P6 | P4 | `lopo_04.txt` |
| `lopo_05` | P1, P2, P3, P4, P6 | P5 | `lopo_05.txt` |
| `lopo_06` | P1, P2, P3, P4, P5 | P6 | `lopo_06.txt` |

## Prepared training folders

Use only the new exports under:

```text
ml/results/memsstudio_lopo_export_n6/
```

Each `lopo_XX` folder contains five class-balanced files:

```text
double_side_tap.csv
double_pinch.csv
pinch_hold.csv
double_flick.csv
null.csv
```

Every fold contains 55 windows per class (275 total). The held-out participant
and all validation windows are absent from these files. The corresponding
`split_window_ids.json` records the exact participant assignment and window
IDs.

## MEMS Studio procedure

Duplicate the same frozen MEMS Studio project six times and replace only its
training datalogs. For project `lopo_XX`, import the five CSVs from the matching
folder. Keep all settings unchanged:

- LSM6DSV16X at 120 Hz;
- accelerometer $\pm8$ g and gyroscope $\pm2000$ dps;
- 128-sample, non-overlapping feature windows;
- the existing feature candidate menu;
- maximum tree depth 6;
- the existing optimizer and class settings;
- the same five output labels.

Do not use the held-out participant to select features, depth, or any other
setting. MEMS Studio training accuracy is not the paper result.

Export the complete decision-tree text report from each project. The file must
contain the tree rules plus the `Classes:` and `Features:` sections. Place the
six new reports in:

```text
ml/st_trees_lopo_n6/lopo_01.txt
ml/st_trees_lopo_n6/lopo_02.txt
ml/st_trees_lopo_n6/lopo_03.txt
ml/st_trees_lopo_n6/lopo_04.txt
ml/st_trees_lopo_n6/lopo_05.txt
ml/st_trees_lopo_n6/lopo_06.txt
```

Do not copy or rename the previous four-participant trees into this directory.

## Final run after all six reports are present

```sh
conda activate ring-microgesture
PYTHONPATH=ml python ml/run_lopo.py \
  --manifest ml/dataset_manifest_lopo.csv \
  --st-tree-dir ml/st_trees_lopo_n6 \
  --skip-mlc-proxy \
  --results-dir ml/results/lopo_p1_p6_final
```

This reruns all six CNN and HDC folds and evaluates the six actual MLC trees.
The paper should be updated only after this command completes, because every
LOPO mean and the previous P1--P4 values can change.

## Data-quality note

P5's guided and null recordings pass the automated checks. P6's null recording
passes; its guided recording has all 15 repetitions per gesture, no gaps, and
no missing samples, but 35% of instances require onset re-anchoring and one
instance exceeds the guided perform interval. The frozen preprocessing excludes
that one overlong instance, leaving 59 guided examples. Retain P6 for the
exploratory LOPO analysis and disclose the small, uneven participant dataset as
a limitation.
