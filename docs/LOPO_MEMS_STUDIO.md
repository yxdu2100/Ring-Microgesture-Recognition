# Four-participant LOPO MLC workflow

This workflow creates four different MLC trees. Each tree excludes one
participant completely and is evaluated only on that held-out participant.

| Fold | MEMS Studio training participants | Held-out test participant | Tree filename |
|---|---|---|---|
| `lopo_01` | P2, P3, P4 | P1 | `lopo_01.txt` |
| `lopo_02` | P1, P3, P4 | P2 | `lopo_02.txt` |
| `lopo_03` | P1, P2, P4 | P3 | `lopo_03.txt` |
| `lopo_04` | P1, P2, P3 | P4 | `lopo_04.txt` |

The held-out participant must not be imported into MEMS Studio for that fold.
MEMS Studio's training accuracy is not the LOPO result; the paper result comes
from replaying the exported tree on the held-out participant in Python.

## 1. Check the new sessions

Activate the project environment and check P4's two recordings:

```sh
conda activate ring-microgesture
PYTHONPATH=ml python ml/check_sessions.py \
  --manifest ml/dataset_manifest_lopo.csv \
  --session 20260713_005 \
  --session 20260713_006
```

The final LOPO manifest is `ml/dataset_manifest_lopo.csv`. It includes P1--P4;
do not use the within-user manifest for this analysis.

## 2. Generate the exact MEMS Studio training files

Run:

```sh
PYTHONPATH=ml python ml/run_lopo.py \
  --manifest ml/dataset_manifest_lopo.csv \
  --export-mlc-dir ml/results/memsstudio_lopo_export \
  --export-mlc-only
```

This produces four folders:

```text
ml/results/memsstudio_lopo_export/
  lopo_01/
  lopo_02/
  lopo_03/
  lopo_04/
```

Each folder contains only its training fold and has five class files:

```text
double_side_tap.csv
double_pinch.csv
pinch_hold.csv
double_flick.csv
null.csv
```

The exporter first balances the contribution of the non-held participants and
then exports equal counts for the five MLC classes. The validation windows and
all windows from the held participant are absent. `split_window_ids.json` in
each folder records the exact split and settings.

For the current four-participant dataset, the verified export contains 33
training windows per class (165 windows total) in every fold.

## 3. Train four projects in MEMS Studio

The safest procedure is to duplicate the existing within-user MEMS Studio
project four times. This preserves the feature candidates, window settings,
tree settings, and output labels already used for the paper. In each copy,
replace only the five training datalogs with the five CSV files from one LOPO
folder.

Keep these settings fixed in every project:

- Sensor: LSM6DSV16X.
- Sampling frequency: 120 Hz.
- Accelerometer full scale: $\pm 8$ g.
- Gyroscope full scale: $\pm 2000$ dps.
- Window length: 128 samples.
- MLC window advance: one complete, non-overlapping feature window.
- Five labels: `double_side_tap`, `double_pinch`, `pinch_hold`,
  `double_flick`, and `null`.
- The same candidate feature menu used by the existing within-user project.
- Decision-tree maximum depth: 6.
- The same class-balancing and optimizer settings for all four folds.

Do not retune the feature menu or tree depth separately after inspecting a
held-out participant. Any setting change must be applied identically to all
four training projects without using held-out results for selection.

For each project, export the complete decision-tree report as a `.txt` file.
The report must contain the tree rules plus the `Classes:` and `Features:`
sections, like the existing files in `ml/st_trees/`. A `.ucf` or generated
header is useful only for deployment and is not required for the offline LOPO
table.

## 4. Place the exported tree reports

Create this directory and place the reports using exactly these names:

```text
ml/st_trees_lopo/lopo_01.txt
ml/st_trees_lopo/lopo_02.txt
ml/st_trees_lopo/lopo_03.txt
ml/st_trees_lopo/lopo_04.txt
```

The runner also accepts names such as
`ST_decision_tree_lopo_01.txt`, but the short names above are less error-prone.

## 5. Run the final four-participant comparison

After all four reports are present, run:

```sh
PYTHONPATH=ml python ml/run_lopo.py \
  --manifest ml/dataset_manifest_lopo.csv \
  --st-tree-dir ml/st_trees_lopo \
  --skip-mlc-proxy \
  --results-dir ml/results/lopo_p1_p4_final
```

This command trains the CNN and HDC for the same four outer LOPO folds and
evaluates the four exported MLC trees. It aborts if any expected tree is
missing. Paper-ready window results are written to:

```text
ml/results/lopo_p1_p4_final/fold_window_metrics.csv
ml/results/lopo_p1_p4_final/summary.csv
ml/results/lopo_p1_p4_final/summary.md
```

Use the `mlc_sensor_tree_lopo` row in the paper. Do not substitute the
`mlc_proxy_tree_lopo_diagnostic` row.
