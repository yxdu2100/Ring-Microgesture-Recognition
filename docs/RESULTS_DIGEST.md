# Canonical results digest

Generated 2026-07-13T02:57:01+00:00 from `ml/results/final`.
The final free-living streams were used only for locked post-training evaluation; all rejection thresholds were fitted on the frozen validation sessions.

## Reproducibility identifiers

- Git commit: `b9d2fc422aaf853b578f4e6ca62a7561ee00c5da` (canonical manifest/split additions may be uncommitted; use the hashes below).
- Dataset hash stored in the split: `0e228d2a3af022e4`.
- Split JSON SHA-256: `c0578071fb80f9cffb1f46127e0510fbc6ef11809e3ee855cdc43edfc78b4f83`.
- Dataset manifest SHA-256: `cfc83b3e02ac2603016ea7786aba05928c55f017786b0398f5cbbb1a7b2a831f`.
- Split guard: all five gesture train/validation/test assignments, structured-null roles, and their window IDs matched the pre-final split byte-for-byte.
- Canonical command: `PYTHONPATH=ml python ml/run_all.py --st-tree-dir ml/st_trees --results-dir ml/results/final --skip-mlc-proxy --skip-hdc-features`.

## Dataset

| Role | Usage | Sessions | Samples | Recorded min | GO markers | Valid windows | Gaps | Missing samples |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gesture | fold | 20 | 827,496 | 114.93 | 1,200 | 1,193 | 0 | 0 |
| free_living_null | development | 1 | 288,861 | 40.12 | 0 | 4,512 | 0 | 0 |
| free_living_null | final_test | 2 | 1,149,928 | 159.71 | 0 | 17,964 | 0 | 0 |
| structured_null | train | 2 | 153,678 | 21.34 | 0 | 2,398 | 0 | 0 |
| structured_null | validation | 1 | 77,468 | 10.76 | 0 | 1,209 | 0 | 0 |

Seven of 1,200 guided gesture windows exceeded the perform interval and were excluded by the frozen rule, leaving 1,193 valid guided windows. The final-test exposure is 159.71 recorded minutes (2.662 h) across two recording parts.

### Session-level record

| Session | Role / usage | Protocol | Samples | Min | GO | Valid windows | Gaps / missing | HW timestamps |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `20260710_001` | gesture / fold | normal | 56,130 | 7.80 | 60 | 58 | 0 / 0 | 100.0% |
| `20260710_002` | gesture / fold | normal | 55,617 | 7.72 | 60 | 60 | 0 / 0 | 100.0% |
| `20260710_003` | gesture / fold | normal | 55,974 | 7.77 | 60 | 60 | 0 / 0 | 100.0% |
| `20260710_004` | gesture / fold | normal | 55,676 | 7.73 | 60 | 59 | 0 / 0 | 100.0% |
| `20260710_005` | gesture / fold | normal | 55,505 | 7.71 | 60 | 58 | 0 / 0 | 100.0% |
| `20260711_001` | gesture / fold | normal | 57,151 | 7.94 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_002` | gesture / fold | normal | 55,763 | 7.74 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_003` | gesture / fold | fast | 33,739 | 4.69 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_004` | gesture / fold | fast | 33,337 | 4.63 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_005` | gesture / fold | fast | 33,354 | 4.63 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_006` | gesture / fold | fast | 34,525 | 4.80 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_007` | gesture / fold | fast | 34,237 | 4.76 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_008` | gesture / fold | fast | 33,427 | 4.64 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_009` | gesture / fold | fast | 33,468 | 4.65 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_010` | gesture / fold | fast | 33,363 | 4.63 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_011` | gesture / fold | fast | 33,229 | 4.62 | 60 | 59 | 0 / 0 | 100.0% |
| `20260711_012` | gesture / fold | fast | 33,440 | 4.64 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_013` | gesture / fold | fast | 32,670 | 4.54 | 60 | 59 | 0 / 0 | 100.0% |
| `20260711_014` | gesture / fold | fast | 33,516 | 4.66 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_015` | gesture / fold | fast | 33,375 | 4.64 | 60 | 60 | 0 / 0 | 100.0% |
| `20260711_019` | free_living_null / development | normal | 288,861 | 40.12 | 0 | 4,512 | 0 / 0 | 100.0% |
| `20260712_001` | free_living_null / final_test | normal | 665,910 | 92.49 | 0 | 10,403 | 0 / 0 | 100.0% |
| `20260712_002` | free_living_null / final_test | normal | 484,018 | 67.22 | 0 | 7,561 | 0 / 0 | 100.0% |
| `20260711_016` | structured_null / train | normal | 85,788 | 11.91 | 0 | 1,339 | 0 / 0 | 100.0% |
| `20260711_017` | structured_null / train | normal | 67,890 | 9.43 | 0 | 1,059 | 0 / 0 | 100.0% |
| `20260711_018` | structured_null / validation | normal | 77,468 | 10.76 | 0 | 1,209 | 0 / 0 | 100.0% |

## Window-level gesture classification

Gesture macro-F1 is calculated over the four gesture classes present in the held-out guided windows; rejection to null is penalized but null has no test support in this table.

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|---:|
| MLC sensor tree | 0.9419 | 0.9131 | 0.9068 | 0.9806 | 0.9829 | 0.9451 ± 0.0360 |
| CNN float | 0.9891 | 0.9590 | 0.9644 | 0.9871 | 0.9914 | 0.9782 ± 0.0153 |
| HDC D=2048 + rejection | 0.6991 | 0.5571 | 0.7185 | 0.7504 | 0.7321 | 0.6914 ± 0.0774 |

## Continuous guided-event evaluation

Recall is pooled across all 1,193 held-out gesture events. Latency is the pooled median onset-to-correct-activation latency; missed and wrong-class events do not have a correct latency.

| Method | M | Correct / events | Event recall | Median latency (ms) |
|---|---:|---:|---:|---:|
| MLC sensor tree | 1 | 1,007 / 1,193 | 0.8441 | 691.7 |
| MLC sensor tree | 2 | 259 / 1,193 | 0.2171 | 1425.0 |
| CNN float | 1 | 980 / 1,193 | 0.8215 | 425.0 |
| CNN float | 2 | 1,063 / 1,193 | 0.8910 | 966.7 |
| HDC D=2048 + rejection | 1 | 603 / 1,193 | 0.5054 | 950.0 |
| HDC D=2048 + rejection | 2 | 217 / 1,193 | 0.1819 | 1283.3 |

CNN M=2 recall can exceed M=1 because consecutive confirmation suppresses early false/wrong activations that otherwise consume the evaluator's one-to-one event match; it does not mean M=2 creates more raw positive windows.

## Free-living false activations per hour

Each fold model is evaluated over the same exposure. Values are per-fold FP/hr followed by mean ± sample SD across the five frozen fold models. Development and final test are never pooled.

### Development stream (0.669 h)

| Method | M | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLC sensor tree | 1 | 32.90 | 32.90 | 38.88 | 41.87 | 62.81 | 41.87 ± 12.33 |
| MLC sensor tree | 2 | 2.99 | 7.48 | 5.98 | 5.98 | 8.97 | 6.28 ± 2.22 |
| CNN float | 1 | 97.21 | 219.84 | 127.12 | 121.14 | 100.20 | 133.10 ± 50.18 |
| CNN float | 2 | 47.86 | 89.73 | 67.30 | 62.81 | 46.36 | 62.81 ± 17.60 |
| HDC D=2048 + rejection | 1 | 927.23 | 655.04 | 825.53 | 870.40 | 1096.22 | 874.88 ± 160.09 |
| HDC D=2048 + rejection | 2 | 424.73 | 266.20 | 418.75 | 445.67 | 595.22 | 430.11 ± 116.75 |

### Final test stream (2.662 h; frozen thresholds)

| Method | M | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLC sensor tree | 1 | 22.16 | 11.65 | 13.52 | 17.28 | 29.30 | 18.78 ± 7.12 |
| MLC sensor tree | 2 | 4.88 | 0.75 | 3.01 | 3.38 | 4.88 | 3.38 ± 1.70 |
| CNN float | 1 | 30.43 | 133.36 | 46.96 | 63.86 | 146.14 | 84.15 ± 52.31 |
| CNN float | 2 | 18.03 | 68.75 | 25.92 | 24.42 | 79.64 | 43.35 ± 28.57 |
| HDC D=2048 + rejection | 1 | 768.26 | 792.68 | 769.76 | 639.40 | 670.58 | 728.13 ± 68.36 |
| HDC D=2048 + rejection | 2 | 370.42 | 381.31 | 380.94 | 355.01 | 326.09 | 362.75 ± 23.12 |

## Feature-HDC representation ablation — Python-only diagnostic

This diagnostic uses tree-style engineered features and has no firmware export or resource claim. It was run before final-test collection; because the non-free-living split is unchanged, its window, guided, and development results remain comparable, but it has no final-test row.

| Window macro-F1 mean ± SD | M | Guided correct / events | Guided recall | Median latency (ms) | Development FP/hr mean ± SD |
|---:|---:|---:|---:|---:|---:|
| 0.9039 ± 0.0386 | 1 | 645 / 1,193 | 0.5407 | 683.3 | 634.70 ± 89.24 |
| 0.9039 ± 0.0386 | 2 | 404 / 1,193 | 0.3386 | 1000.0 | 487.24 ± 43.05 |

## Fold-1 CNN full-int8 gate

Acceptance was decided on validation retention only (required ≥95%); test results were reported after the gate.

| Split | Float macro-F1 | Int8 macro-F1 | Retention | Gate |
|---|---:|---:|---:|---|
| Validation | 0.94225 | 0.94712 | 100.52% | PASS |
| Test | 0.98913 | 0.98913 | 100.00% | Report-only |

## Frozen HDC rejection thresholds

| Fold | Max distance fraction | Min margin fraction | Validation macro-F1 |
|---|---:|---:|---:|
| within_user_01 | 0.17026367 | 0.00830078 | 0.5244 |
| within_user_02 | 0.15371094 | 0.00488281 | 0.4420 |
| within_user_03 | 0.27246094 | 0.00781250 | 0.4074 |
| within_user_04 | 0.21225586 | 0.00830078 | 0.4932 |
| within_user_05 | 0.26806641 | 0.00488281 | 0.4285 |

## Paper figures and system measurements

- Aggregated row-normalized 4×5 confusion panel: [`ml/results/final/figures/primary_methods_confusion_4x5.pdf`](../ml/results/final/figures/primary_methods_confusion_4x5.pdf). Individual PNG/PDF figures and count/normalized CSVs are in the same directory.
- Measured power, flash, static RAM, and measurement limitations: [POWER_MEMORY_RESULTS.md](POWER_MEMORY_RESULTS.md).
- Primary result source files: `fold_window_metrics.csv`, `event_metrics.csv`, `event_matches.csv`, and `chronological_predictions.csv` under `ml/results/final`.
