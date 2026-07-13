# Measured system cost — benchmark builds (BLE off)

Conditions: PPK2 source-meter mode at **4.2 V**, 15 s warm-up, one 60 s trace per
build per condition (single-trial; no ±SD is claimed). Firmware: benchmark
variants with `CONFIG_BT=n`, logging/console/shell disabled, tickless System ON
idle. MLC build runs the deployed **fold within_user_01 MEMS Studio tree**
(`ml/st_trees/within_user_01.h` installed as `firmware/src/modules/mlc.h`).

## Power (mean current over 60 s)

| Build | Motionless (mA) | Scripted-active 12/min (mA) | Power @4.2 V (mW) | Δ vs NONE (mW) |
|---|---:|---:|---:|---:|
| NONE (sense + FIFO drain, no classifier) | 1.86 | 1.86 | 7.81 | — |
| MLC (in-sensor tree, interrupt wake) | 1.58 | 1.59 | 6.64 | **−1.18** |
| HDC D=2048 (MCU, every 64 samples) | 3.34 | 3.34 | 14.03 | +6.22 |
| CNN int8 (MCU TFLM, every 64 samples) | 4.90 | 4.92 | 20.58 | +12.77 |

Notes:
- Power is activity-independent for all four builds (≤0.02 mA difference
  between motionless and scripted-active). CNN/HDC run a fixed inference
  cadence regardless of motion; MLC interrupt-service cost is negligible.
- MLC's **negative** delta is a deployment-path effect: classification moves
  into the IMU *and* the MCU stops draining the 120 Hz FIFO. It is not the
  isolated cost of adding a classifier.
- Observed peaks: ~4.0 mA (MLC active), ~7.1 mA (HDC encode bursts). Headline
  metric is the 60 s mean.
- Illustrative battery life at a nominal 40 mAh ring cell (h = 40/I):
  NONE 21.5 h, MLC 25.3 h, HDC 12.0 h, CNN 8.2 h. Label as an estimate at a
  fixed 4.2 V rail.

## Flash / RAM (linker totals, benchmark builds)

| Build | Flash (B) | Δ vs NONE | Static RAM (B) | Δ vs NONE | Main additions |
|---|---:|---:|---:|---:|---|
| NONE | 63,412 | — | 27,288 | — | Zephyr + drivers + FIFO path |
| MLC | 63,440 | **+28** | 21,184 | **−6,104** | UCF register table; window buffers removed |
| HDC | 76,228 | +12,816 | 41,208 | +13,920 | codebooks + prototypes (~11.5 KB const) |
| CNN | 170,568 | +107,156 | 75,568 | +48,280 | model 21,728 B + TFLM/CMSIS-NN runtime ≈70 KB; 40,960 B tensor arena (RAM) |

CNN flash breakdown (rom_report, build-benchmark-cnn): `g_cnn_model` weights
21,728 B; TFLM + CMSIS-NN library code ≈ 70 KB; remainder is the shared
Zephyr/driver/application base also present in NONE.

## Latency (derived cadence + measured event latency; no per-inference cycle counts)

Per-inference execution time was not measured with cycle counters. The
reported latency is (a) architectural decision cadence and (b) the measured
onset-to-confirmed-activation latency from the continuous evaluation
(median over held-out guided sessions, mean across the five folds).

| Method | Decision cadence | Event latency M=1 (ms) | Event latency M=2 (ms) | M=2 added delay |
|---|---:|---:|---:|---:|
| MLC sensor tree | 128 samples ≈ 1.07 s | 691.7 | 1,425.0 | +1 feature window ≈ 1.07 s |
| CNN float evaluation | 64-sample hop ≈ 0.53 s | 425.0 | 966.7 | +1 hop ≈ 0.53 s |
| HDC D=2048 | 64-sample hop ≈ 0.53 s | 950.0 | 1,283.3 | +1 hop ≈ 0.53 s |

(Pooled median correct-event latencies from the canonical
`ml/results/final/event_matches.csv`. The CNN timing row uses float-model
predictions; the fold-1 full-int8 gate matched its float test macro-F1, but a
five-fold int8 continuous-stream equivalence test was not run.)

## Verification status

- All four builds compile with BLE fully excluded (zero BT symbols in map).
- MLC execution path confirmed in code: no FIFO drain; MCU blocks on the MLC
  interrupt (System ON idle with interrupt wake — **not** "deep sleep").
- MLC live output: verified over BLE against the parsed tree — **re-verify
  once with the fold-1 tree now deployed** if the previous BLE check used the
  older 2026-07-09 tree.
