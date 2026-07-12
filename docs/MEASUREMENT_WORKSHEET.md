# Final system measurement worksheet

Use this sheet for the paper's primary resource comparison. Do not change ODR,
windowing, model capacity, or generated assets between repetitions.

## Frozen settings

| Method | Build directory | Configuration | IMU ODR | Accel / gyro range | Window / hop | Frozen classifier asset | BLE / logs | Measure tomorrow? |
|---|---|---|---:|---|---|---|---|---|
| NONE sensing baseline | `build-benchmark-none` | `overlay-none.conf;overlay-benchmark.conf` | 120 Hz | +/-8 g / +/-2000 dps | Raw FIFO drain/parse only; no classifier | None | BLE excluded (`CONFIG_BT=n`); logs disabled | Yes |
| Actual in-sensor MLC | `build-benchmark-mlc` | `overlay-mlc.conf;overlay-benchmark.conf` | 120 Hz | +/-8 g / +/-2000 dps | Current MEMS Studio/UCF cadence; MCU does not drain FIFO | Current verified `mlc.h`/UCF; record its hash | BLE excluded (`CONFIG_BT=n`); logs disabled | Yes |
| CNN | `build-benchmark-cnn` | `overlay-cnn.conf;overlay-benchmark.conf` | 120 Hz | +/-8 g / +/-2000 dps | 128 / 64 samples | Accepted int8 model, 21,728 bytes; record header hash | BLE excluded (`CONFIG_BT=n`); logs disabled | Yes |
| HDC primary | `build-benchmark-hdc` | `overlay-hdc.conf;overlay-benchmark.conf` | 120 Hz | +/-8 g / +/-2000 dps | 128 / 64 samples | D=2048, 32 levels, baseline flat-update export; record header hash | BLE excluded (`CONFIG_BT=n`); logs disabled | Yes |

Do **not** measure D=8192, the 64-level HDC diagnostic, or the rejected
phase/scaled HDC ablation. They are not primary configurations. M=1 and M=2 are
event-output policies, not different CNN/HDC inference workloads, so they do not
need separate CNN/HDC power builds. If M=2 is implemented inside a different MLC
UCF, treat it as a separate optional MLC build and label it explicitly.

## Build record

| Method | Git commit | Generated asset hash | Build completed? | Flash image / ELF path | Notes or warnings |
|---|---|---|---|---|---|
| NONE |  | N/A |  |  |  |
| MLC |  |  |  |  |  |
| CNN int8 |  |  |  |  |  |
| HDC D=2048 |  |  |  |  |  |

Use pristine builds. In an nRF Connect SDK terminal:

```sh
west build -p always -d build-benchmark-none -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-none.conf;overlay-benchmark.conf"
west build -p always -d build-benchmark-mlc  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-mlc.conf;overlay-benchmark.conf"
west build -p always -d build-benchmark-cnn  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-cnn.conf;overlay-benchmark.conf"
west build -p always -d build-benchmark-hdc  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-hdc.conf;overlay-benchmark.conf"
```

## PPK2 setup record

| Field | Value |
|---|---|
| PPK2 mode | Source meter / ampere meter: |
| Supply voltage |  V |
| PPK2 sample rate |  |
| Ring battery disconnected? | Yes / No |
| Startup settling time | 15 s minimum; actual: |
| Trace duration | 60 s minimum; actual: |
| Firmware commit |  |
| Date / room conditions |  |

Never power the ring from the battery and PPK2 simultaneously.

## Power traces

Collect **both** conditions for all four builds. Motionless is a legitimate idle
operating point, but it cannot alone support a general “average power” claim:
the MLC MCU is interrupt-driven, whereas CNN/HDC run at a fixed cadence.

- **Motionless:** ring fixed on the table for 60 seconds.
- **Scripted active:** 60 seconds, one prompted gesture every 5 seconds,
  cycling `double_side_tap`, `double_pinch`, `pinch_hold`, and `double_flick`
  three times. Use the same metronome/audio script, hand, fit, and order for all
  builds. Human motion will not be perfectly identical, so report the protocol,
  not “identical motion.”

Run three traces per method/condition. For MLC, record the change in
`g_classifier_benchmark_stats.inference_count` during every trace and report
interrupt/service events per minute. Record the same count for CNN/HDC as a
cadence sanity check.

| Method | Condition | Trial 1 mean current (mA) | Trial 2 (mA) | Trial 3 (mA) | Mean current +/- SD (mA) | Supply (V) | Mean power +/- SD (mW) | Decisions or MLC services per min | Trace filenames |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NONE | Motionless |  |  |  |  |  |  | N/A |  |
| MLC | Motionless |  |  |  |  |  |  |  |  |
| CNN int8 | Motionless |  |  |  |  |  |  |  |  |
| HDC D=2048 | Motionless |  |  |  |  |  |  |  |  |
| NONE | Scripted active |  |  |  |  |  |  | N/A |  |
| MLC | Scripted active |  |  |  |  |  |  |  |  |
| CNN int8 | Scripted active |  |  |  |  |  |  |  |  |
| HDC D=2048 | Scripted active |  |  |  |  |  |  |  |  |

Calculate `power_mW = current_mA * supply_V`. Save raw PPK2 traces, not only
screenshots or copied averages. Unexpected radio bursts invalidate a BLE-off
trace.

In the paper, label these numbers **motionless steady-state** and **scripted
active (12 prompted gestures/min)**. Neither is a universal daily-life average.
Do not estimate battery life from one row without stating an activity-duty-cycle
assumption. If a workload estimate is useful, report it separately as
`I_est = (1-duty) * I_motionless + duty * I_active` for explicitly chosen duty
values; do not present it as measured free-living current.

Do not make “a few triggers per hour” the primary PPK2 trace: over 60 seconds it
is statistically almost identical to motionless idle. If you want a sparse-use
scenario, first measure many isolated prompted events, integrate incremental
energy above the motionless baseline over a fixed event window, and report the
assumption explicitly:

`P_est_mW(r events/hour) = P_motionless_mW + E_event_mJ * r / 3600`.

This is a modeled workload estimate, not a third measured operating condition.

## Flash and RAM

Run both reports for each build and preserve the corresponding `zephyr.map`:

```sh
west build -d build-benchmark-none -t rom_report
west build -d build-benchmark-none -t ram_report
```

Repeat with `mlc`, `cnn`, and `hdc`. Use the values from the final clean builds,
not older build folders.

| Method | Total flash used (bytes) | Flash delta vs NONE (bytes) | Static RAM used (bytes) | RAM delta vs NONE (bytes) | Classifier asset bytes | Tensor arena / working memory | Report and map paths |
|---|---:|---:|---:|---:|---:|---:|---|
| NONE |  | 0 |  | 0 | 0 | Raw FIFO baseline |  |
| MLC |  |  |  |  |  | IMU-internal MLC memory is not MCU RAM |  |
| CNN int8 |  |  |  |  | 21,728 | 40,960-byte tensor arena; report runtime watermark only if measured |  |
| HDC D=2048 |  |  |  |  | ~11,520 generated HV bytes, verify from map | Integer count/query working arrays |  |

Negative MLC deltas are possible and are not automatically errors. The NONE
baseline reads and drains the 120 Hz FIFO, while the MLC deployment moves
classification into the IMU and stops MCU FIFO reads. Therefore the MLC delta
is a **deployment-path delta**, not the isolated cost of adding a classifier.
This is the honest system comparison and must be stated in the paper.

Report both **total system memory** and **delta versus NONE**. Do not call the
MLC's MCU memory zero, and do not compare only model-file sizes while excluding
the CNN runtime or HDC codebooks.

`ram_report` is the linked/static RAM footprint. It includes statically reserved
objects such as the CNN's 40,960-byte tensor arena and Zephyr thread stacks, but
does not establish runtime stack high-water marks or dynamic heap peaks. Use
“linked/static RAM footprint” in the paper. The current firmware heap is zero;
report a runtime watermark only if one is actually measured.

## Execution/service timing (required while each build is flashed)

Inspect `g_classifier_benchmark_stats` after at least 100 decisions, excluding
startup. CNN/HDC values time `clf_process_window`; MLC values time MCU service of
the sensor result and are not the sensor's unknown internal inference latency.
Record counter values before and after the timed interval or reset them with the
debugger; do not mix startup work into the result. Compute mean cycles as the
delta in `total_cycles` divided by the delta in `inference_count`.

### Exact GDB counter procedure

Open the ELF for the currently flashed variant in a GDB session connected to
the probe. For example, with a J-Link GDB server already listening on port 2331:

```gdb
file build-benchmark-cnn/firmware/zephyr/zephyr.elf
target remote :2331
monitor reset
continue
```

After the 15-second warm-up, interrupt execution with Ctrl-C and reset only the
benchmark counters:

```gdb
set var g_classifier_benchmark_stats.inference_count = 0
set var g_classifier_benchmark_stats.total_cycles = 0
set var g_classifier_benchmark_stats.min_cycles = 0xffffffff
set var g_classifier_benchmark_stats.max_cycles = 0
p g_classifier_benchmark_stats.cycles_per_second
continue
```

After at least 100 CNN/HDC decisions, or after the complete timed MLC trace,
interrupt with Ctrl-C again. The target must be halted while reading the
64-bit total so it cannot tear on this 32-bit MCU. Then run:

```gdb
p g_classifier_benchmark_stats.inference_count
p g_classifier_benchmark_stats.total_cycles
p g_classifier_benchmark_stats.min_cycles
p g_classifier_benchmark_stats.max_cycles
p g_classifier_benchmark_stats.cycles_per_second
set $mean_cycles = (double)g_classifier_benchmark_stats.total_cycles / g_classifier_benchmark_stats.inference_count
set $mean_ms = 1000.0 * $mean_cycles / g_classifier_benchmark_stats.cycles_per_second
p $mean_cycles
p $mean_ms
```

Use the corresponding `mlc`, `cnn`, or `hdc` ELF in the first command. NONE has
no classifier/service count. In an IDE Debug Console that passes commands to
GDB/MI, prefix each expression command with `-exec` (for example,
`-exec p g_classifier_benchmark_stats.inference_count`). Do not reset these
counters during a PPK2 trace when the count delta itself is being used to report
services per minute; halt immediately before and after that trace instead.

Counter semantics are fixed by build: CNN/HDC increment once after each
128-sample window inference (64-sample hop), while MLC increments once for every
GPIO interrupt service attempt, including a spurious interrupt whose output read
does not yield a reportable class. Thus the MLC count is an interrupt/service
rate, not a count of accepted gestures.

| Method | Decisions measured | Mean cycles | Min cycles | Max cycles | CPU cycle frequency (Hz) | Mean measured time (ms) | Definition / caveat |
|---|---:|---:|---:|---:|---:|---:|---|
| MLC |  |  |  |  |  |  | MCU interrupt-result read/service only |
| CNN int8 |  |  |  |  |  |  | 128-sample classifier execution only |
| HDC D=2048 |  |  |  |  |  |  | 128-sample classifier execution only |

## Minimum completion rule

The primary paper table is ready only when all four methods have:

1. A clean build from the same commit and common 120 Hz sensor configuration.
2. Three valid 60-second motionless and three scripted-active power traces with
   BLE excluded from the build; MLC service rate is recorded for both conditions.
3. Total flash/RAM and deltas versus NONE from preserved reports/maps.
4. The exact deployed CNN, HDC, or MLC asset recorded.
5. Required classifier execution/service counters captured while each build is
   already flashed.

If an emergency prevents the full active matrix, MLC active traces are the
minimum scientifically necessary addition because its MCU workload is
event-driven. Mark the matrix incomplete rather than comparing unlike unlabeled
conditions.
