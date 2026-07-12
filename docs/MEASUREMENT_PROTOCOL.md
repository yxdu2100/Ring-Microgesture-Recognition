# Firmware measurement protocol

Use the same charged battery or regulated supply, firmware revision, ring,
sensor configuration, room setup, and measurement interval for every method.
Record raw traces and a small metadata sheet; do not copy values by eye only.

## Build matrix

The fill-in matrix for the final experiment is in
`docs/MEASUREMENT_WORKSHEET.md`. The primary comparison fixes every method at
120 Hz and fixes HDC at D=2048/L=32; frequency and HDC-capacity sweeps are not
part of tomorrow's hardware measurements.

Build four pristine variants for `nrf54v1/nrf54l15/cpuapp`:

```sh
python3 -m west build -p always -d build-benchmark-none -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-none.conf;overlay-benchmark.conf"
python3 -m west build -p always -d build-benchmark-mlc  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-mlc.conf;overlay-benchmark.conf"
python3 -m west build -p always -d build-benchmark-cnn  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-cnn.conf;overlay-benchmark.conf"
python3 -m west build -p always -d build-benchmark-hdc  -b nrf54v1/nrf54l15/cpuapp -- -DBOARD_ROOT=/Users/yuxindu/VSCode/nrf/nrf54_projects -DEXTRA_CONF_FILE="overlay-hdc.conf;overlay-benchmark.conf"
```

Use the Nordic toolchain's Python for `python3` if `west` is not on PATH.  Save
the git commit, generated-model hash, build configuration, and compiler output.

The benchmark configuration sets `CONFIG_BT=n`, so the BLE host/controller and
application transport source are not linked; it also disables logging, console,
and shell. CNN/HDC still sample and classify through the normal FIFO/window path.
MLC leaves sensing and classification inside the IMU while the MCU waits forever
on the MLC interrupt queue; its benchmark worker continues before the raw-FIFO
drain path. Call
this **System ON idle with interrupt wake** unless a power-state trace proves a
deeper state; do not simply label it “deep sleep.”

The NONE variant is the **BLE-off sensing/FIFO baseline**: the IMU runs at 120 Hz
and the MCU drains and parses its timestamped FIFO, but performs no classifier or
transport work. It is deliberately not a minimum-power or deep-sleep baseline.

## Steady-state power with PPK2

1. Connect the PPK2 at the battery rail in source-meter mode at the normal ring
   voltage.  Do not power from both PPK2 and battery.
2. Flash one build, reset, wait 15 seconds for startup, then record three
   60-second motionless traces and three 60-second scripted-active traces. Use
   one prompted gesture every 5 seconds in a fixed four-gesture cycle for every
   build. Record MLC interrupt/service events per minute in both conditions.
3. Keep BLE off for all four primary measurements.  A separate BLE-streaming
   measurement may be reported as context, never mixed into one variant.
4. For each trace save mean current, median, 95th percentile, supply voltage,
   duration, and energy.  Report mean ± sample standard deviation across traces.
5. Inspect the trace: MLC should show a low idle baseline plus interrupt/service
   bursts; CNN/HDC should show periodic FIFO and inference work.  Unexpected
   radio bursts invalidate the BLE-off measurement.

Power is continuous-system power, not “energy per inference.”  Only report
energy/inference if inference boundaries are instrumented and the idle baseline
is subtracted with a written method.

## Latency

The firmware exposes `g_classifier_benchmark_stats` to the debugger:
`inference_count`, `total_cycles`, `min_cycles`, and `max_cycles`.

- **CNN/HDC classifier execution time:** cycles around `clf_process_window`,
  converted with the runtime CPU-cycle frequency.  Report mean, min, max, and
  count after warm-up.  This excludes the 128-sample acquisition time.
- **CNN/HDC activation latency:** M=1 requires one completed window; M=2 requires
  a second agreeing window and therefore adds one 64-sample hop, about 533 ms.
  If reporting end-to-end gesture latency, calculate from aligned gesture onset
  to confirmed event and report its distribution.
- **MLC MCU service time:** cycles to read and decode the MLC output after its
  interrupt.  This is not the sensor's internal classification latency.
- **MLC end-to-end activation latency:** use a synchronized physical stimulus or
  reference sensor and a GPIO pulse at the MLC interrupt.  If that setup is not
  available, report service time only and mark internal MLC latency unavailable;
  do not compare unlike latency definitions in one numeric column.
- **MLC M=2 cadence:** verify the output interval on the physical sensor.  The
  offline evaluator conservatively uses the MEMS Studio non-overlapping
  128-sample feature cadence (about 1.07 s per additional decision); replace that
  assumption only with recorded on-sensor evidence.

Use enough decisions to stabilize the mean (at least 100).  M=2 belongs
in the event-level latency/accuracy analysis after each raw method works; it does
not require retraining.

## Flash and RAM

For every final build, save `zephyr.map` and the build reports:

```sh
python3 -m west build -d build-benchmark-cnn -t rom_report
python3 -m west build -d build-benchmark-cnn -t ram_report
```

Repeat for all variants.  Report total used flash/RAM and the increment above
NONE.  For CNN also report the configured tensor arena and, if implemented, its
runtime high-water mark.  For MLC distinguish MCU flash occupied by the UCF
configuration from memory used inside the IMU, which is not MCU RAM.

## Required evidence before numbers enter the paper

- Exact build/commit and generated asset.
- Three raw power traces per method.
- Counter sample size and clock conversion for latency.
- Map/report files for memory.
- Actual MEMS Studio MLC result verified against recorded sensor outputs.
- No value copied from a datasheet as if it were measured on this ring.
- Any “on-device enrollment” claim has an implemented firmware update path,
  retained counter/prototype state, and measured update latency/energy.  The
  current binary-prototype HDC inference build alone does not satisfy this.
