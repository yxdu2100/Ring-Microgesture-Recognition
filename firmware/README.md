# Ring Microgesture Firmware

Single Zephyr/NCS application for nRF54L15 ring gesture experiments. The app always initializes BLE and the LSM6DSV16X IMU; the selected build-time classifier decides whether samples are only streamed or also processed on-device.

## Build Variants

Replace the board with your target if needed. The current local build cache uses `nrf54v1/nrf54l15/cpuapp`.

```sh
west build -b nrf54v1/nrf54l15/cpuapp -- -DEXTRA_CONF_FILE=overlay-none.conf
west build -b nrf54v1/nrf54l15/cpuapp -- -DEXTRA_CONF_FILE=overlay-mlc.conf
west build -b nrf54v1/nrf54l15/cpuapp -- -DEXTRA_CONF_FILE=overlay-cnn.conf
west build -b nrf54v1/nrf54l15/cpuapp -- -DEXTRA_CONF_FILE=overlay-hdc.conf
```

When switching variants in the nRF Connect extension, set the same value in Extra CMake arguments, for example `-DEXTRA_CONF_FILE=overlay-hdc.conf`, and use a pristine build if cached configuration gets sticky. The CNN variant also requires the Zephyr `tflite-micro` module; its overlay enables `CONFIG_TENSORFLOW_LITE_MICRO` and `CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS`.

The Kconfig choice selects which classifier source file is compiled in and which config symbols are set. The four variants come from one codebase with zero runtime classifier dispatch overhead and no code drift between them.

## Generated Assets

Generated classifier assets live in `src/classifiers/generated/`:

- `cnn_model.h`: exported int8 TFLite model bytes
- `hdc_memories.h`: HDC level, channel, and class hypervectors

The active MEMS Studio MLC export is `src/modules/mlc.h`; `clf_mlc.c` applies
its generated operation list directly.

The placeholder headers intentionally warn at build time until replaced by training exports.

## BLE-off measurement builds

Append `overlay-benchmark.conf` to a classifier overlay for power, latency, and
memory measurements. For example:

```sh
west build -p always -d build-benchmark-hdc \
  -b nrf54v1/nrf54l15/cpuapp -- \
  -DEXTRA_CONF_FILE="overlay-hdc.conf;overlay-benchmark.conf"
```

The benchmark build does not initialize BLE. CNN/HDC use the normal raw FIFO,
128-sample window, and 64-sample hop; MLC preserves the MEMS Studio configuration
and waits on its interrupt without draining the raw FIFO. Classification timing
accumulates in `g_classifier_benchmark_stats`, which can be inspected with a
debugger. See `../docs/MEASUREMENT_PROTOCOL.md` for definitions and the PPK2
procedure.
