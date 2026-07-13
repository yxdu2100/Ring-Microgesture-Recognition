# 3. Platform and Methods

## 3.1 Ring platform

Our prototype ring integrates an STMicroelectronics LSM6DSV16X IMU and a
Nordic nRF54L15 MCU running Zephyr RTOS in a 3D-printed keyed enclosure that
enforces a repeatable wearing orientation [TODO: finger, ring dimensions,
battery capacity]. The IMU samples 3-axis accelerometer (±8 g) and gyroscope
(±2000 dps) data at 120 Hz into its internal FIFO. During data collection,
timestamped samples stream over BLE to a host; during deployment and all
power measurements, BLE is compiled out entirely. The LSM6DSV16X embeds a
Machine Learning Core (MLC): a configurable feature-extraction and
decision-tree engine that classifies inside the sensor package, allowing the
MCU to remain in System ON idle with interrupt wake. This single platform
lets us place classification in three different locations — inside the IMU,
on the MCU as a quantized neural network, or on the MCU as a
hyperdimensional classifier — while holding the sensor, data, and evaluation
pipeline constant.

## 3.2 Gesture set

We use four thumb-scale microgestures: *double side tap*, *double pinch*,
*pinch hold*, and *double flick*, plus a *null* class covering everything
else. The set deliberately includes a difficult pair — double pinch and
pinch hold share the same contact posture and differ mainly in temporal
profile — so that the comparison exercises temporal discrimination rather
than only spatial separation.

## 3.3 Data collection

**Guided gesture sessions.** The primary wearer (an author) recorded 20
gesture sessions across two days, removing and re-donning the ring between
sessions so that session boundaries capture realistic placement variation.
Sessions followed an app-guided cue protocol ([TODO: n] cued instances per
gesture per session; 7 sessions at a relaxed pace and 13 at a faster pace),
yielding 1,193 valid gesture instances after quality screening. Cue markers
are refined to motion onset by an energy-based re-anchoring step, and each
instance is segmented as a 128-sample (1.07 s) window.

**Null data.** Three structured-null recordings (~8–11 min each) captured
scripted everyday activity — typing, object manipulation, phone use [TODO:
confirm activity list] — as a single undifferentiated null class. One 40 min
free-living recording served as development data for calibrating rejection
thresholds and event policy. After all models, thresholds, and code were
frozen, we recorded a final free-living set (2.662 h across two recordings)
that was evaluated exactly once and never influenced any design decision.
All recordings passed integrity checks (zero sample-ID gaps, 100% hardware
timestamps, median rate 119.73 Hz).

**Splits.** We evaluate within-user, leave-session-out: five frozen folds,
each holding out four gesture sessions for testing, with two validation
sessions and fourteen training sessions, stratified by day and pacing. No
session contributes windows to more than one role. Two structured-null
recordings join training; the third serves validation (early stopping and
rejection calibration). A dataset/split hash is embedded in every result and
generated firmware artifact.

## 3.4 Three classification deployments

All three methods consume identical 128-sample, 6-channel windows at 120 Hz;
the MCU-resident methods slide with a 64-sample (0.53 s) hop, while the MLC
advances by complete non-overlapping feature windows (1.07 s), its native
cadence.

**In-sensor MLC.** For each fold we export class-balanced training windows
and train a depth-≤6 decision tree in MEMS Studio [TODO: version] over
on-sensor statistical features (per-axis mean, variance, energy,
peak-to-peak, zero crossings, and accelerometer/gyroscope norms). The
resulting configuration is deployed to the sensor; a Python re-implementation
of each exported tree reproduces its decisions for offline evaluation, and
live sensor output was verified against the parsed tree over BLE. The MCU
performs no signal processing: it wakes on the MLC interrupt and reads one
output register.

**MCU CNN (TFLM).** A small convolutional network [TODO: 1-line
architecture] is trained per fold with train-fold standardization,
non-wrapping temporal-shift augmentation, and session-grouped early stopping
on validation macro-F1. Models are post-training quantized to full int8 and
accepted only if int8 validation macro-F1 retains ≥95% of the float value
(the deployed model retained 100.5%). Inference runs in TensorFlow Lite
Micro (21.7 kB model, 40 kB tensor arena).

**MCU HDC.** We use binary spatter codes (D=2048). Each sample is quantized
into 32 levels per channel using train-fold percentile bounds rounded to the
integers deployed in firmware; level and channel codebooks are bound and
combined into trigram temporal n-grams, bundled by majority vote into one
window hypervector. Class prototypes are accumulated from training windows
and refined with perceptron-style updates; prediction is nearest prototype
by Hamming distance. Because heterogeneous null cannot be captured by a
single prototype, a rejection rule (maximum distance and minimum margin,
calibrated only on validation data) maps low-confidence windows to null. A
known-window test verifies bit-exact agreement between the Python trainer
and the C inference path.

## 3.5 Continuous operation and metrics

Deployment behavior is evaluated on complete recordings, not segmented
windows. Chronological windows are classified in order; an activation fires
when M consecutive windows agree on the same non-null gesture, with a 1 s
refractory period and state reset at any sample gap. We report both M=1 and
M=2. Activations are matched to cued instances within the gesture interval
plus one hop of grace, yielding event recall, wrong-gesture and duplicate
rates, and onset-to-activation latency; on free-living data every activation
is false, yielding false activations per hour. Window-level accuracy is
reported as macro-F1 over the four gestures (with a null-rejection column),
since averaging overlapping null windows would understate deployment false
positives. Resource cost is measured on hardware: steady-state current at
4.2 V with BLE off for four firmware variants (no-classifier baseline plus
one per method), and flash/RAM from linked images, reported as totals and
deltas versus the baseline.
