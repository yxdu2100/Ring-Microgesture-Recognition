# Revised paper structure — deadline-feasible version

Working title: **Three Places to Classify: An In-Sensor, TinyML, and HDC Comparison on a Smart Ring**

The defensible paper is a systems comparison with a strong longitudinal
single-user experiment.  The primary result is **within-user,
leave-session-out** performance from 20 re-donned sessions.  Results from three
labmates with two sessions each are an **exploratory cross-user and enrollment
study**, not a population-level user study.

## Claims to make—and claims to avoid

1. **Make:** same ring, recordings, session-level folds, event policy, and
   measurement harness compare an actual in-sensor MLC, an int8 CNN, and HDC.
2. **Make:** report the accuracy/resource/false-activation trade-off; let the
   measurements determine which method wins each axis.
3. **Make conditionally:** HDC supports inexpensive prototype enrollment, if the
   two-session participant experiment shows recovery on held-out session B.
   Describe it as an offline simulation of an on-device-feasible update until
   the counter state, update API, latency, and energy are implemented on-ring.
4. **Avoid until measured:** “MLC is most accurate,” “CNN catches up with more
   data,” or a specific enrollment recovery claim.
5. **Avoid:** treating a software decision-tree proxy as the in-sensor result.
   It may remain an engineering control or appendix result, clearly named
   `mlc_proxy_tree`; the paper's MLC number must come from the MEMS Studio tree
   deployed and verified on the sensor.
6. **Avoid:** presenting N=4, with only two sessions for three participants, as
   evidence of broad population generalization.

## Research questions

- **RQ1 — Personal model:** Across re-donned sessions, how accurately does each
  method recognize four microgestures for the primary wearer?
- **RQ2 — Continuous operation:** On full guided recordings and held-out
  free-living wear, what event recall and false activations/hour result from
  one-window (M=1) and two-consecutive-agreeing-window (M=2) activation?
- **RQ3 — System cost:** With BLE off, what steady-state power, classifier
  execution/service latency, flash, and RAM does each deployment require?
- **RQ4 — Exploratory personalization:** For a participant absent from base
  training, does simulated HDC prototype enrollment on session A improve
  performance on session B?

## 1. Introduction

- Motivate the deployment choice: classify inside the IMU, on the MCU with a
  neural model, or on the MCU with a lightweight adaptable representation.
- State the controlled-comparison gap.
- Contributions: controlled platform/pipeline; longitudinal within-user plus
  continuous-stream evaluation; measured resource trade-offs; exploratory HDC
  personalization if supported.

## 2. Related work

- Ring and microgesture sensing.
- In-sensor MLC, MCU TinyML, and HDC.
- Personalization and continuous gesture false activations.
- **HyperCam (Lee et al., MobiCom 2025):** the closest MCU-HDC systems
  reference. It uses D=10,000 Binary Spatter Codes, offline modified-OnlineHD
  training, and optimized image encoders. Its count-sketch variant reports
  93.60% on MNIST and 72.79% on seven-class face identification, with 63.00 KB
  and 59.52 KB flash and 0.26 s and 0.27 s latency for those tasks,
  respectively. Its Bloom-filter variant uses still less flash/latency but has
  different accuracies. Avoid a single “20x/100x” comparison: the ratios versus
  quantized CNN baselines vary substantially by model and task.
- HyperCam evaluates balanced, fixed-class image datasets with an 80/20 split;
  its training occurs offline. Our HDC question is different: session shift,
  arbitrary-phase continuous windows, heterogeneous open-set null rejection,
  false activations per hour, and exploratory personalization. This is a
  characterization contribution, not a claim that our trainer reproduces
  HyperCam or that inertial HDC should match its image accuracy.

Keep “first” claims narrow and only after the citations are checked.

## 3. Platform and data collection

- Ring hardware, keyed enclosure, 120 Hz, sensor ranges, timestamped BLE data.
- Four frozen gestures and the intentionally difficult pinch pair.
- Guided cue/marker protocol, re-donning, onset alignment, and quality checks.
- Primary wearer: 20 gesture sessions over two days.
- Null data: structured activities are one `null` class with no activity
  boundaries; the current free-living recording is development data.  Collect a
  new untouched free-living recording for final false-activation reporting.
- Additional participants: three people, two re-donned gesture sessions each;
  collect one structured-null recording per person if feasible.

## 4. Methods and controlled pipeline

- Shared 128-sample windows at 120 Hz and 64-sample hop for MCU methods.
- Actual MLC: deterministic class-balanced MEMS Studio export, UCF deployment,
  and live output verification.
- CNN: train-fold normalization, balanced training, int8 post-training
  quantization accepted only when it retains at least 95% of float macro-F1.
- HDC: frozen-primary D=2048 local n-gram encoding, train-fold integer level
  bounds, deterministic flat integer perceptron retraining, and
  validation-fitted null rejection. A timeboxed ablation combining off-phase
  shift augmentation with confidence-scaled, OnlineHD-inspired updates produced
  only a small M=2 gain while reducing window macro-F1, so it remains an
  explicitly named diagnostic and is not the exported primary trainer. D=8192
  and 64-level variants are likewise validation-only diagnostics.
- Feature-HDC diagnostic: encode the same 40 window statistics used by the
  software-tree proxy using train-fold 1st/99th percentile float32 bounds, 32
  levels, and feature-ID binding at D=2048. It is Python-only, labeled
  diagnostic, and does not inherit raw-HDC firmware power/memory measurements.
- The current firmware stores binary prototypes only; it does not yet retain the
  signed counters required for exact incremental enrollment.  Do not call the
  enrollment experiment “on-device” unless that state/update path is added and
  measured on the ring.
- Event policy: report both M=1 and strict M=2 consecutive predictions of the
  same gesture.  M=2 adds one method-specific decision hop: about 0.53 s for
  CNN/HDC (64 samples) and about 1.07 s for the non-overlapping 128-sample MEMS
  Studio feature cadence, unless on-sensor verification demonstrates a different
  cadence.  It is acceptable only if the measured recall/latency trade-off is
  useful. Treat M=1 and M=2 as predeclared operating points for every method;
  do not select a different method-specific M after inspecting final
  free-living results.

## 5. Evaluation design

### Primary within-user evaluation

- Five frozen session-grouped folds.
- Each fold: 14 gesture sessions train, 2 validation, 4 test.
- Structured-null sessions 016–017 train; session 018 validation.
- The existing free-living session 019 is development-only and may tune the
  event policy/rejection rule.  Tomorrow's new free-living session is final
  test and must not influence tuning.
- Window metrics: per-class precision/recall/F1, gesture macro-F1, confusion
  matrix.  Do not average overlapping null windows into a misleading FP/hr.
- Continuous metrics: guided event recall and false events, plus free-living
  false activations/hour for M=1 and M=2.

### Exploratory cross-user and enrollment evaluation

- LOPO is leave-one-**participant**-out; leave-session-out is the primary
  wearer's within-user evaluation.  Use these names consistently.
- Report all participant folds, not only a mean.
- With two sessions per new participant, use session A only for enrollment and
  session B only for test.  Never enroll and test on windows from one session.
- Evaluate k in {0, 1, 3, 5, 10} examples per gesture only where enough examples
  exist.  This is exploratory because there is one ordered A→B comparison per
  participant; do not imply robust learning-curve statistics.
- Compare weighted additions to the base counters at weights {1,4,16,64} with a
  separate personal-prototype method that takes the per-class minimum distance
  over the base and enrollment-only prototypes. Both use session A only;
  session B remains untouched test data.

### Resource evaluation

- Measure NONE, actual MLC, CNN, and HDC builds with BLE disabled.
- Report two power conditions for every build: motionless steady-state and a
  scripted 60-second sequence with 12 prompted gestures/min. Report the observed
  MLC interrupt/service rate; neither condition is labeled a universal daily
  average.
- Report steady-state current/power, execution or service time with precise
  definitions, total and incremental flash/RAM, and M=2 decision latency.
- The NONE baseline drains the 120 Hz FIFO, whereas the MLC deployment stops MCU
  FIFO reads and classifies inside the IMU. Therefore MLC-minus-NONE is a
  deployment-path delta, not the isolated cost of adding a classifier.
- Define reported RAM as the linked/static footprint from `ram_report`; it
  includes reserved thread stacks and the CNN tensor arena but not measured
  runtime stack high-water marks unless those are collected separately.

## 6. Results

1. Dataset and quality summary.
2. Primary within-user window and continuous-stream accuracy.
3. Held-out final free-living false activations/hour, M=1 versus M=2.
4. Measured resource table and accuracy-versus-power plot.
5. Exploratory LOPO/enrollment results, only if completed correctly.

The representation diagnostic currently yields 0.904 gesture macro-F1 versus
0.691 for raw HDC and 0.934 for the software-tree proxy. However, feature-HDC
guided event recall is only 0.541/0.339 at M=1/M=2 and development FP/hr is
635/487. Thus statistical features explain much of the onset-aligned window gap
but do not remove arbitrary-phase or heterogeneous-null deployment failures.

Do not mix the current development free-living result with the final test
result.  Do not call proxy-tree accuracy “MLC accuracy.”

## 7. Discussion and limitations

- Interpret the measured Pareto trade-off instead of preselecting a winner.
- M=2 is a deployable operating point only if its reduction in false activation
  justifies the added delay and missed short events.
- Limitations: primary accuracy result is one wearer; small and imbalanced
  cross-user sample; one ring fit/position and four gestures; limited final
  free-living duration; MLC/CNN/HDC feature and training capacities differ.
- HDC enrollment is a structural advantage, but empirical benefit must be
  described as preliminary with this participant/session count.
- The feature-HDC diagnostic isolates representation as an important factor in
  aligned-window accuracy, not as the sole explanation of the paradigm gap:
  continuous recall and false activations remain poor. Because the diagnostic
  has no measured MCU feature extractor, it cannot replace raw HDC in the
  on-device resource table.

## Candidate system interpretation (conditional on measurement)

- **MLC:** expected to minimize system power because the MCU can idle while the
  sensor classifies. Its hypothesis class, feature menu, and UCF update path are
  constrained. Say “lowest power” and “lowest incremental classifier memory”
  only after on-ring measurement; do not say the complete system uses
  near-zero MCU memory.
- **CNN:** currently has the best within-user accuracy and continuous robustness
  on development data, at the expected cost of TFLM/CMSIS-NN, tensor-arena, and
  compute energy. Keep this conditional until the actual MLC, int8 firmware,
  and final null results exist.
- **HDC:** adaptability is the intended axis: inference is integer-only and an
  update can be expressed as additions to prototype counters. Its absence of an
  inference framework may reduce flash relative to CNN, but memory/power claims
  require measured builds, and the current firmware still lacks persistent
  enrollment counters. Do not claim graceful degradation without a defined and
  measured degradation experiment.

## 8. Conclusion

Summarize the controlled comparison, the personal-device result, continuous
activation behavior, and the measured deployment trade-off without extrapolating
beyond the data.

## Six-page figure/table budget

- Figure 1: ring, gestures, and three deployment paths.
- Figure 2: accuracy versus average power; marker size = incremental memory.
- Figure 3: one compact confusion-matrix panel for the three actual methods.
- Figure 4: HDC enrollment, only if the experiment is valid and informative.
- Table 1: dataset/split summary.
- Table 2: accuracy, M=1/M=2 continuous metrics, power, latency, flash, RAM.

Cut order: sampling-rate sweep; CNN learning-curve figure; proxy tree; offline
retraining references; enrollment figure if participant data or separation is
insufficient.  Never cut split definitions, actual-versus-proxy distinction,
continuous false-activation reporting, or resource-measurement definitions.
