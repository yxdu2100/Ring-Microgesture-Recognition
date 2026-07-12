# Deadline-first next steps

## Current verified checkpoint (development data only)

- Five-fold within-user gesture macro-F1: CNN float 0.978 ± 0.015; HDC with
  validation-fitted rejection 0.691 ± 0.077; sklearn MLC proxy 0.934 ± 0.027.
  The proxy value is not the paper's actual MLC result.
- Fold-1 CNN validation macro-F1 is 0.942 float and 0.947 int8; gesture-test
  macro-F1 is 0.989 for both.  PTQ passes the validation-based 95% gate.
- On the current 40-minute **development** free-living recording, mean false
  activations/hour across folds are CNN 133 (M=1) / 63 (M=2), HDC 875 / 430,
  and proxy tree 51 / 4.5.  These are diagnostic, not final paper numbers.
- Guided event recall is CNN 0.821 (M=1) / 0.891 (M=2), HDC 0.505 / 0.182,
  and proxy tree 0.811 / 0.214.  Therefore strict M=2 is promising for CNN but
  is not a universally safe policy; actual sensor MLC and the final null stream
  must decide its deployment use.
- The timeboxed HDC phase/scaled ablation did **not** replace the primary:
  window gesture macro-F1 changed 0.691→0.671, guided M=2 recall 0.182→0.210,
  and development M=2 FP/hr 430→383. The continuous gains were too small and
  M=2 recall remained unusable; the baseline stays frozen/exported.
- Validation-only capacity diagnostics also do not justify changing the frozen
  primary: experimental D=8192/L=32 reached 0.538 validation macro-F1 and
  D=2048/L=64 reached 0.527 on fold 1, versus 0.524 for the baseline primary.
  Their estimated exported HDC data increase from about 11.5 KB to 46.1 KB and
  19.7 KB, respectively. No test-set selection was performed.
- Feature-HDC is useful as a paper diagnostic but not a primary deployed method:
  window F1 is 0.904 (raw HDC 0.691; proxy tree 0.934), while guided event recall
  is 0.541/0.339 and development FP/hr is 635/487 at M=1/M=2. Report the
  representation insight; do not attach raw-HDC firmware resource numbers to it.

## Do now

1. Run the five-fold primary pipeline and inspect all confusion matrices and
   M=1/M=2 event rows.  Freeze code before collecting the final null test.
2. Export one MEMS Studio bundle for each fold, train/export the actual sensor
   tree, and save each tree text/UCF.  Evaluate those trees with `run_all.py`;
   the sklearn proxy is not a paper result.
3. **Completed timebox:** HDC off-phase augmentation and confidence-scaled
   updates did not improve guided M=2 recall materially. Keep the ablation
   reproducible, freeze/export the baseline, and do not spend more of the
   MLC/power/collection budget on it.
4. Train CNN and HDC.  Deploy only generated assets that pass the quantization or
   Python-to-C verification gates.
5. Build and flash the four benchmark variants, then collect power, latency, and
   memory evidence using `MEASUREMENT_PROTOCOL.md`.

## Collect next

1. Collect a new free-living primary-wearer recording that remains completely
   untouched until all thresholds and M=2 choices are frozen.  Duration matters,
   but an honest shorter exposure is better than claiming 2–3 hours not recorded.
   Immediately run `PYTHONPATH=ml python ml/check_sessions.py --session ID`;
   marker-free structured and free-living null recordings are valid.
2. For each of three labmates, collect two independently re-donned gesture
   sessions.  Also collect one 8–11 minute structured-null recording per person
   if time permits, prioritizing typing and object manipulation.
3. Do not require 15–20 labmate sessions.  Two sessions support only exploratory
   LOPO and a clean enrollment-A/test-B check, which is enough for the deadline
   if described accurately.

## Analysis after participant collection

1. Rebuild the manifest and frozen participant folds once.
2. Report each LOPO participant separately.
3. Simulate HDC enrollment on session A and test only on session B for
   k={0,1,3,5,10}.  Call it on-device enrollment only after implementing and
   measuring persistent prototype counters and the update path in firmware.
4. Add tree/CNN offline few-shot references only after the actual HDC curve,
   primary within-user result, continuous false-activation result, and system
   costs are complete.

## Ethics checkpoint

Confirm IRB/exemption or obtain written guidance **before** using new labmate
data as research data.  Making participants co-authors does not replace consent
or IRB review, and authorship should reflect substantive scholarly contribution,
not be used as an ethics workaround.  If approval is uncertain at submission,
keep the paper's empirical scope to properly authorized data and state the
limitation.
