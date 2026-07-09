# MEMS Studio MLC Reproduction Notes

This folder does not train the in-sensor MLC directly. The Python tree is a
controlled stand-in using the same feature family and frozen train split.

## Suggested MEMS Studio flow

1. Open MEMS Studio and create a new LSM6DSV16X Machine Learning Core project.
2. Import the tab-delimited files produced by:

   ```sh
   PYTHONPATH=ml python3 ml/train_mlc/export_memsstudio.py --data-dir data
   ```

3. Map columns as `ax ay az gx gy gz`, 120 Hz, accelerometer ±8 g, gyroscope
   ±2000 dps.
4. Create classes:
   `double_side_tap`, `double_pinch`, `pinch_hold`, `double_flick`, `null`.
5. Configure the feature set to match `ml/train_mlc/features.py`:
   mean, variance, energy, peak-to-peak, and zero-crossings on each axis plus
   accel and gyro norms.
6. Train a decision tree with max depth 6 and balanced class handling where
   available.
7. Export the `.ucf`.
8. TODO: Record the exact MEMS Studio version.
9. TODO: Record each GUI panel setting and generated feature register mapping.
10. TODO: Paste the final `.ucf` export checklist here.
