# RingCollector

SwiftUI iOS app for BLE IMU data collection from the study ring firmware.

## Open in Xcode

```
open ios-app/RingCollector/RingCollector.xcodeproj
```

Build and run on a physical iPhone (BLE required).

## BLE protocol (from firmware)

| Item | UUID / value |
|------|----------------|
| Service | `12345678-9abc-11ee-be56-0242ac120002` |
| IMU data (notify) | `1234567D-9ABC-11EE-BE56-0242AC120002` |
| IMU mode (read/write) | `1234567B-9ABC-11EE-BE56-0242AC120002` |
| Command (write) | `12345678-1234-5678-1234-56789abcde01` |

**Stream control** uses the **command** characteristic (not IMU mode):
- `1` = start streaming
- `0` = stop streaming
- `3` = keepalive (sent every 2 s; firmware lease timeout is 3 s)

IMU mode characteristic maps to `imu_set_trigger_mode()` which is currently a no-op in streaming firmware.

## Session export layout

Each session folder under `Documents/Sessions/`:

- `imu.csv` — raw samples (unwrapped sample_id)
- `markers.csv` — gesture cues and connection events
- `meta.json` — session metadata and quality summary
