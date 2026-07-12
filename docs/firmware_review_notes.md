# Firmware review notes

**Update:** items 0–3 below have now been applied to the firmware source at the user's
request (after confirming on hardware that "Ring" doesn't show up in the app's raw BLE
scanner at all — i.e. genuinely not advertising, not just an app-side reconnect issue).
Item 4 was deliberately *not* applied — see its note for why. **You still need to build,
flash, and re-test on real hardware** — this environment has no `west`/NCS toolchain, so
none of this was compiled or verified beyond visual/grep inspection.

Scope: `firmware/src/modules/imu.c`, `firmware/src/ble/ble.c`, `firmware/src/main.c`,
data-collection path only (`overlay-none.conf` / `CONFIG_CLASSIFIER_NONE`). CNN/MLC/HDC
classifier code was not reviewed per the current task.

## Overall assessment

The IMU → BLE data path is internally consistent and matches the hard rules in
`Codex Prompt.md`:

- FIFO tag/slot decode (`tag = row[0]>>3`, `slot = (row[0]>>1)&0x3`) matches the
  LSM6DSV16X FIFO_DATA_OUT_TAG bit layout, and tag values `0x01/0x02/0x04` for
  gyro/accel/timestamp are correct.
- CTRL1/CTRL2 ODR codes (`0x05`=60 Hz, `0x06`=120 Hz), CTRL6 gyro FS (`0x04`=±2000 dps),
  and CTRL8 accel FS (`0x02`=±8g) match the datasheet and the project's "do not lower"
  full-scale rule.
- FIFO overrun handling resyncs instead of replaying a stale backlog, and the
  watchdog-poll fallback (`IMU_POLL_FALLBACK_MS`) prevents the watermark IRQ from
  wedging — both are good robustness choices already in place.
- The BLE payload layout (23 bytes: sample_id u16, timestamp_us u32, timestamp_ticks
  u32, flags u8, accel×3 i16, gyro×3 i16) matches `IMUSampleParser.swift` byte-for-byte.

## 0. [APPLIED] Reconnect failure ("connects once, then stuck scanning forever") — top suspect

Investigated for the "first connection works, later ones never do, RTT only shows the
boot lines" report. Found via `grep` that **`bt_app_stream_should_continue()` is defined
but never called anywhere in the firmware** — not in `imu.c`, not in `main.c`, nowhere.
That function is the only caller of `recover_stream_transport()` / `stream_lease_expired()`,
i.e. the watchdog that's supposed to force `bt_conn_disconnect()` (and thus trigger
`disconnected()` → `bt_le_adv_start()` to resume advertising) if the phone goes silent
(no keepalive for `STREAM_LEASE_TIMEOUT_MS` = 3s) while still marked as streaming. As
written, this recovery path is dead code — if the link ever ends up in a state where the
normal HCI disconnect event doesn't fire promptly (radio still thinks it's connected to
the phone even though the app/phone side has moved on), nothing in the firmware will ever
notice or force it loose, so advertising never resumes and the ring is invisible to any
new scan.

**Applied fix:** `main.c`'s main loop (which already ticks every 20 ms) now calls
`bt_app_stream_should_continue();` unconditionally on every iteration, so a stuck lease
gets noticed within milliseconds and forces a disconnect + re-advertise instead of
sitting there forever. This was the simplest correct hook point — it was already looping
at 20 ms and already has visibility into `sys_events`, so no new thread/timer was needed.

**How to confirm on hardware:** next time it happens, check the RTT log (scroll past the
boot lines) for whether `"Disconnected (reason 0x..)"` and `"Advertising restarted"` ever
printed after the first session ended. If they never appear, this dead-watchdog theory is
confirmed — the peripheral thinks it's still connected. If `"Advertising failed to restart
(err %d)"` appears instead, that's a different bug (the restart call itself is erroring)
and worth reporting the err code.

**App-side workaround already shipped this session** (independent of the firmware fix):
`RingBLEManager` now calls `retrieveConnectedPeripherals(withServices:)` before scanning.
This covers the *other* likely half of the symptom — the app never calls `disconnect()`
in normal use (by design, so sessions can survive app restarts), so if the app process
ever ends without a clean teardown, iOS's system Bluetooth daemon can keep holding the
ring "connected" at the OS level. A peripheral that's already connected doesn't advertise,
so a plain scan from a fresh `CBCentralManager` would never find it — the app now checks
for and reconnects to an already-connected peripheral directly instead of scanning
forever. A new **BLE Scanner (debug)** screen (Home → "Scan for nearby BLE devices") also
lets you see every nearby advertisement unfiltered, to tell firmware-side ("Ring" never
shows up at all) from app-side (it shows up but won't connect) issues in the field.

## 1. [APPLIED] BLE notification size vs. negotiated ATT MTU

`BT_APP_IMU_SAMPLES_PER_PACKET = 10` at `BT_APP_IMU_SAMPLE_PAYLOAD_LEN = 23` bytes meant
a full packet was **230 bytes**, requiring a negotiated ATT MTU of **≥ 233 bytes**
(230 + 3-byte ATT header). The firmware requests up to MTU 247 (`CONFIG_BT_L2CAP_TX_MTU=247`),
but the *central* (iPhone) ultimately decides the negotiated value, and CoreBluetooth
gives the app no direct way to query or force it. Some iPhone/iOS combinations have been
reported to cap around 185 bytes rather than the max 251/517 — if that happens, every
10-sample notification would be rejected by `bt_gatt_notify()` (handled gracefully — it
just increments `dropped_imu_batches` — but that's a 100%, not partial, loss of every
full packet).

This was self-diagnosing (the app's Data Quality panel already surfaces drop % and
effective Hz, so an MTU shortfall would show up as near-100% drops), but of the two
options previously listed I applied the simpler/lower-risk one: **`BT_APP_IMU_SAMPLES_PER_PACKET`
is now `7`** (23 × 7 = 161 bytes, +3-byte ATT header = 164, safe under even a conservative
185-byte MTU) instead of dynamically querying `bt_gatt_get_mtu(conn)` per packet, which
would have added more moving parts for the same practical benefit. At 120 Hz this just
means slightly more frequent, smaller notifications (~17/s instead of ~12/s) — well
within what the connection interval (15 ms) supports.

## 2. [APPLIED] Removed dead code: `bt_app_send_imu_sample` (singular)

The single-sample send helper (was `ble.c:348` / `ble.h:37`) was never called anywhere in
the tree — confirmed via `grep` before deleting. `bt_app_send_imu_samples` (plural, batch)
is the one actually used by `imu.c` and is unaffected.

## 3. [APPLIED] `imu_sample_id` now resets on every `imu_start_streaming()`

Previously the FIFO parser state was reset but not the `uint16_t imu_sample_id` counter
itself, so sample IDs kept incrementing across multiple start/stop cycles within one
boot. Not a correctness bug (the app's `SampleIDUnwrapper.reset()` already re-bases per
session on the first sample it sees), but `imu_sample_id = 0;` was added right next to the
other per-session resets in `imu_start_streaming()` so RTT logs are easier to read
per-session too.

## 4. [NOT APPLIED — judgment call] "Streaming confirmed" signal from firmware

Deliberately skipped rather than added a half-used feature. The suggested approach (ACK
the start command by notifying a byte on the *command* characteristic once
`imu_start_streaming()` returns 0) would be dead on arrival: the app never subscribes to
notifications on the command characteristic (`RingBLEManager` only calls
`setNotifyValue(true, for:)` on the IMU data characteristic), so the ACK would have no
consumer unless the app is also changed to subscribe and handle it. Since the app already
has an equivalent signal — `DataQualityTracker.isReceivingSamples`, gating "Start Block"
on actual IMU sample arrival — adding a second, redundant "stream started" mechanism on
the command channel would just be more surface area for the same outcome. If you want a
harder guarantee than "first IMU notification arrived" (e.g., to distinguish "started but
zero samples yet" from "command never received"), say so and I'll wire up both sides
together rather than leaving inert code in the firmware.
