import CoreBluetooth
import SwiftUI

struct BLEDebugScanView: View {
    @State private var scanner = BLEDebugScanner()

    var body: some View {
        List {
            Section {
                LabeledContent("Bluetooth state", value: scanner.centralStateLabel)
                Text("Unfiltered scan - shows every BLE advertisement nearby, not just \"\(BLEConstants.deviceName)\". Name shown is the LIVE advertised name when available; iOS's own per-address name cache is shown separately in orange whenever it disagrees with what's actually being broadcast right now.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Nearby BLE devices (\(scanner.sortedDiscovered.count))") {
                if scanner.sortedDiscovered.isEmpty {
                    Text(scanner.isScanning ? "Scanning… nothing seen yet" : "Not scanning")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(scanner.sortedDiscovered) { device in
                        deviceRow(device)
                    }
                }
            }
        }
        .navigationTitle("BLE Scanner")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(scanner.isScanning ? "Stop" : "Rescan") {
                    if scanner.isScanning {
                        scanner.stopScan()
                    } else {
                        scanner.startScan()
                    }
                }
            }
        }
        .onAppear { scanner.startScan() }
        .onDisappear { scanner.stopScan() }
    }

    private func isLikelyRing(_ device: DiscoveredPeripheral) -> Bool {
        device.displayName.hasPrefix(BLEConstants.deviceName)
            || device.advertisedServiceUUIDs.contains { $0.caseInsensitiveCompare(BLEConstants.serviceUUID.uuidString) == .orderedSame }
    }

    @ViewBuilder
    private func deviceRow(_ device: DiscoveredPeripheral) -> some View {
        let matches = isLikelyRing(device)
        let namesMismatch = device.advertisedName != nil && device.cachedName != nil
            && device.advertisedName != device.cachedName
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                if matches {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                }
                Text(device.displayName)
                    .font(.headline)
                    .foregroundStyle(matches ? .green : .primary)
                Spacer()
                Text("\(device.rssi) dBm")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            // iOS caches CBPeripheral.name per physical BLE address; it can lag behind
            // what's actually in the live advertisement (e.g. a stale name left over
            // from different firmware previously flashed onto the same hardware
            // address). Surface both explicitly whenever they disagree.
            if namesMismatch {
                Label {
                    Text("live ad name: \"\(device.advertisedName ?? "-")\"  ·  iOS cached name: \"\(device.cachedName ?? "-")\"")
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                }
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.orange)
            }
            if !device.advertisedServiceUUIDs.isEmpty {
                Text(device.advertisedServiceUUIDs.joined(separator: ", "))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Text(device.isConnectable ? "Connectable" : "Not connectable")
                Spacer()
                Text("seen \(device.sightings)×")
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
    }
}
