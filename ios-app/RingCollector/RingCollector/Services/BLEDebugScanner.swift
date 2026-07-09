import CoreBluetooth
import Foundation

struct DiscoveredPeripheral: Identifiable {
    let id: UUID
    /// The name from THIS specific advertisement packet's local-name field - always
    /// live/current, never stale.
    var advertisedName: String?
    /// CBPeripheral.name - iOS caches this per physical BLE address at the OS level
    /// and it can lag behind what the device is actually broadcasting right now (e.g.
    /// left over from a much earlier session with different firmware on the same
    /// address). Shown separately so a live/cached mismatch is obvious at a glance.
    var cachedName: String?
    var rssi: Int
    var advertisedServiceUUIDs: [String]
    var isConnectable: Bool
    var lastSeen: Date
    var sightings: Int

    var displayName: String {
        advertisedName ?? cachedName ?? "(no name)"
    }
}

/// Unfiltered BLE scanner for diagnosing connection problems: lists every nearby
/// advertisement (name, RSSI, advertised services), independent of RingBLEManager's
/// "Ring"-only filter. Uses its own CBCentralManager so it can never interfere with
/// the app's real connection state machine.
@Observable
final class BLEDebugScanner: NSObject {
    private(set) var isScanning = false
    private(set) var discovered: [UUID: DiscoveredPeripheral] = [:]
    private(set) var centralStateLabel = "Unknown"

    private var central: CBCentralManager!

    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: nil,
                                    options: [CBCentralManagerOptionShowPowerAlertKey: false])
    }

    var sortedDiscovered: [DiscoveredPeripheral] {
        discovered.values.sorted { $0.rssi > $1.rssi }
    }

    func startScan() {
        guard central.state == .poweredOn else { return }
        discovered.removeAll()
        isScanning = true
        central.scanForPeripherals(withServices: nil,
                                    options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
    }

    func stopScan() {
        isScanning = false
        central.stopScan()
    }
}

extension BLEDebugScanner: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn: centralStateLabel = "Powered On"
        case .poweredOff: centralStateLabel = "Powered Off"
        case .unauthorized: centralStateLabel = "Unauthorized"
        case .unsupported: centralStateLabel = "Unsupported"
        case .resetting: centralStateLabel = "Resetting"
        default: centralStateLabel = "Unknown"
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let advertisedName = advertisementData[CBAdvertisementDataLocalNameKey] as? String
        let uuids = (advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID])?.map(\.uuidString) ?? []
        let connectable = (advertisementData[CBAdvertisementDataIsConnectable] as? NSNumber)?.boolValue ?? false
        let previousSightings = discovered[peripheral.identifier]?.sightings ?? 0

        discovered[peripheral.identifier] = DiscoveredPeripheral(
            id: peripheral.identifier,
            advertisedName: advertisedName,
            cachedName: peripheral.name,
            rssi: RSSI.intValue,
            advertisedServiceUUIDs: uuids,
            isConnectable: connectable,
            lastSeen: Date(),
            sightings: previousSightings + 1
        )
    }
}
