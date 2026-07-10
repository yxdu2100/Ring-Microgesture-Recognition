import CoreBluetooth
import Foundation

@Observable
final class RingBLEManager: NSObject {
    var connectionState: BLEConnectionState = .idle
    var isStreaming = false
    var lastUnwrappedSampleID: UInt64?
    var connectedPeripheralName: String?
    var latestInference: InferenceResult?
    var recentInferenceResults: [InferenceResult] = []

    var onSamplesReceived: (([IMUSample]) -> Void)?
    var onConnectionEvent: ((String, UInt64?) -> Void)?
    var onInferenceReceived: ((InferenceResult) -> Void)?

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var commandCharacteristic: CBCharacteristic?
    private var imuDataCharacteristic: CBCharacteristic?
    private var classificationCharacteristic: CBCharacteristic?

    private let unwrapper = SampleIDUnwrapper()
    private var shouldAutoReconnect = true
    private var keepaliveTimer: Timer?
    private var lastPeripheralIdentifier: UUID?
    private var disconnectedThisSession = false
    private var pendingStreamStart = false
    private var imuNotificationsReady = false
    private var classificationNotificationsReady = false
    /// Last live advertised local-name seen for the peripheral we're connecting to.
    /// CBPeripheral.name is an iOS-level cache keyed by physical BLE address and can
    /// be stale (e.g. a name left over from different firmware on the same address),
    /// so prefer this for display instead.
    private var lastAdvertisedName: String?

    override init() {
        super.init()
        central = CBCentralManager(
            delegate: self,
            queue: nil,
            options: [CBCentralManagerOptionRestoreIdentifierKey: BLEConstants.restoreIdentifier]
        )
    }

    func setAutoReconnect(_ enabled: Bool) {
        shouldAutoReconnect = enabled
    }

    func startScanning() {
        guard central.state == .poweredOn else { return }

        // The ring may already be connected at the iOS/Bluetooth-daemon level from a
        // previous app session that ended without an explicit disconnect (we never call
        // disconnect() during normal use, by design, so sessions can survive app
        // restarts). A peripheral that's already connected stops advertising per the BLE
        // spec, so a plain scan would never find it again - check for that first and
        // connect directly if so, instead of scanning forever.
        if let already = central.retrieveConnectedPeripherals(withServices: [BLEConstants.serviceUUID]).first {
            connectToKnownPeripheral(already)
            return
        }

        connectionState = .scanning
        central.scanForPeripherals(
            withServices: [BLEConstants.serviceUUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
    }

    private func connectToKnownPeripheral(_ found: CBPeripheral) {
        stopScanning()
        peripheral = found
        found.delegate = self
        lastPeripheralIdentifier = found.identifier
        connectionState = .connecting
        central.connect(found, options: nil)
    }

    func stopScanning() {
        central.stopScan()
        if connectionState == .scanning {
            connectionState = peripheral == nil ? .idle : connectionState
        }
    }

    func connect() {
        startScanning()
    }

    func disconnect() {
        shouldAutoReconnect = false
        stopKeepalive()
        if let peripheral {
            central.cancelPeripheralConnection(peripheral)
        }
    }

    func resetForNewSession() {
        unwrapper.reset()
        lastUnwrappedSampleID = nil
        disconnectedThisSession = false
    }

    func startStreaming() {
        pendingStreamStart = true
        isStreaming = true
        activateStreamingIfReady()
    }

    func stopStreaming() {
        pendingStreamStart = false
        stopKeepalive()
        writeCommand(BLEConstants.commandStop)
        isStreaming = false
    }

    private func activateStreamingIfReady() {
        guard pendingStreamStart, commandCharacteristic != nil, imuNotificationsReady else { return }
        if classificationCharacteristic != nil && !classificationNotificationsReady {
            return
        }
        if disconnectedThisSession {
            unwrapper.markFirmwareWillRestart()
        }
        writeCommand(BLEConstants.commandStart)
        startKeepalive()
    }

    private func writeCommand(_ byte: UInt8) {
        guard let peripheral, let commandCharacteristic else { return }
        let data = Data([byte])
        peripheral.writeValue(data, for: commandCharacteristic, type: .withoutResponse)
    }

    private func startKeepalive() {
        stopKeepalive()
        keepaliveTimer = Timer.scheduledTimer(withTimeInterval: BLEConstants.keepaliveInterval, repeats: true) { [weak self] _ in
            self?.writeCommand(BLEConstants.commandKeepalive)
        }
    }

    private func stopKeepalive() {
        keepaliveTimer?.invalidate()
        keepaliveTimer = nil
    }

}

extension RingBLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            if connectionState == .poweredOff || connectionState == .idle {
                if shouldAutoReconnect, lastPeripheralIdentifier != nil {
                    connectionState = .reconnecting
                    startScanning()
                }
            }
        case .unauthorized:
            connectionState = .unauthorized
        case .poweredOff:
            connectionState = .poweredOff
            isStreaming = false
            stopKeepalive()
        default:
            break
        }
    }

    func centralManager(_ central: CBCentralManager, willRestoreState dict: [String: Any]) {
        guard let peripherals = dict[CBCentralManagerRestoredStatePeripheralsKey] as? [CBPeripheral],
              let restored = peripherals.first else { return }

        // Previously this only recorded the restored peripheral and set the state to
        // .reconnecting without ever calling connect() - the app would sit there
        // indefinitely after an OS-triggered relaunch. Actually (re)connect now.
        if restored.state == .connected {
            peripheral = restored
            restored.delegate = self
            lastPeripheralIdentifier = restored.identifier
            connectionState = .connecting
            restored.discoverServices([BLEConstants.serviceUUID])
        } else {
            connectToKnownPeripheral(restored)
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        // Prefer the LIVE advertisement's local-name field over CBPeripheral.name.
        // peripheral.name is cached by iOS per physical BLE address and can lag behind
        // what's actually being broadcast right now (e.g. a name left over from
        // different firmware previously flashed onto the same hardware address) -
        // using it first here silently rejected a ring that was live-advertising the
        // right name but had a stale/mismatched cached name, making it look like the
        // scan just never found the device.
        let liveName = advertisementData[CBAdvertisementDataLocalNameKey] as? String
        let name = liveName ?? peripheral.name ?? ""
        guard name == BLEConstants.deviceName || name.hasPrefix(BLEConstants.deviceName) else { return }

        lastAdvertisedName = liveName
        connectToKnownPeripheral(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectionState = .connected
        connectedPeripheralName = lastAdvertisedName ?? peripheral.name
        peripheral.discoverServices([BLEConstants.serviceUUID])
        if disconnectedThisSession {
            onConnectionEvent?("reconnect", lastUnwrappedSampleID)
        }
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        connectionState = .disconnected
        if shouldAutoReconnect {
            connectionState = .reconnecting
            startScanning()
        }
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        self.peripheral = nil
        commandCharacteristic = nil
        imuDataCharacteristic = nil
        classificationCharacteristic = nil
        imuNotificationsReady = false
        classificationNotificationsReady = false
        isStreaming = false
        stopKeepalive()
        connectionState = .disconnected
        disconnectedThisSession = true
        onConnectionEvent?("disconnect", lastUnwrappedSampleID)

        if shouldAutoReconnect {
            connectionState = .reconnecting
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.startScanning()
            }
        }
    }
}

extension RingBLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil, let service = peripheral.services?.first(where: { $0.uuid == BLEConstants.serviceUUID }) else { return }
        peripheral.discoverCharacteristics(
            [
                BLEConstants.imuDataUUID,
                BLEConstants.imuModeUUID,
                BLEConstants.commandUUID,
                BLEConstants.classificationUUID,
            ],
            for: service
        )
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard error == nil, let characteristics = service.characteristics else { return }

        for characteristic in characteristics {
            switch characteristic.uuid {
            case BLEConstants.commandUUID:
                commandCharacteristic = characteristic
            case BLEConstants.imuDataUUID:
                imuDataCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            case BLEConstants.classificationUUID:
                classificationCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
            default:
                break
            }
        }
        activateStreamingIfReady()
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil else { return }

        if characteristic.uuid == BLEConstants.imuDataUUID {
            imuNotificationsReady = characteristic.isNotifying
        } else if characteristic.uuid == BLEConstants.classificationUUID {
            classificationNotificationsReady = characteristic.isNotifying
        } else {
            return
        }

        if characteristic.isNotifying, isStreaming {
            activateStreamingIfReady()
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil, let data = characteristic.value else { return }

        if characteristic.uuid == BLEConstants.classificationUUID {
            guard let result = InferenceResultParser.parse(from: data) else { return }
            latestInference = result
            recentInferenceResults.insert(result, at: 0)
            if recentInferenceResults.count > 20 {
                recentInferenceResults.removeLast(recentInferenceResults.count - 20)
            }
            onInferenceReceived?(result)
            return
        }

        guard characteristic.uuid == BLEConstants.imuDataUUID else { return }

        let samples = IMUSampleParser.parseSamples(from: data, unwrapper: unwrapper)
        guard !samples.isEmpty else { return }

        if let last = samples.last {
            lastUnwrappedSampleID = last.unwrappedSampleID
        }
        onSamplesReceived?(samples)
    }
}
