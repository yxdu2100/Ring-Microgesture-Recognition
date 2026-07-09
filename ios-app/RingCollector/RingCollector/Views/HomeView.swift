import SwiftUI

struct HomeView: View {
    @Environment(RecordingCoordinator.self) private var coordinator
    @Environment(AppSettings.self) private var settings
    @Environment(\.scenePhase) private var scenePhase

    @State private var showGuided = false
    @State private var showNull = false
    @State private var showSessions = false
    @State private var showSettings = false
    @State private var showBLEScanner = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    participantSection
                    connectionSection
                    modeSection
                    statsSection
                }
                .padding()
            }
            .navigationTitle("RingCollector")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .onAppear {
                coordinator.sessionStore.refresh()
                if !coordinator.ble.connectionState.isConnected {
                    coordinator.ble.setAutoReconnect(true)
                    coordinator.ble.connect()
                }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active {
                    coordinator.sessionStore.refresh()
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
            .navigationDestination(isPresented: $showGuided) {
                GuidedSessionView()
            }
            .navigationDestination(isPresented: $showNull) {
                NullRecordingView()
            }
            .navigationDestination(isPresented: $showSessions) {
                SessionsListView()
            }
            .navigationDestination(isPresented: $showBLEScanner) {
                BLEDebugScanView()
            }
        }
    }

    private var participantSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Participant")
                .font(.headline)
            TextField("Participant ID", text: Bindable(settings).participantID)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
    }

    private var connectionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ring Connection")
                .font(.headline)
            HStack {
                Circle()
                    .fill(coordinator.ble.connectionState.isConnected ? .green : .orange)
                    .frame(width: 10, height: 10)
                Text(coordinator.ble.connectionState.label)
                Spacer()
                if !coordinator.ble.connectionState.isConnected {
                    Button("Connect") {
                        coordinator.ble.setAutoReconnect(true)
                        coordinator.ble.connect()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))

            if !coordinator.ble.connectionState.isConnected {
                Button("Scan for nearby BLE devices (debug)") {
                    showBLEScanner = true
                }
                .font(.caption)
            }
        }
    }

    private var modeSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recording Mode")
                .font(.headline)

            if !coordinator.ble.connectionState.isConnected {
                Text("Connect the ring before starting a session — recording without a connection captures no IMU data.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            Button {
                guard !settings.participantID.isEmpty else { return }
                showGuided = true
            } label: {
                ModeCard(
                    title: "Guided Gestures",
                    subtitle: "\(settings.enabledGestures.count) gestures · \(settings.repsPerBlock) reps/block",
                    systemImage: "hand.tap.fill",
                    tint: .blue
                )
            }
            .disabled(settings.participantID.isEmpty || coordinator.isActiveSession || !coordinator.ble.connectionState.isConnected)

            Button {
                guard !settings.participantID.isEmpty else { return }
                showNull = true
            } label: {
                ModeCard(
                    title: "Null Recording",
                    subtitle: "Free-form baseline capture",
                    systemImage: "waveform",
                    tint: .purple
                )
            }
            .disabled(settings.participantID.isEmpty || coordinator.isActiveSession || !coordinator.ble.connectionState.isConnected)

            Button {
                showSessions = true
            } label: {
                ModeCard(
                    title: "Sessions",
                    subtitle: "\(coordinator.sessionStore.sessions.count) recorded",
                    systemImage: "folder.fill",
                    tint: .green
                )
            }
        }
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Study Stats")
                .font(.headline)
            LabeledContent("Null time (all sessions)") {
                Text(String(format: "%.1f min", coordinator.sessionStore.totalNullMinutes))
                    .monospacedDigit()
            }
            LabeledContent("Storage used") {
                Text(ByteCountFormatterUtil.string(from: coordinator.sessionStore.totalStorageBytes))
            }
        }
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct ModeCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: Color

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundStyle(tint)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .foregroundStyle(.tertiary)
        }
        .padding(14)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}
