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
        TabView {
            collectionView
                .tabItem {
                    Label("Collect", systemImage: "record.circle")
                }

            LiveInferenceView()
                .tabItem {
                    Label("Live", systemImage: "dot.radiowaves.left.and.right")
                }
        }
    }

    private var collectionView: some View {
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

private struct LiveInferenceView: View {
    @Environment(RecordingCoordinator.self) private var coordinator
    @State private var liveRequested = false

    private let holdDuration: TimeInterval = 1.4

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    connectionPanel
                    liveControls
                    gesturePanel
                    historyPanel
                }
                .padding()
            }
            .navigationTitle("Live Gesture")
            .onAppear {
                if !coordinator.ble.connectionState.isConnected {
                    coordinator.ble.setAutoReconnect(true)
                    coordinator.ble.connect()
                }
            }
        }
    }

    private var connectionPanel: some View {
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
        }
    }

    private var liveControls: some View {
        HStack {
            if coordinator.isActiveSession {
                Label("Recording", systemImage: "record.circle.fill")
                    .foregroundStyle(.red)
            } else if coordinator.ble.isStreaming && liveRequested {
                Button {
                    coordinator.ble.stopStreaming()
                    liveRequested = false
                } label: {
                    Label("Stop Live", systemImage: "stop.fill")
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button {
                    liveRequested = true
                    coordinator.ble.startStreaming()
                } label: {
                    Label("Start Live", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!coordinator.ble.connectionState.isConnected)
            }
            Spacer()
            if let latest = coordinator.ble.latestInference {
                Text(latest.classifier.label)
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.thinMaterial, in: Capsule())
            }
        }
    }

    private var gesturePanel: some View {
        TimelineView(.periodic(from: Date(), by: 0.25)) { context in
            let displayed = displayedGesture(now: context.date)
            VStack(alignment: .leading, spacing: 12) {
                Text("Gesture")
                    .font(.headline)
                Text(displayed?.classLabel ?? "None")
                    .font(.system(size: 38, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentTransition(.numericText())
                if let latest = coordinator.ble.latestInference {
                    LabeledContent("Raw latest") {
                        Text("\(latest.classLabel) · code \(latest.rawCode)")
                            .monospacedDigit()
                    }
                    LabeledContent("Sample") {
                        Text("\(latest.sampleID)")
                            .monospacedDigit()
                    }
                }
            }
            .padding(14)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var historyPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent")
                .font(.headline)
            ForEach(coordinator.ble.recentInferenceResults.prefix(8)) { result in
                HStack {
                    Text(result.classLabel)
                    Spacer()
                    Text("raw \(result.rawCode)")
                        .foregroundStyle(.secondary)
                    Text("#\(result.sampleID)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
                Divider()
            }
        }
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func displayedGesture(now: Date) -> InferenceResult? {
        guard let result = coordinator.ble.recentInferenceResults.first(where: { !$0.isNull }) else {
            return nil
        }

        if now.timeIntervalSince(result.receivedAt) <= holdDuration {
            return result
        }
        return nil
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
