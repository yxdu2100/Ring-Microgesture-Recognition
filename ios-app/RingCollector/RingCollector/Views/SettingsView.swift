import SwiftUI

struct SettingsView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var settings = settings

        NavigationStack {
            Form {
                Section("Study") {
                    TextField("Participant ID", text: $settings.participantID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Stepper("Reps per block: \(settings.repsPerBlock)", value: $settings.repsPerBlock, in: 1...50)
                    TextField("Gesture set version", text: $settings.gestureSetVersion)
                    TextField("IMU config", text: $settings.imuConfig)
                }

                Section("Gestures") {
                    ForEach(GestureDefinition.defaults) { gesture in
                        Toggle(isOn: gestureBinding(gesture.id)) {
                            VStack(alignment: .leading) {
                                Text(gesture.name)
                                Text(gesture.description)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                Section("Guided Mode") {
                    Toggle("Re-don ring prompt between blocks", isOn: $settings.showRedonPrompt)
                }

                Section("BLE Protocol") {
                    LabeledContent("Service", value: "…120002")
                    LabeledContent("Stream control", value: "Command char (1=start, 0=stop)")
                    LabeledContent("Keepalive", value: "Command 3 every 2s")
                    LabeledContent("IMU mode char", value: "Trigger mode (no-op in firmware)")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func gestureBinding(_ id: String) -> Binding<Bool> {
        Binding(
            get: { settings.enabledGestureIDs.contains(id) },
            set: { enabled in
                if enabled {
                    if !settings.enabledGestureIDs.contains(id) {
                        settings.enabledGestureIDs.append(id)
                    }
                } else {
                    settings.enabledGestureIDs.removeAll { $0 == id }
                }
            }
        )
    }
}
