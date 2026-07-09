import SwiftUI

@main
struct RingCollectorApp: App {
    @State private var coordinator = RecordingCoordinator()
    @State private var settings = AppSettings()

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environment(coordinator)
                .environment(settings)
                .onAppear { FeedbackManager.prepare() }
        }
    }
}
