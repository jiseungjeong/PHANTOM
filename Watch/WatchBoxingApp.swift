import SwiftUI

@main
struct WatchBoxingApp: App {
    @StateObject private var manager = WatchBoxingManager()

    var body: some Scene {
        WindowGroup {
            WatchContentView()
                .environmentObject(manager)
        }
    }
}
