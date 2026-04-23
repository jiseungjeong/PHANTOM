import SwiftUI

struct ContentView: View {
    @StateObject private var historyManager    = WorkoutHistoryManager()
    @StateObject private var sessionManager    = BoxingSessionManager()
    @StateObject private var collector         = DataCollectManager()
    @StateObject private var tsCollector       = TimeSeriesCollectManager()

    var body: some View {
        TabView {
            // MARK: - Session Tab
            NavigationStack {
                SessionView()
                    .environmentObject(sessionManager)
                    .environmentObject(historyManager)
                    .navigationTitle("Boxing Tracker")
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            NavigationLink("Stats") {
                                StatsView()
                                    .environmentObject(sessionManager)
                                    .environmentObject(historyManager)
                            }
                        }
                    }
            }
            .tabItem { Label("Session", systemImage: "figure.boxing") }

            // MARK: - History Tab
            HistoryView()
                .environmentObject(historyManager)
                .tabItem { Label("History", systemImage: "chart.bar.fill") }

            // MARK: - Data Collect Tab  (spike-window labeled collection)
            DataCollectView()
                .environmentObject(collector)
                .tabItem { Label("Collect", systemImage: "waveform.badge.plus") }

            // MARK: - Time Series Tab  (continuous raw stream collection)
            TimeSeriesCollectView()
                .environmentObject(tsCollector)
                .tabItem { Label("Raw Stream", systemImage: "waveform.path") }

            // MARK: - Files Tab
            FilesView()
                .tabItem { Label("Files", systemImage: "folder") }
        }
    }
}
