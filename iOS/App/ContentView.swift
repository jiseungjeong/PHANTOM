import SwiftUI

struct ContentView: View {
    @StateObject private var historyManager = WorkoutHistoryManager()
    @StateObject private var sessionManager = BoxingSessionManager()
    @StateObject private var tsCollector    = TimeSeriesCollectManager()

    var body: some View {
        TabView {
            // MARK: - Session Tab
            NavigationStack {
                SessionView()
                    .environmentObject(sessionManager)
                    .environmentObject(historyManager)
                    .navigationTitle("Boxing Tracker")
                    .toolbar {
                        ToolbarItem(placement: .navigationBarLeading) {
                            NavigationLink("Calibrate") {
                                CalibrationView()
                                    .environmentObject(sessionManager)
                            }
                        }
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

            // MARK: - None Calibration Tab (Phase 9 (2))
            // Replaces the legacy Collect tab (DataCollectView): raw data is now
            // gathered via the Raw Stream tab (HAR format). This tab teaches the
            // model to reject non-boxing motion. DataCollectManager itself is kept
            // (its label constants are used by FilesView / TimeSeriesCollectManager).
            NavigationStack {
                NoneCalibrationView()
                    .environmentObject(sessionManager)
            }
            .tabItem { Label("Reject", systemImage: "hand.raised.slash") }

            // MARK: - Time Series Tab
            TimeSeriesCollectView()
                .environmentObject(tsCollector)
                .tabItem { Label("Raw Stream", systemImage: "waveform.path") }

            // MARK: - Files Tab
            FilesView()
                .tabItem { Label("Files", systemImage: "folder") }
        }
        .onAppear {
            // Wire the post-analysis callback once at app launch.
            sessionManager.onAnalysisComplete = { [weak historyManager] stats in
                historyManager?.recordSession(stats)
            }
        }
    }
}
