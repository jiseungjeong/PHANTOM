import Foundation

// MARK: - PHANTOM Fusion Engine  (session timing + post-analysis stats)
//
// In the new post-session classification architecture, the engine no longer
// processes real-time spike windows. Its responsibilities are:
//   1. Track session start / end time (drives the live duration timer in SessionView).
//   2. Hold PHANTOMSessionStats, populated by applyResults() after the
//      SessionAnalyzer finishes classifying the session buffer.

@MainActor
final class PHANTOMFusionEngine: ObservableObject {

    @Published private(set) var stats = PHANTOMSessionStats()

    private(set) var anchorTs: TimeInterval = 0

    func startSession(anchor: TimeInterval? = nil) {
        anchorTs = anchor ?? Date().timeIntervalSince1970
        stats    = PHANTOMSessionStats(startTime: Date())
    }

    func stopSession() {
        stats.endTime = .now
    }

    /// Called by BoxingSessionManager once SessionAnalyzer finishes.
    /// Replaces the empty stats with counts derived from the classified sequence.
    func applyResults(_ sequence: DetectedMotionSequence) {
        let start = stats.startTime ?? .now
        let end   = stats.endTime   ?? .now
        stats = PHANTOMSessionStats(from: sequence, startTime: start, endTime: end)
    }

    /// Session-relative milliseconds helper used by BoxingSessionManager.
    func relativeMs(from absoluteTs: TimeInterval) -> Double {
        (absoluteTs - anchorTs) * 1000.0
    }
}
