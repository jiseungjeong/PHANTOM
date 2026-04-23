import Foundation

// MARK: - PHANTOM Fusion Engine
//
// Central fusion hub running on iPhone. Receives three sensor streams:
//   1. Wrist spike windows from Apple Watch   → Lead hand punches
//   2. Wrist spike windows from iPhone IMU    → Rear hand punches
//   3. Head motion samples from AirPods Pro   → Defensive gestures
//
// Fuses all events onto a single session-relative timestamp timeline and
// computes Offense–Defense Transition Time (TT).
//
// Phase 1: threshold-based counting + approximate TT.
// Phase 2: DTW classification of spike windows for specific punch labels.

@MainActor
final class PHANTOMFusionEngine: ObservableObject {

    // MARK: - Published State

    @Published private(set) var stats       = PHANTOMSessionStats()
    @Published private(set) var lastEvent:  BoxingEvent?
    @Published private(set) var lastTTms:   Double?

    // MARK: - Session Anchor

    /// Unix epoch of session start on iPhone. Watch aligns to this.
    private(set) var anchorTs: TimeInterval = 0

    // MARK: - Transition Time State

    private var lastPunchTsMs: Double? = nil

    // MARK: - Session Lifecycle

    /// Start a new session. If `anchor` is provided (Watch-initiated), uses that timestamp
    /// so Watch and iPhone share the same time origin.
    func startSession(anchor: TimeInterval? = nil) {
        anchorTs      = anchor ?? Date().timeIntervalSince1970
        stats         = PHANTOMSessionStats(startTime: Date())
        lastPunchTsMs = nil
        lastEvent     = nil
        lastTTms      = nil
    }

    func stopSession() {
        stats.endTime = .now
    }

    // MARK: - Stream 1: Watch spike window (Lead hand)

    func receiveWristSpike(window: [Double], tsMs: Double) {
        // Phase 1: undifferentiated lead punch
        // Phase 2: replace with DTW → specific PunchLabel
        let label: PunchLabel = .leadPunch
        let event = BoxingEvent(tsMs: tsMs, kind: .punch(label))
        record(event)
    }

    // MARK: - Stream 2: iPhone IMU spike window (Rear hand)

    func receiveIPhoneSpike(window: [Double], tsMs: Double) {
        let label: PunchLabel = .rearPunch
        let event = BoxingEvent(tsMs: tsMs, kind: .punch(label))
        record(event)
    }

    // MARK: - Stream 3: AirPods head motion → defense gesture

    func receiveDefenseGesture(label: DefenseLabel, tsMs: Double) {
        var event = BoxingEvent(tsMs: tsMs, kind: .defense(label))

        // Compute Transition Time: time since last punch
        if let punchTs = lastPunchTsMs {
            let tt = tsMs - punchTs
            // Sanity bounds: human reaction limit (150 ms) to same-attack pairing limit (2 s)
            if tt > 150 && tt < 2000 {
                event.transitionTimeMs = tt
                lastTTms = tt
                lastPunchTsMs = nil
            }
        }

        record(event)
    }

    // MARK: - Private

    private func record(_ event: BoxingEvent) {
        stats.events.append(event)
        lastEvent = event
        if event.isPunch {
            lastPunchTsMs = event.tsMs  // always keep the most recent punch
        }
    }

    // MARK: - Relative Timestamp Helper

    /// Convert an absolute Unix timestamp to session-relative milliseconds.
    func relativeMs(from absoluteTs: TimeInterval) -> Double {
        (absoluteTs - anchorTs) * 1000.0
    }

    /// Current session-relative time in milliseconds.
    var currentTsMs: Double {
        relativeMs(from: Date().timeIntervalSince1970)
    }
}
