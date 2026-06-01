import Foundation

// MARK: - Detected Motion Sequence  (Feature D)
//
// The ordered list of single primitive motions extracted from a live-count
// session on stop. Each item corresponds to one event (Watch lead spike,
// iPhone rear spike, or AirPods defense gesture) whose surrounding raw IMU
// window was classified by MotionRuleClassifier.classifySegment.
//
// This is current-session state only — it is not persisted to disk. It lives
// on BoxingSessionManager and is shown in the Stats tab, where it can be
// cleared independently of the cumulative punch/defense records.

struct DetectedMotionSequence {

    struct Item: Identifiable {
        let id = UUID()
        let label:      String   // single-motion label, e.g. "Lead Jab"
        let tsMs:       Double    // session-relative timestamp of the event center
        let confidence: Double
    }

    var items: [Item] = []

    var isEmpty: Bool { items.isEmpty }

    // MARK: - Defense aggregation (Feature E)

    /// Single-motion labels that count as defense. The current model emits 10
    /// single actions; the defensive ones are slips and rolls (Step/Guard are
    /// not in the model's label set). Punches (jab/hook/uppercut) are excluded.
    static let defenseLabels: Set<String> = [
        "Left Slip", "Right Slip", "Left Roll", "Right Roll",
    ]

    /// Stable display order for the defense breakdown.
    static let defenseLabelOrder: [String] = [
        "Left Slip", "Right Slip", "Left Roll", "Right Roll",
    ]

    /// Number of detected motions that are defensive.
    var defenseCount: Int {
        items.filter { Self.defenseLabels.contains($0.label) }.count
    }

    /// Per-label counts of defensive motions in this sequence.
    var defenseBreakdown: [String: Int] {
        var d: [String: Int] = [:]
        for item in items where Self.defenseLabels.contains(item.label) {
            d[item.label, default: 0] += 1
        }
        return d
    }
}
