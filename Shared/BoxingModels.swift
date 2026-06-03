import Foundation

// MARK: - Punch Labels  (6 classes: Lead/Rear × Jab/Hook/Uppercut)
// Phase 1: Watch sends undifferentiated spikes → .leadPunch / .rearPunch
// Phase 2: DTW classification fills in specific labels

enum PunchLabel: String, Codable, CaseIterable {
    case leadJab      = "Lead Jab"
    case leadHook     = "Lead Hook"
    case leadUppercut = "Lead Uppercut"
    case rearJab      = "Rear Jab"
    case rearHook     = "Rear Hook"
    case rearUppercut = "Rear Uppercut"
    case leadPunch    = "Lead"   // Phase 1 undifferentiated
    case rearPunch    = "Rear"   // Phase 1 undifferentiated

    var isLead: Bool {
        [.leadJab, .leadHook, .leadUppercut, .leadPunch].contains(self)
    }
}

// MARK: - Defense Labels  (6 classes)

enum DefenseLabel: String, Codable, CaseIterable {
    case guard_    = "Guard"
    case leftSlip  = "Left Slip"
    case rightSlip = "Right Slip"
    case leftRoll  = "Left Roll"
    case rightRoll = "Right Roll"
    case step      = "Step"
}

// MARK: - Combination Labels  (multi-punch sequences for dataset collection)

enum CombinationLabel: String, Codable, CaseIterable {
    case leadJabRearJab              = "Lead Jab-Rear Jab"
    case leadJabLeadJab              = "Lead Jab-Lead Jab"
    case leadHookRearHook            = "Lead Hook-Rear Hook"
    case rearHookLeadHook            = "Rear Hook-Lead Hook"
    case leadJabRearHook             = "Lead Jab-Rear Hook"
    case rearUppercutLeadHook        = "Rear Uppercut-Lead Hook"
    case leadJabRightSlipRearJab     = "Lead Jab-Right Slip-Rear Jab"
    case leftRollLeadHookRearHook    = "Left Roll-Lead Hook-Rear Hook"
    case rightRollRearHookLeadHook   = "Right Roll-Rear Hook-Lead Hook"
    case leadJabRightSlipRearHook    = "Lead Jab-Right Slip-Rear Hook"

    var displayName: String {
        rawValue
    }

}

// MARK: - Boxing Event

struct BoxingEvent: Identifiable {
    let id    = UUID()
    let tsMs: Double
    var transitionTimeMs: Double?

    enum Kind {
        case punch(PunchLabel)
        case defense(DefenseLabel)
    }
    let kind: Kind

    var isPunch:   Bool { if case .punch   = kind { return true }; return false }
    var isDefense: Bool { if case .defense = kind { return true }; return false }

    var displayLabel: String {
        switch kind {
        case .punch(let l):   return l.rawValue
        case .defense(let l): return l.rawValue
        }
    }
}

// MARK: - Session Statistics

struct PHANTOMSessionStats {
    var events:    [BoxingEvent] = []
    var startTime: Date?
    var endTime:   Date?

    var totalPunches:  Int { events.filter(\.isPunch).count }
    var totalDefenses: Int { events.filter(\.isDefense).count }

    var leadPunches: [BoxingEvent] {
        events.filter {
            if case .punch(let l) = $0.kind { return l.isLead }; return false
        }
    }
    var rearPunches: [BoxingEvent] {
        events.filter {
            if case .punch(let l) = $0.kind { return !l.isLead }; return false
        }
    }

    var transitionTimes: [Double] { events.compactMap(\.transitionTimeMs) }
    var averageTTms: Double? {
        let tts = transitionTimes
        guard !tts.isEmpty else { return nil }
        return tts.reduce(0, +) / Double(tts.count)
    }

    var duration: TimeInterval? {
        guard let s = startTime else { return nil }
        return (endTime ?? .now).timeIntervalSince(s)
    }

    var punchBreakdown: [PunchLabel: Int] {
        var d: [PunchLabel: Int] = [:]
        for e in events {
            if case .punch(let l) = e.kind { d[l, default: 0] += 1 }
        }
        return d
    }

    var defenseBreakdown: [DefenseLabel: Int] {
        var d: [DefenseLabel: Int] = [:]
        for e in events {
            if case .defense(let l) = e.kind { d[l, default: 0] += 1 }
        }
        return d
    }

    // Fatigue: late-session TT minus early-session TT (positive = guard returning slower)
    var fatigueDriftMs: Double? {
        let tts = transitionTimes
        guard tts.count >= 4 else { return nil }
        let mid   = tts.count / 2
        let early = tts.prefix(mid)
        let late  = tts.suffix(tts.count - mid)
        let earlyAvg = early.reduce(0, +) / Double(early.count)
        let lateAvg  = late.reduce(0, +)  / Double(late.count)
        return lateAvg - earlyAvg
    }
}

#if os(iOS)
extension PHANTOMSessionStats {
    /// Build stats from a post-session DetectedMotionSequence.
    /// Maps KNN string labels back to PunchLabel / DefenseLabel enums.
    ///
    /// iOS-only: `DetectedMotionSequence` is iOS-side current-session state and
    /// is not part of the Watch target, which only emits raw spikes.
    ///
    /// Declared in an extension so the compiler still synthesizes the memberwise
    /// `init(events:startTime:endTime:)` used elsewhere (e.g. startSession).
    init(from sequence: DetectedMotionSequence, startTime: Date, endTime: Date) {
        self.startTime = startTime
        self.endTime   = endTime
        self.events    = sequence.items.compactMap { item in
            if let punch = PunchLabel(rawValue: item.label) {
                return BoxingEvent(tsMs: item.tsMs, kind: .punch(punch))
            }
            if let defense = DefenseLabel(rawValue: item.label) {
                return BoxingEvent(tsMs: item.tsMs, kind: .defense(defense))
            }
            return nil
        }
    }
}
#endif
