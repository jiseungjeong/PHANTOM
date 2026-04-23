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

// MARK: - Defense Labels  (6 classes: Sepanosian taxonomy + Step)

enum DefenseLabel: String, Codable, CaseIterable {
    case leftSlip  = "Left Slip"
    case rightSlip = "Right Slip"
    case leftRoll  = "Left Roll"
    case rightRoll = "Right Roll"
    case pullBack  = "Pull Back"
    case step      = "Step"
    case guard_    = "Guard"
}

// MARK: - Combination Labels  (multi-punch sequences for dataset collection)

enum CombinationLabel: String, Codable, CaseIterable {
    case shadowboxing                  = "Shadowboxing"
    case slipCrossHook                 = "Slip - Cross - Hook"
    case slipUpperHookCross            = "Slip - Upper - Hook - Cross"
    case slipHook                      = "Slip - Hook"
    case jabJab                        = "Jab - Jab"
    case jabStraight                   = "Jab - Straight"
    case leadHookLeadUppercut          = "Lead Hook - Lead Uppercut"
    case rearStraightRearUppercut      = "Rear Straight - Rear Uppercut"
    case leadHookRearStraightLeadHook  = "Lead Hook - Rear Straight - Lead Hook"
    case leadUpperRearUpperLeadHook    = "Lead Uppercut - Rear Uppercut - Lead Hook"
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

// MARK: - Watch Message Keys

enum WatchMessageKey {
    // Session lifecycle – iPhone → Watch
    static let action       = "action"
    static let sessionStart = "sessionStart"
    static let sessionStop  = "sessionStop"
    static let anchorTs     = "anchorTs"      // Unix epoch, used to sync timestamps

    // Session lifecycle – Watch → iPhone
    static let watchSessionStart = "watchSessionStart"  // Watch-initiated session start
    static let watchSessionStop  = "watchSessionStop"   // Watch-initiated session stop

    // Spike window (Message-based)
    static let wristSpike   = "wristSpike"
    static let window       = "window"        // [Double] raw magnitude array
    static let tsMs         = "tsMs"          // Double session-relative ms

    // Raw time-series streaming – iPhone → Watch
    static let rawStreamStart = "rawStreamStart"
    static let rawStreamStop  = "rawStreamStop"
    static let label          = "label"       // activity label forwarded to Watch

    // Raw time-series batch – Watch → iPhone
    // Payload: flat [Double] of stride-8 tuples: tsMs, accX, accY, accZ, gyroX, gyroY, gyroZ, (pad)
    static let rawBatch       = "rawBatch"
    static let rawSamples     = "rawSamples"  // key for the flat [Double] array

    // File transfer metadata – Watch → iPhone
    static let fileTag        = "fileTag"
    static let rawWatchFile   = "rawWatchFile"   // value for fileTag identifying a Watch raw CSV
}
