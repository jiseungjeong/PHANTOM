import Foundation

// MARK: - Session Summary
// Compact, Codable snapshot of one completed boxing session.
// Derived from PHANTOMSessionStats; stored as part of DailyRecord.

struct SessionSummary: Codable, Identifiable {
    let id: UUID
    let startTime: Date
    let durationSeconds: Double
    let totalPunches: Int
    let totalDefenses: Int
    let leadPunches: Int
    let rearPunches: Int
    let averageTTms: Double?
    let fatigueDriftMs: Double?
    let punchBreakdown: [String: Int]
    let defenseBreakdown: [String: Int]

    init(from stats: PHANTOMSessionStats) {
        id              = UUID()
        startTime       = stats.startTime ?? .now
        durationSeconds = stats.duration  ?? 0
        totalPunches    = stats.totalPunches
        totalDefenses   = stats.totalDefenses
        leadPunches     = stats.leadPunches.count
        rearPunches     = stats.rearPunches.count
        averageTTms     = stats.averageTTms
        fatigueDriftMs  = stats.fatigueDriftMs
        punchBreakdown  = Dictionary(uniqueKeysWithValues:
            stats.punchBreakdown.map { ($0.key.rawValue, $0.value) })
        defenseBreakdown = Dictionary(uniqueKeysWithValues:
            stats.defenseBreakdown.map { ($0.key.rawValue, $0.value) })
    }
}

// MARK: - Daily Record
// Aggregates every session in a single calendar day.
// Kept in granular storage for 7 days, then archived into DailySummary.

struct DailyRecord: Codable, Identifiable {
    let id: UUID
    let date: Date           // Normalized to start-of-day
    var sessions: [SessionSummary]

    var totalPunches:        Int    { sessions.reduce(0) { $0 + $1.totalPunches } }
    var totalDefenses:       Int    { sessions.reduce(0) { $0 + $1.totalDefenses } }
    var leadPunches:         Int    { sessions.reduce(0) { $0 + $1.leadPunches } }
    var rearPunches:         Int    { sessions.reduce(0) { $0 + $1.rearPunches } }
    var sessionCount:        Int    { sessions.count }
    var totalDurationSeconds: Double { sessions.reduce(0) { $0 + $1.durationSeconds } }

    var averageTTms: Double? {
        let vals = sessions.compactMap(\.averageTTms)
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }
}

// MARK: - Daily Summary
// Lightweight archive record kept indefinitely for the monthly trend.
// Created from DailyRecord when granular data expires (>7 days).

struct DailySummary: Codable, Identifiable {
    let id: UUID
    let date: Date
    let totalPunches: Int
    let totalDefenses: Int
    let leadPunches: Int
    let rearPunches: Int
    let averageTTms: Double?
    let sessionCount: Int

    init(from record: DailyRecord) {
        id            = UUID()
        date          = record.date
        totalPunches  = record.totalPunches
        totalDefenses = record.totalDefenses
        leadPunches   = record.leadPunches
        rearPunches   = record.rearPunches
        averageTTms   = record.averageTTms
        sessionCount  = record.sessionCount
    }
}

// MARK: - Seven Day Average
// Value type passed to the 7-day average UI card.

struct SevenDayAverage {
    let avgPunches:     Double
    let avgDefenses:    Double
    let avgLeadPunches: Double
    let avgRearPunches: Double
    let avgTTms:        Double?
    let dayCount:       Int
}
