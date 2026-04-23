import Foundation

// MARK: - Workout History Manager
//
// Responsible for:
//   • Persisting session data after each workout
//   • Daily reset at midnight (or on user request)
//   • Archiving granular records older than 7 days into compact summaries
//   • Providing today / yesterday / 7-day average / 30-day trend data

@MainActor
final class WorkoutHistoryManager: ObservableObject {

    // MARK: - Published state

    @Published private(set) var granularRecords: [DailyRecord] = []  // last 7 days
    @Published private(set) var summaries: [DailySummary]       = []  // indefinite archive

    // MARK: - Private

    private let calendar = Calendar.current
    private var midnightTimer: Timer?

    private var granularURL: URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("workout_granular.json")
    }

    private var summariesURL: URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("workout_summaries.json")
    }

    // MARK: - Init

    init() {
        load()
        archiveExpired()
        scheduleMidnightReset()
    }

    // MARK: - Public API

    /// Call when a session ends. Skips empty sessions.
    func recordSession(_ stats: PHANTOMSessionStats) {
        guard stats.totalPunches > 0 || stats.totalDefenses > 0 else { return }
        let summary = SessionSummary(from: stats)
        let today   = calendar.startOfDay(for: .now)

        if let idx = granularRecords.firstIndex(where: {
            calendar.isDate($0.date, inSameDayAs: today)
        }) {
            granularRecords[idx].sessions.append(summary)
        } else {
            granularRecords.append(DailyRecord(id: UUID(), date: today, sessions: [summary]))
        }
        save()
    }

    /// User-initiated reset: clears all sessions recorded today.
    func resetToday() {
        let today = calendar.startOfDay(for: .now)
        granularRecords.removeAll { calendar.isDate($0.date, inSameDayAs: today) }
        save()
    }

    // MARK: - Computed helpers

    var todayRecord: DailyRecord? {
        granularRecords.first { calendar.isDate($0.date, inSameDayAs: .now) }
    }

    var yesterdayRecord: DailyRecord? {
        guard let yesterday = calendar.date(byAdding: .day, value: -1, to: .now) else { return nil }
        return granularRecords.first { calendar.isDate($0.date, inSameDayAs: yesterday) }
    }

    /// Average metrics over the last 7 days (granular window only).
    var sevenDayAverage: SevenDayAverage? {
        guard let cutoff = calendar.date(byAdding: .day, value: -7, to: .now) else { return nil }
        let recent = granularRecords.filter { $0.date >= cutoff }
        guard !recent.isEmpty else { return nil }
        let n      = Double(recent.count)
        let ttVals = recent.compactMap(\.averageTTms)
        return SevenDayAverage(
            avgPunches:     Double(recent.reduce(0) { $0 + $1.totalPunches })  / n,
            avgDefenses:    Double(recent.reduce(0) { $0 + $1.totalDefenses }) / n,
            avgLeadPunches: Double(recent.reduce(0) { $0 + $1.leadPunches })   / n,
            avgRearPunches: Double(recent.reduce(0) { $0 + $1.rearPunches })   / n,
            avgTTms:        ttVals.isEmpty ? nil : ttVals.reduce(0, +) / Double(ttVals.count),
            dayCount:       recent.count
        )
    }

    /// Merged daily data for the last 30 days (summaries + granular).
    /// Granular takes precedence over archived summary for the same day.
    var monthlyTrendData: [DailySummary] {
        guard let cutoff = calendar.date(byAdding: .day, value: -30, to: .now) else { return [] }
        let archived  = summaries.filter { $0.date >= cutoff }
        let fromGranular = granularRecords
            .filter { $0.date >= cutoff }
            .map    { DailySummary(from: $0) }

        var merged = archived
        for g in fromGranular {
            if !merged.contains(where: { calendar.isDate($0.date, inSameDayAs: g.date) }) {
                merged.append(g)
            } else if let idx = merged.firstIndex(where: {
                calendar.isDate($0.date, inSameDayAs: g.date)
            }) {
                merged[idx] = g     // prefer live granular data
            }
        }
        return merged.sorted { $0.date < $1.date }
    }

    // MARK: - Archiving

    /// Move records older than 7 days from granular storage into compact summaries.
    private func archiveExpired() {
        guard let cutoff = calendar.date(byAdding: .day, value: -7, to: .now) else { return }
        let expired = granularRecords.filter { $0.date < cutoff }
        guard !expired.isEmpty else { return }

        for record in expired {
            if !summaries.contains(where: { calendar.isDate($0.date, inSameDayAs: record.date) }) {
                summaries.append(DailySummary(from: record))
            }
        }
        granularRecords.removeAll { $0.date < cutoff }
        save()
    }

    // MARK: - Midnight reset scheduling

    private func scheduleMidnightReset() {
        midnightTimer?.invalidate()
        guard let midnight = nextMidnight() else { return }
        let interval = midnight.timeIntervalSinceNow
        midnightTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.archiveExpired()
                self?.scheduleMidnightReset()
            }
        }
    }

    private func nextMidnight() -> Date? {
        var components      = calendar.dateComponents([.year, .month, .day], from: .now)
        components.day!    += 1
        components.hour    = 0
        components.minute  = 0
        components.second  = 0
        return calendar.date(from: components)
    }

    // MARK: - Persistence

    private func save() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(granularRecords) {
            try? data.write(to: granularURL)
        }
        if let data = try? encoder.encode(summaries) {
            try? data.write(to: summariesURL)
        }
    }

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let data    = try? Data(contentsOf: granularURL),
           let records = try? decoder.decode([DailyRecord].self, from: data) {
            granularRecords = records
        }
        if let data = try? Data(contentsOf: summariesURL),
           let sums  = try? decoder.decode([DailySummary].self, from: data) {
            summaries = sums
        }
    }
}
