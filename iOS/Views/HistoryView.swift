import SwiftUI
import Charts

// MARK: - History View
//
// Dashboard showing:
//   • Today's workout summary
//   • Yesterday's workout summary
//   • 7-day average
//   • Monthly trend chart (30 days)

struct HistoryView: View {
    @EnvironmentObject var history: WorkoutHistoryManager

    @State private var showResetConfirm = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    DaySummaryCard(title: "Today",     record: history.todayRecord)
                    DaySummaryCard(title: "Yesterday", record: history.yesterdayRecord)
                    SevenDayCard(avg: history.sevenDayAverage)
                    MonthlyTrendCard(data: history.monthlyTrendData)
                }
                .padding()
            }
            .navigationTitle("History")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showResetConfirm = true
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    .tint(.red)
                }
            }
            .confirmationDialog(
                "Reset today's workout data?",
                isPresented: $showResetConfirm,
                titleVisibility: .visible
            ) {
                Button("Reset Today", role: .destructive) {
                    history.resetToday()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This cannot be undone.")
            }
        }
    }
}

// MARK: - Day Summary Card

private struct DaySummaryCard: View {
    let title:  String
    let record: DailyRecord?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                if let r = record {
                    Text("\(r.sessionCount) session\(r.sessionCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let r = record {
                VStack(spacing: 8) {
                    HStack(spacing: 10) {
                        StatPill(label: "Lead",    value: "\(r.leadPunches)",   tint: .blue)
                        StatPill(label: "Rear",    value: "\(r.rearPunches)",   tint: .orange)
                        StatPill(label: "Defense", value: "\(r.totalDefenses)", tint: .purple)
                        if let tt = r.averageTTms {
                            StatPill(label: "Avg TT", value: "\(Int(tt)) ms", tint: .cyan)
                        }
                    }

                    HStack {
                        Image(systemName: "bolt.fill").foregroundStyle(.secondary).font(.caption2)
                        Text("\(r.totalPunches) total punches")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if r.totalDurationSeconds > 0 {
                            Image(systemName: "clock").foregroundStyle(.secondary).font(.caption2)
                            Text(formatDuration(r.totalDurationSeconds))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("No workout recorded")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 8)
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func formatDuration(_ seconds: Double) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return m > 0 ? "\(m)m \(s)s" : "\(s)s"
    }
}

// MARK: - 7-Day Average Card

private struct SevenDayCard: View {
    let avg: SevenDayAverage?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("7-Day Average").font(.headline)
                Spacer()
                if let a = avg {
                    Text("\(a.dayCount) active day\(a.dayCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let a = avg {
                HStack(spacing: 10) {
                    StatPill(label: "Lead",    value: String(format: "%.1f", a.avgLeadPunches), tint: .blue)
                    StatPill(label: "Rear",    value: String(format: "%.1f", a.avgRearPunches), tint: .orange)
                    StatPill(label: "Defense", value: String(format: "%.1f", a.avgDefenses),    tint: .purple)
                    if let tt = a.avgTTms {
                        StatPill(label: "Avg TT", value: "\(Int(tt)) ms", tint: .cyan)
                    }
                }
            } else {
                Text("Not enough data yet")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 8)
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Monthly Trend Card

private struct MonthlyTrendCard: View {
    let data: [DailySummary]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Monthly Trend").font(.headline)

            if data.isEmpty {
                Text("No data for the past 30 days")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 24)
            } else {
                Chart(data) { day in
                    BarMark(
                        x: .value("Date", day.date, unit: .day),
                        y: .value("Punches", day.totalPunches)
                    )
                    .foregroundStyle(.blue.opacity(0.75))
                    .cornerRadius(3)

                    LineMark(
                        x: .value("Date", day.date, unit: .day),
                        y: .value("Defenses", day.totalDefenses)
                    )
                    .foregroundStyle(.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    .symbol(.circle)
                    .symbolSize(24)
                }
                .frame(height: 160)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: 7)) {
                        AxisValueLabel(format: .dateTime.month().day())
                        AxisGridLine()
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }

                HStack(spacing: 16) {
                    LegendItem(color: .blue.opacity(0.75), label: "Punches")
                    LegendItem(color: .purple,            label: "Defenses")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Shared sub-components

struct StatPill: View {
    let label: String
    let value: String
    let tint:  Color

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.subheadline, design: .rounded))
                .fontWeight(.bold)
                .foregroundStyle(tint)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(tint.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct LegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 12, height: 8)
            Text(label)
        }
    }
}
