import SwiftUI

struct StatsView: View {
    @EnvironmentObject var session: BoxingSessionManager
    @EnvironmentObject var history: WorkoutHistoryManager

    private var stats: PHANTOMSessionStats { session.fusionEngine.stats }

    @State private var showResetConfirm = false

    var body: some View {
        List {

            // MARK: - Current Session Overview
            Section("Session Overview") {
                LabeledContent("Total Punches") {
                    Text("\(stats.totalPunches)").fontWeight(.semibold)
                }
                LabeledContent("Lead Punches") {
                    Text("\(stats.leadPunches.count)").fontWeight(.semibold)
                }
                LabeledContent("Rear Punches") {
                    Text("\(stats.rearPunches.count)").fontWeight(.semibold)
                }
                LabeledContent("Total Defense") {
                    Text("\(stats.totalDefenses)").fontWeight(.semibold)
                }
                if let dur = stats.duration {
                    LabeledContent("Duration") {
                        Text(formatted(dur)).fontWeight(.semibold)
                    }
                }
            }

            // MARK: - Transition Time
            Section("Transition Time") {
                if let avg = stats.averageTTms {
                    LabeledContent("Average TT") {
                        Text("\(Int(avg)) ms").fontWeight(.semibold)
                    }
                    LabeledContent("Samples") {
                        Text("\(stats.transitionTimes.count)").fontWeight(.semibold)
                    }
                } else {
                    Text("No transitions recorded yet.")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                }
                if let drift = stats.fatigueDriftMs {
                    LabeledContent("Fatigue Drift") {
                        HStack {
                            Text("\(Int(drift)) ms")
                                .fontWeight(.semibold)
                                .foregroundStyle(drift > 50 ? .orange : .green)
                            if drift > 50 {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                        }
                    }
                }
            }

            // MARK: - Defense Breakdown
            Section("Defense Breakdown") {
                let bd = stats.defenseBreakdown
                if bd.isEmpty {
                    Text("No defense events recorded.")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                } else {
                    ForEach(DefenseLabel.allCases, id: \.self) { label in
                        let count = bd[label] ?? 0
                        if count > 0 {
                            LabeledContent(label.rawValue) {
                                Text("\(count)").fontWeight(.semibold)
                            }
                        }
                    }
                }
            }

            // MARK: - Recent Events (current session only, not persisted)
            if !stats.events.isEmpty {
                Section("Recent Events") {
                    ForEach(stats.events.suffix(20).reversed()) { event in
                        HStack {
                            Image(systemName: event.isPunch ? "bolt.fill" : "shield.fill")
                                .foregroundStyle(event.isPunch ? .blue : .purple)
                                .frame(width: 20)
                            Text(event.displayLabel).fontWeight(.medium)
                            Spacer()
                            if let tt = event.transitionTimeMs {
                                Text("TT \(Int(tt)) ms")
                                    .font(.caption)
                                    .foregroundStyle(.cyan)
                            }
                        }
                    }
                }
            }

            // MARK: - Today's Cumulative Records
            Section {
                if let today = history.todayRecord {
                    LabeledContent("Sessions Today") {
                        Text("\(today.sessionCount)").fontWeight(.semibold)
                    }
                    LabeledContent("Total Punches") {
                        Text("\(today.totalPunches)").fontWeight(.semibold)
                    }
                    LabeledContent("Lead Punches") {
                        Text("\(today.leadPunches)").fontWeight(.semibold)
                    }
                    LabeledContent("Rear Punches") {
                        Text("\(today.rearPunches)").fontWeight(.semibold)
                    }
                    LabeledContent("Total Defense") {
                        Text("\(today.totalDefenses)").fontWeight(.semibold)
                    }
                    if let tt = today.averageTTms {
                        LabeledContent("Avg TT (all sessions)") {
                            Text("\(Int(tt)) ms").fontWeight(.semibold)
                        }
                    }
                    if today.totalDurationSeconds > 0 {
                        LabeledContent("Total Duration") {
                            Text(formatted(today.totalDurationSeconds)).fontWeight(.semibold)
                        }
                    }

                    // Per-session breakdown
                    if today.sessionCount > 1 {
                        ForEach(Array(today.sessions.enumerated()), id: \.element.id) { idx, s in
                            LabeledContent("Session \(idx + 1)") {
                                Text("\(s.totalPunches) punches · \(s.totalDefenses) def")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } else {
                    Text("No sessions recorded today.")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                }
            } header: {
                Text("Today's Cumulative Records")
            } footer: {
                Text("Aggregates all sessions completed today. Resets at midnight or on manual reset.")
                    .font(.caption2)
            }
        }
        .navigationTitle("Statistics")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showResetConfirm = true
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                }
                .tint(.red)
                .disabled(history.todayRecord == nil)
            }
        }
        .confirmationDialog(
            "Reset today's cumulative records?",
            isPresented: $showResetConfirm,
            titleVisibility: .visible
        ) {
            Button("Reset Today", role: .destructive) {
                history.resetToday()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will delete all session records for today. This cannot be undone.")
        }
    }

    private func formatted(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }
}
