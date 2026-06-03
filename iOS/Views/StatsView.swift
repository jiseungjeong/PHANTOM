import SwiftUI

struct StatsView: View {
    @EnvironmentObject var session: BoxingSessionManager
    @EnvironmentObject var history: WorkoutHistoryManager

    private var stats: PHANTOMSessionStats { session.fusionEngine.stats }

    @State private var showResetConfirm = false

    var body: some View {
        List {

            // MARK: - Session Overview  (derived from classified motionSequence)
            Section("Session Overview") {
                let seq = session.motionSequence
                LabeledContent("Total Punches") {
                    Text("\(seq?.punchCount ?? stats.totalPunches)").fontWeight(.semibold)
                }
                LabeledContent("Lead Punches") {
                    Text("\(seq?.leadPunchCount ?? stats.leadPunches.count)").fontWeight(.semibold)
                }
                LabeledContent("Rear Punches") {
                    Text("\(seq?.rearPunchCount ?? stats.rearPunches.count)").fontWeight(.semibold)
                }
                LabeledContent("Total Defense") {
                    Text(defenseOverviewValue).fontWeight(.semibold)
                }
                if let dur = stats.duration {
                    LabeledContent("Duration") {
                        Text(formatted(dur)).fontWeight(.semibold)
                    }
                }
            }

            // MARK: - Defense Breakdown
            Section("Defense Breakdown") {
                let bd = session.motionSequence?.defenseBreakdown ?? [:]
                if session.isSegmenting {
                    HStack {
                        ProgressView()
                        Text("Analyzing session…").foregroundStyle(.secondary).font(.caption)
                    }
                } else if bd.isEmpty {
                    Text("No defense motions detected this session.")
                        .foregroundStyle(.secondary).font(.caption)
                } else {
                    ForEach(DetectedMotionSequence.defenseLabelOrder, id: \.self) { label in
                        let count = bd[label] ?? 0
                        if count > 0 {
                            LabeledContent(label) {
                                Text("\(count)").fontWeight(.semibold)
                            }
                        }
                    }
                }
            }

            // MARK: - Punch Breakdown
            if let bd = session.motionSequence?.punchBreakdown, !bd.isEmpty {
                Section("Punch Breakdown") {
                    let punchOrder = ["Lead Jab","Lead Hook","Lead Uppercut",
                                      "Rear Jab","Rear Hook","Rear Uppercut"]
                    ForEach(punchOrder, id: \.self) { label in
                        let count = bd[label] ?? 0
                        if count > 0 {
                            LabeledContent(label) {
                                Text("\(count)").fontWeight(.semibold)
                            }
                        }
                    }
                }
            }

            // MARK: - Detected Motions (full sequence)
            if let seq = session.motionSequence {
                Section {
                    if session.isSegmenting {
                        HStack {
                            ProgressView()
                            Text("Analyzing session…")
                                .foregroundStyle(.secondary).font(.caption)
                        }
                    } else if seq.isEmpty {
                        Text("No motions detected.")
                            .foregroundStyle(.secondary).font(.caption)
                    } else {
                        ForEach(Array(seq.items.enumerated()), id: \.element.id) { idx, item in
                            HStack {
                                Text("\(idx + 1)")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 24, alignment: .trailing)
                                Image(systemName: DetectedMotionSequence.punchLabels.contains(item.label)
                                      ? "bolt.fill" : "shield.fill")
                                    .foregroundStyle(DetectedMotionSequence.punchLabels.contains(item.label)
                                                     ? .blue : .purple)
                                    .frame(width: 20)
                                Text(item.label).fontWeight(.medium)
                                Spacer()
                                Text("\(Int((item.confidence * 100).rounded()))%")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: {
                    HStack {
                        Text("Detected Motions (this session)")
                        Spacer()
                        if !seq.isEmpty {
                            Button("Clear") { session.resetMotionSequence() }
                                .font(.caption).textCase(nil)
                        }
                    }
                } footer: {
                    Text("Actions identified by KNN classifier from the full session buffer, in order.")
                        .font(.caption2)
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
                    if today.totalDurationSeconds > 0 {
                        LabeledContent("Total Duration") {
                            Text(formatted(today.totalDurationSeconds)).fontWeight(.semibold)
                        }
                    }
                    if today.sessionCount > 1 {
                        ForEach(Array(today.sessions.enumerated()), id: \.element.id) { idx, s in
                            LabeledContent("Session \(idx + 1)") {
                                Text("\(s.totalPunches) punches · \(s.totalDefenses) def")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } else {
                    Text("No sessions recorded today.")
                        .foregroundStyle(.secondary).font(.caption)
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
            Button("Reset Today", role: .destructive) { history.resetToday() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will delete all session records for today. This cannot be undone.")
        }
    }

    private var defenseOverviewValue: String {
        guard let count = session.motionSequence?.defenseCount else { return "—" }
        return "\(count)"
    }

    private func formatted(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60; let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }
}
