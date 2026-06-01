import SwiftUI

// MARK: - Session View
//
// Live session dashboard showing:
//   • Lead / Rear punch counts (Watch + iPhone IMU)
//   • Defense gesture count with breakdown
//   • Latest Transition Time (offense → defense)
//   • Average TT for the session
//   • Device connection status
//
// Subcomponents live in iOS/Views/Session/.

struct SessionView: View {
    @EnvironmentObject var session: BoxingSessionManager
    @EnvironmentObject var history: WorkoutHistoryManager

    @State private var isShowingStartConfirm = false

    private var stats:    PHANTOMSessionStats { session.fusionEngine.stats }
    private var lastEvent: BoxingEvent?       { session.fusionEngine.lastEvent }

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {

                // Recording-mode selector. Disabled while a session is running so
                // the user can't switch pipelines mid-recording. Only shown when
                // the rule-based classifier loaded successfully.
                if session.isRuleBasedAvailable {
                    Picker("Mode", selection: $session.mode) {
                        ForEach(BoxingSessionManager.SessionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(session.isSessionActive)
                }

                ConnectionBar(
                    watchConnected:   session.isWatchConnected,
                    airPodsAvailable: session.isAirPodsAvailable
                )

                // Safety warning (first run)
                if session.showSafetyAlert {
                    SafetyBanner { session.dismissSafetyAlert() }
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                if session.mode == .ruleBased {
                    ruleBasedContent
                } else {
                    liveCountContent
                }

                Spacer(minLength: 20)

                SessionToggleButton(isActive: session.isSessionActive) {
                    if session.isSessionActive {
                        session.stopSession()
                        // Only the live-count pipeline produces a history entry.
                        if session.mode == .liveCount {
                            history.recordSession(session.fusionEngine.stats)
                        }
                    } else {
                        isShowingStartConfirm = true
                    }
                }
            }
            .padding()
        }
        .animation(.easeInOut(duration: 0.3), value: session.fusionEngine.lastEvent?.id)
        .animation(.easeInOut(duration: 0.25), value: session.showSafetyAlert)
        .animation(.easeInOut(duration: 0.25), value: session.mode)
        .animation(.easeInOut(duration: 0.25), value: session.isClassifying)
        .animation(.easeInOut(duration: 0.25), value: session.classification?.label)
        .animation(.easeInOut(duration: 0.25), value: session.isSegmenting)
        .animation(.easeInOut(duration: 0.25), value: session.motionSequence?.items.count)
        .overlay {
            if isShowingStartConfirm {
                SessionStartConfirmOverlay(
                    onStart: {
                        isShowingStartConfirm = false
                        session.startSession()
                    },
                    onCancel: {
                        isShowingStartConfirm = false
                    }
                )
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isShowingStartConfirm)
    }

    // MARK: - Live Count Content

    @ViewBuilder
    private var liveCountContent: some View {
        // Session duration — shown above the live-count cards. Ticks live while
        // the session is active; freezes at the final value after stop.
        durationCard

        // Punch row: Lead + Rear
        HStack(spacing: 12) {
            MetricCard(
                title:   "Lead",
                value:   "\(stats.leadPunches.count)",
                icon:    "hand.point.left.fill",
                tint:    .blue
            )
            MetricCard(
                title:   "Rear",
                value:   "\(stats.rearPunches.count)",
                icon:    "hand.point.right.fill",
                tint:    .orange
            )
        }

        // Defense row — counted from the segmented single-motion sequence
        // (slips, rolls, step, guard), available after the session stops.
        MetricCard(
            title:   "Defense",
            value:   defenseValue,
            icon:    "shield.fill",
            tint:    .purple,
            detail:  defenseDetail
        )

        // Last event indicator
        if let event = lastEvent {
            LastEventBadge(event: event)
                .transition(.scale.combined(with: .opacity))
                .id(event.id)
        }

        // Detected single-motion sequence — below the live-count cards.
        // The full ordered list lives in the Stats tab; this is a summary.
        singleMotionSummary
    }

    // MARK: - Duration (Feature D)

    @ViewBuilder
    private var durationCard: some View {
        if session.isSessionActive {
            // Tick once per second while active.
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                MetricCard(
                    title: "Duration",
                    value: formattedDuration(stats.duration ?? 0),
                    icon:  "timer",
                    tint:  .teal
                )
            }
        } else if let dur = stats.duration, dur > 0 {
            MetricCard(
                title: "Duration",
                value: formattedDuration(dur),
                icon:  "timer",
                tint:  .teal,
                detail: "Last session"
            )
        }
    }

    private func formattedDuration(_ seconds: TimeInterval) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    // MARK: - Single-motion summary (Feature D)

    @ViewBuilder
    private var singleMotionSummary: some View {
        if !session.isRuleBasedAvailable {
            EmptyView()
        } else if session.isSegmenting {
            MetricCard(
                title:  "Detected Motions",
                value:  "Analyzing…",
                icon:   "list.bullet.rectangle",
                tint:   .green,
                detail: "Breaking the session into single motions."
            )
        } else if let seq = session.motionSequence {
            MetricCard(
                title:  "Detected Motions",
                value:  "\(seq.items.count)",
                icon:   "list.bullet.rectangle",
                tint:   .green,
                detail: seq.isEmpty ? "No motions detected." : "See the full sequence in Stats."
            )
        }
    }

    // MARK: - Rule-based Content

    @ViewBuilder
    private var ruleBasedContent: some View {
        if session.isSessionActive {
            MetricCard(
                title:  "Detected Motion",
                value:  "Recording…",
                icon:   "waveform.path",
                tint:   .green,
                detail: "Perform one motion, then Stop to classify."
            )
        } else if session.isClassifying {
            MetricCard(
                title:  "Detected Motion",
                value:  "Analyzing…",
                icon:   "waveform.path.ecg",
                tint:   .green
            )
        } else if let result = session.classification {
            MetricCard(
                title:  "Detected Motion",
                value:  result.label,
                icon:   "figure.boxing",
                tint:   .green,
                detail: "confidence \(Int((result.confidence * 100).rounded()))% · \(result.sourceSignature)"
            )
        } else {
            MetricCard(
                title:  "Detected Motion",
                value:  "—",
                icon:   "figure.boxing",
                tint:   .green,
                detail: "Start a session, perform one motion, then Stop."
            )
        }
    }

    // MARK: - Defense (from segmented single motions, Feature E)

    /// "—" until the session stops and segmentation fills in the count.
    private var defenseValue: String {
        guard let count = session.detectedDefenseCount else { return "—" }
        return "\(count)"
    }

    private var defenseDetail: String? {
        guard let seq = session.motionSequence, !session.isSegmenting else { return nil }
        let bd = seq.defenseBreakdown
        guard !bd.isEmpty else { return nil }
        return bd.sorted { $0.value > $1.value }
            .prefix(2)
            .map { "\($0.key) ×\($0.value)" }
            .joined(separator: "  ")
    }
}
