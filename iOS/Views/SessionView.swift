import SwiftUI

// MARK: - Session View
//
// Live session dashboard.
//
// During a session:  duration timer + "Recording in progress" indicator.
// After stop:        "Analyzing session…" spinner while SessionAnalyzer runs.
// After analysis:    motion counts (punches / defense) from the classified sequence.
//
// The full labeled action list lives in Stats (NavigationLink in the toolbar).

struct SessionView: View {
    @EnvironmentObject var session: BoxingSessionManager
    @EnvironmentObject var history: WorkoutHistoryManager

    @State private var isShowingStartConfirm = false

    private var stats: PHANTOMSessionStats { session.fusionEngine.stats }

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {

                ConnectionBar(
                    watchConnected:   session.isWatchConnected,
                    airPodsAvailable: session.isAirPodsAvailable
                )

                if session.showSafetyAlert {
                    SafetyBanner { session.dismissSafetyAlert() }
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                sessionContent

                Spacer(minLength: 20)

                SessionToggleButton(isActive: session.isSessionActive) {
                    if session.isSessionActive {
                        session.stopSession()
                    } else {
                        isShowingStartConfirm = true
                    }
                }
            }
            .padding()
        }
        .animation(.easeInOut(duration: 0.25), value: session.showSafetyAlert)
        .animation(.easeInOut(duration: 0.25), value: session.isSegmenting)
        .animation(.easeInOut(duration: 0.25), value: session.motionSequence?.items.count)
        .overlay {
            if isShowingStartConfirm {
                SessionStartConfirmOverlay(
                    onStart:  { isShowingStartConfirm = false; session.startSession() },
                    onCancel: { isShowingStartConfirm = false }
                )
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isShowingStartConfirm)
    }

    // MARK: - Main Content

    @ViewBuilder
    private var sessionContent: some View {
        durationCard

        // Demo toggle: single-only vs combo-augmented KNN reference (fast-transition).
        // featureDim is identical so user calibration carries across the toggle.
        if session.comboAvailable && !session.isSessionActive {
            Toggle(isOn: $session.usingCombo) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Combo-augmented reference")
                        .font(.subheadline).fontWeight(.medium)
                    Text(session.usingCombo
                         ? "single + combo windows (fast transitions)"
                         : "single-only")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding()
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }

        if session.isSessionActive {
            MetricCard(
                title:  "Status",
                value:  "Recording…",
                icon:   "waveform.path",
                tint:   .green,
                detail: "Stop the session to classify actions."
            )
        } else if session.isSegmenting {
            MetricCard(
                title:  "Analyzing",
                value:  "Please wait…",
                icon:   "waveform.path.ecg",
                tint:   .orange,
                detail: "Identifying punches and defenses."
            )
        } else if let seq = session.motionSequence {
            resultsCards(seq)
        } else if session.classifier == nil {
            MetricCard(
                title:  "Classifier",
                value:  "Unavailable",
                icon:   "exclamationmark.triangle",
                tint:   .red,
                detail: "knn_params.json not found in app bundle."
            )
        }
    }

    // MARK: - Results (shown after analysis)

    @ViewBuilder
    private func resultsCards(_ seq: DetectedMotionSequence) -> some View {
        HStack(spacing: 12) {
            MetricCard(
                title: "Lead",
                value: "\(seq.leadPunchCount)",
                icon:  "hand.point.left.fill",
                tint:  .blue
            )
            MetricCard(
                title: "Rear",
                value: "\(seq.rearPunchCount)",
                icon:  "hand.point.right.fill",
                tint:  .orange
            )
        }

        MetricCard(
            title:  "Defense",
            value:  "\(seq.defenseCount)",
            icon:   "shield.fill",
            tint:   .purple,
            detail: defenseDetail(seq)
        )

        MetricCard(
            title:  "Total Motions",
            value:  "\(seq.items.count)",
            icon:   "list.bullet.rectangle",
            tint:   .green,
            detail: seq.isEmpty ? "No motions detected." : "Full sequence in Stats →"
        )
    }

    // MARK: - Duration

    @ViewBuilder
    private var durationCard: some View {
        if session.isSessionActive {
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
                title:  "Duration",
                value:  formattedDuration(dur),
                icon:   "timer",
                tint:   .teal,
                detail: "Last session"
            )
        }
    }

    // MARK: - Helpers

    private func defenseDetail(_ seq: DetectedMotionSequence) -> String? {
        let bd = seq.defenseBreakdown
        guard !bd.isEmpty else { return nil }
        return bd.sorted { $0.value > $1.value }
            .prefix(2)
            .map { "\($0.key) ×\($0.value)" }
            .joined(separator: "  ")
    }

    private func formattedDuration(_ seconds: TimeInterval) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
