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

                ConnectionBar(
                    watchConnected:   session.isWatchConnected,
                    airPodsAvailable: session.isAirPodsAvailable
                )

                // Safety warning (first run)
                if session.showSafetyAlert {
                    SafetyBanner { session.dismissSafetyAlert() }
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

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

                // Defense row
                MetricCard(
                    title:   "Defense",
                    value:   "\(stats.totalDefenses)",
                    icon:    "shield.fill",
                    tint:    .purple,
                    detail:  defenseDetail
                )

                // Transition Time
                TransitionTimeCard(
                    lastTTms:   session.fusionEngine.lastTTms,
                    averageTTms: stats.averageTTms
                )

                // Last event indicator
                if let event = lastEvent {
                    LastEventBadge(event: event)
                        .transition(.scale.combined(with: .opacity))
                        .id(event.id)
                }

                Spacer(minLength: 20)

                SessionToggleButton(isActive: session.isSessionActive) {
                    if session.isSessionActive {
                        session.stopSession()
                        history.recordSession(session.fusionEngine.stats)
                    } else {
                        isShowingStartConfirm = true
                    }
                }
            }
            .padding()
        }
        .animation(.easeInOut(duration: 0.3), value: session.fusionEngine.lastEvent?.id)
        .animation(.easeInOut(duration: 0.25), value: session.showSafetyAlert)
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

    private var defenseDetail: String? {
        guard stats.totalDefenses > 0 else { return nil }
        let bd = stats.defenseBreakdown
        return bd.sorted { $0.value > $1.value }
            .prefix(2)
            .map { "\($0.key.rawValue) ×\($0.value)" }
            .joined(separator: "  ")
    }
}
