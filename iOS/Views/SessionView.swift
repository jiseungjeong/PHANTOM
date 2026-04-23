import SwiftUI

// MARK: - Session View
//
// Live session dashboard showing:
//   • Lead / Rear punch counts (Watch + iPhone IMU)
//   • Defense gesture count with breakdown
//   • Latest Transition Time (offense → defense)
//   • Average TT for the session
//   • Device connection status

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

// MARK: - Session Start Confirm Overlay

struct SessionStartConfirmOverlay: View {
    let onStart: () -> Void
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()

            VStack(spacing: 28) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 40))
                    .foregroundStyle(.yellow)

                Text("Before you start the session, please ensure you have enough open space around you.")
                    .font(.headline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white)
                    .padding(.horizontal)

                HStack(spacing: 16) {
                    Button(action: onCancel) {
                        Text("Cancel")
                            .font(.headline)
                            .frame(width: 120)
                            .padding()
                            .background(Color.red)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .buttonStyle(.plain)

                    Button(action: onStart) {
                        Text("Start")
                            .font(.headline)
                            .frame(width: 120)
                            .padding()
                            .background(Color.blue)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(32)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 24))
            .padding(24)
        }
    }
}

// MARK: - Safety Banner

struct SafetyBanner: View {
    let onDismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text("Safety Notice")
                    .font(.headline)
                Text("Hold iPhone securely in your rear hand. Ensure sufficient space around you before shadowboxing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(Color.yellow.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Connection Bar

struct ConnectionBar: View {
    let watchConnected:   Bool
    let airPodsAvailable: Bool

    var body: some View {
        HStack(spacing: 20) {
            DeviceIndicator(label: "Watch",   icon: "applewatch",  connected: watchConnected)
            DeviceIndicator(label: "iPhone",  icon: "iphone",       connected: true)
            DeviceIndicator(label: "AirPods", icon: "airpodspro",   connected: airPodsAvailable)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct DeviceIndicator: View {
    let label:     String
    let icon:      String
    let connected: Bool

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.caption)
            Text(label).font(.caption2)
            Circle()
                .fill(connected ? Color.green : Color.gray.opacity(0.5))
                .frame(width: 6, height: 6)
        }
        .foregroundStyle(connected ? .primary : .secondary)
    }
}

// MARK: - Metric Card

struct MetricCard: View {
    let title:   String
    let value:   String
    let icon:    String
    let tint:    Color
    var detail:  String? = nil

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                Image(systemName: icon).foregroundStyle(tint)
                Text(title).font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            Text(value)
                .font(.system(size: 42, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
                .contentTransition(.numericText())
                .frame(maxWidth: .infinity, alignment: .leading)
            if let detail {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Transition Time Card

struct TransitionTimeCard: View {
    let lastTTms:    Double?
    let averageTTms: Double?

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Image(systemName: "arrow.right.circle.fill").foregroundStyle(.cyan)
                Text("Transition Time").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("Offense → Defense")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            HStack(alignment: .lastTextBaseline, spacing: 4) {
                Text(lastTTms.map { "\(Int($0))" } ?? "—")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .foregroundStyle(lastTTms != nil ? Color.cyan : Color.secondary)
                    .contentTransition(.numericText())
                if lastTTms != nil {
                    Text("ms").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let avg = averageTTms {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("avg").font(.caption2).foregroundStyle(.tertiary)
                        Text("\(Int(avg)) ms").font(.caption).fontWeight(.semibold)
                    }
                }
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Last Event Badge

struct LastEventBadge: View {
    let event: BoxingEvent

    private var color: Color {
        event.isPunch ? .blue : .purple
    }
    private var icon: String {
        event.isPunch ? "bolt.fill" : "shield.fill"
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon).foregroundStyle(color)
            Text(event.displayLabel)
                .fontWeight(.semibold)
            if let tt = event.transitionTimeMs {
                Spacer()
                Text("TT \(Int(tt)) ms")
                    .font(.caption)
                    .foregroundStyle(.cyan)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.cyan.opacity(0.15))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(color.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Session Toggle Button

struct SessionToggleButton: View {
    let isActive: Bool
    let action:   () -> Void

    var body: some View {
        Button(action: action) {
            Text(isActive ? "Stop Session" : "Start Session")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(isActive ? Color.red : Color.blue)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}
