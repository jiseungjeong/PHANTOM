import SwiftUI

struct WatchContentView: View {
    @EnvironmentObject var manager: WatchBoxingManager

    var body: some View {
        VStack(spacing: 6) {

            // Phone connection indicator — treat active session as "linked",
            // because the workout keep-alive guarantees the channel stays open
            // even when WCSession.isReachable momentarily flips during wrist-down.
            HStack(spacing: 4) {
                let linked = manager.isSessionActive || manager.isPhoneReachable
                Circle()
                    .fill(linked ? Color.green : Color.gray)
                    .frame(width: 6, height: 6)
                Text(linked ? "iPhone" : "Offline")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                if manager.isWorkoutActive {
                    Text("• HK")
                        .font(.system(size: 10))
                        .foregroundStyle(.green)
                }
                Spacer()
            }

            // Lead punch spike count
            Text("\(manager.spikeCount)")
                .font(.system(size: 52, weight: .bold, design: .rounded))
                .foregroundStyle(.blue)
                .contentTransition(.numericText())
                .animation(.spring(duration: 0.3), value: manager.spikeCount)

            Text(manager.isSessionActive ? "Lead Spikes" : "Stopped")
                .font(.headline)
                .foregroundStyle(.secondary)

            Spacer(minLength: 4)

            // Session toggle
            Button {
                if manager.isSessionActive {
                    manager.stopSession(notifyPhone: true)
                } else {
                    // When started from Watch, use current time as anchor and notify iPhone
                    manager.startSession(anchorTs: Date().timeIntervalSince1970, notifyPhone: true)
                }
            } label: {
                Text(manager.isSessionActive ? "Stop" : "Start")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(manager.isSessionActive ? .red : .blue)
        }
        .padding(.horizontal, 4)
    }
}
