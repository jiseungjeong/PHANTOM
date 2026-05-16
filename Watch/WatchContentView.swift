import SwiftUI

struct WatchContentView: View {
    @EnvironmentObject var manager: WatchBoxingManager

    var body: some View {
        switch manager.hkAuthState {
        case .notDetermined, .requesting:
            initializingView
        case .denied:
            permissionGateView
        case .granted:
            mainView
        }
    }

    // MARK: - Initializing (very brief, during HK auth request)

    private var initializingView: some View {
        VStack(spacing: 8) {
            ProgressView()
            Text("Initializing…")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Permission gate (HK denied)

    private var permissionGateView: some View {
        VStack(spacing: 8) {
            Image(systemName: "heart.text.square.fill")
                .font(.system(size: 28))
                .foregroundStyle(.red)
            Text("HealthKit Required")
                .font(.headline)
            Text("PHANTOM uses a workout session to keep wrist sensors running. Grant HealthKit access to continue.")
                .font(.system(size: 10))
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button {
                manager.retryHealthKitAuthorization()
            } label: {
                Text("Grant Permission")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            // watchOS won't re-present the system dialog after a denial, so
            // surface the Settings path explicitly.
            Text("If denied, enable it in Settings → Privacy → Health.")
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Main display (granted)

    private var mainView: some View {
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

            Spacer(minLength: 8)

            // Lead punch spike count
            Text("\(manager.spikeCount)")
                .font(.system(size: 52, weight: .bold, design: .rounded))
                .foregroundStyle(.blue)
                .contentTransition(.numericText())
                .animation(.spring(duration: 0.3), value: manager.spikeCount)

            Text(statusLabel)
                .font(.headline)
                .foregroundStyle(.secondary)

            Spacer(minLength: 4)

            // No buttons — iPhone drives Start/Stop. workout ends when the
            // user kills the Watch app from the dock.
            Text("Use iPhone to start")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Derived state

    private var statusLabel: String {
        if manager.isSessionActive { return "Lead Spikes"      }
        if manager.isRawStreaming  { return "Recording"        }
        if manager.isWorkoutActive { return "Ready"            }
        return "Starting workout…"
    }
}
