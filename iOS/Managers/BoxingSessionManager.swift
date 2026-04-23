import Foundation
import Combine

// MARK: - Boxing Session Manager
//
// Orchestrates the three sensor streams and the PHANTOMFusionEngine.
// This is the @MainActor ViewModel that SwiftUI views observe.
//
// Device roles (PHANTOM architecture):
//   Apple Watch  — Lead hand wrist spike detector → raw windows → iPhone
//   iPhone IMU   — Rear hand wrist spike detector (always active, not a fallback)
//   AirPods Pro  — Head gesture stream → PHANTOMFusionEngine

@MainActor
final class BoxingSessionManager: ObservableObject {

    // MARK: - Dependencies

    let fusionEngine       = PHANTOMFusionEngine()
    private let phoneMotion       = PhoneMotionManager()
    private let headphoneMotion   = HeadphoneMotionManager()
    private let watchConnectivity = WatchConnectivityManager.shared

    // MARK: - Published State (forwarded from fusionEngine)

    @Published var isSessionActive  = false
    @Published var showSafetyAlert  = false     // first-launch warning

    @Published var isWatchConnected   = false
    @Published var isAirPodsAvailable = false

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        wireCallbacks()
        bindPublished()

        // Show safety alert on first run
        if !UserDefaults.standard.bool(forKey: "phantomSafetyAlertShown") {
            showSafetyAlert = true
        }
    }

    // MARK: - Session Control

    func startSession() {
        guard !isSessionActive else { return }
        fusionEngine.startSession()
        let anchor = fusionEngine.anchorTs

        phoneMotion.startTracking(anchorTs: anchor)
        headphoneMotion.startTracking(anchorTs: anchor)
        watchConnectivity.sendStartSession(anchorTs: anchor)

        isSessionActive = true
    }

    func stopSession() {
        guard isSessionActive else { return }
        fusionEngine.stopSession()

        phoneMotion.stopTracking()
        headphoneMotion.stopTracking()
        watchConnectivity.sendStopSession()

        isSessionActive = false
    }

    func dismissSafetyAlert() {
        UserDefaults.standard.set(true, forKey: "phantomSafetyAlertShown")
        showSafetyAlert = false
    }

    // MARK: - Wiring

    private func wireCallbacks() {
        // Watch spike windows → Lead hand punches
        watchConnectivity.onWristSpike = { [weak self] window, tsMs in
            self?.fusionEngine.receiveWristSpike(window: window, tsMs: tsMs)
        }

        // iPhone IMU spike windows → Rear hand punches
        phoneMotion.onSpike = { [weak self] window, tsMs in
            self?.fusionEngine.receiveIPhoneSpike(window: window, tsMs: tsMs)
        }

        // AirPods defense gestures
        headphoneMotion.onDefenseGestureDetected = { [weak self] label, tsMs in
            self?.fusionEngine.receiveDefenseGesture(label: label, tsMs: tsMs)
        }

        // Watch-initiated session: start iPhone side using Watch's anchor for timestamp alignment
        watchConnectivity.onSessionStartFromWatch = { [weak self] anchor in
            guard let self, !self.isSessionActive else { return }
            self.fusionEngine.startSession(anchor: anchor)
            self.phoneMotion.startTracking(anchorTs: anchor)
            self.headphoneMotion.startTracking(anchorTs: anchor)
            // Do NOT send sessionStart back to Watch — Watch already started
            self.isSessionActive = true
        }

        // Watch-initiated session stop
        watchConnectivity.onSessionStopFromWatch = { [weak self] in
            guard let self, self.isSessionActive else { return }
            self.fusionEngine.stopSession()
            self.phoneMotion.stopTracking()
            self.headphoneMotion.stopTracking()
            self.isSessionActive = false
        }
    }

    private func bindPublished() {
        watchConnectivity.$isWatchReachable
            .receive(on: RunLoop.main)
            .assign(to: \.isWatchConnected, on: self)
            .store(in: &cancellables)

        headphoneMotion.$isAvailable
            .receive(on: RunLoop.main)
            .assign(to: \.isAirPodsAvailable, on: self)
            .store(in: &cancellables)
    }
}
