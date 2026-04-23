import Foundation
import CoreMotion

// MARK: - Phone Motion Manager  (Rear hand)
//
// iPhone is held in the rear (non-dominant) hand during shadowboxing.
// This manager samples iPhone IMU at 100 Hz, detects spike windows,
// and forwards them to PHANTOMFusionEngine as Rear punch events.
//
// Unlike the previous architecture, the iPhone is NOT a fallback —
// it is always the primary Rear-hand sensor.
// Info.plist key: NSMotionUsageDescription — no Xcode capability needed.

@MainActor
final class PhoneMotionManager: ObservableObject {

    private let motionManager = CMMotionManager()
    private let detector      = SpikeDetector()
    private let updateInterval: TimeInterval = 1.0 / 100.0

    @Published var isActive = false

    var onSpike: (([Double], Double) -> Void)?

    init() {
        detector.onSpike = { [weak self] window, tsMs in
            self?.onSpike?(window, tsMs)
        }
    }

    // MARK: - Control

    func startTracking(anchorTs: TimeInterval) {
        guard motionManager.isDeviceMotionAvailable, !isActive else { return }
        motionManager.deviceMotionUpdateInterval = updateInterval

        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let mag  = accelerationMagnitude(x: motion.userAcceleration.x,
                                             y: motion.userAcceleration.y,
                                             z: motion.userAcceleration.z)
            let tsMs = (Date().timeIntervalSince1970 - anchorTs) * 1000
            self.detector.processSample(magnitude: mag, tsMs: tsMs)
        }
        isActive = true
    }

    func stopTracking() {
        motionManager.stopDeviceMotionUpdates()
        detector.reset()
        isActive = false
    }
}
