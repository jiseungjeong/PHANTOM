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

    /// Optional raw-sample hook (Feature D). When set, every motion sample is
    /// emitted as a HARApp-format TimeSeriesRawSample (total acceleration incl.
    /// gravity + rotation rate, absolute Unix timestamp) alongside the existing
    /// spike detection. Used to accumulate raw IMU during a live-count session
    /// so the session can be segmented into single motions on stop.
    var onRawSample: ((TimeSeriesRawSample) -> Void)?

    /// The label to stamp on emitted raw samples. Live-count sessions don't have
    /// a fixed label, so this stays empty; segmentation ignores the label column.
    var rawLabel: String = ""

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
            let ts   = Date().timeIntervalSince1970
            let tsMs = (ts - anchorTs) * 1000
            self.detector.processSample(magnitude: mag, tsMs: tsMs)

            if let onRawSample = self.onRawSample {
                let ua = motion.userAcceleration, grav = motion.gravity, gr = motion.rotationRate
                onRawSample(TimeSeriesRawSample(
                    timestamp: ts, source: "iPhone", label: self.rawLabel,
                    accX: ua.x + grav.x, accY: ua.y + grav.y, accZ: ua.z + grav.z,
                    gyroX: gr.x, gyroY: gr.y, gyroZ: gr.z
                ))
            }
        }
        isActive = true
    }

    func stopTracking() {
        motionManager.stopDeviceMotionUpdates()
        detector.reset()
        isActive = false
    }
}
