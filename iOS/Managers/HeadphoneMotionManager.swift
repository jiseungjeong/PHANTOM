import Foundation
import CoreMotion

// MARK: - Headphone Motion Manager
//
// Wraps CMHeadphoneMotionManager to stream gyroY + accX from AirPods to
// PHANTOMFusionEngine for defense gesture detection.
//
// Supported devices: AirPods Pro (all generations), AirPods Max,
//                    AirPods 3rd gen, AirPods 4 (with ANC).
// Info.plist key: NSHeadphoneMotionUsageDescription — no Xcode capability needed.
//
// IMPORTANT: CMHeadphoneMotionManager requires the AirPods to be the active
// AVAudioSession route.  Call startTracking() after activating the audio session.

@MainActor
final class HeadphoneMotionManager: ObservableObject {

    private let headphoneMotion   = CMHeadphoneMotionManager()
    private let gestureClassifier = HeadGestureClassifier()

    @Published var isAvailable: Bool = false
    @Published var isActive:    Bool = false

    var onDefenseGestureDetected: ((DefenseLabel, Double) -> Void)?

    /// Optional raw-sample hook (Feature D). See PhoneMotionManager.onRawSample.
    var onRawSample: ((TimeSeriesRawSample) -> Void)?

    /// Label stamped on emitted raw samples (ignored by segmentation).
    var rawLabel: String = ""

    init() {
        isAvailable = headphoneMotion.isDeviceMotionAvailable

        gestureClassifier.onGestureDetected = { [weak self] label, tsMs in
            self?.onDefenseGestureDetected?(label, tsMs)
        }
    }

    // MARK: - Control

    func startTracking(anchorTs: TimeInterval) {
        guard !isActive else { return }

        AudioSessionActivator.activateForHeadphoneMotion()

        headphoneMotion.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                print("[HeadphoneMotion] Error: \(error.localizedDescription)")
                return
            }
            guard let motion else { return }

            let ts = Date().timeIntervalSince1970
            // Use wall-clock relative to anchor
            let wallTsMs = (ts - anchorTs) * 1000

            self.gestureClassifier.processSample(
                accX:  motion.userAcceleration.x,
                gyroY: motion.rotationRate.y,
                tsMs:  wallTsMs
            )

            if let onRawSample = self.onRawSample {
                let ua = motion.userAcceleration, grav = motion.gravity, gr = motion.rotationRate
                onRawSample(TimeSeriesRawSample(
                    timestamp: ts, source: "AirPods", label: self.rawLabel,
                    accX: ua.x + grav.x, accY: ua.y + grav.y, accZ: ua.z + grav.z,
                    gyroX: gr.x, gyroY: gr.y, gyroZ: gr.z
                ))
            }
        }
        isActive = true
    }

    func stopTracking() {
        headphoneMotion.stopDeviceMotionUpdates()
        gestureClassifier.reset()
        isActive = false
    }
}
