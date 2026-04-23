import Foundation
import CoreMotion
import AVFoundation

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

    init() {
        isAvailable = headphoneMotion.isDeviceMotionAvailable

        gestureClassifier.onGestureDetected = { [weak self] label, tsMs in
            self?.onDefenseGestureDetected?(label, tsMs)
        }
    }

    // MARK: - Control

    func startTracking(anchorTs: TimeInterval) {
        guard !isActive else { return }

        // Activate the audio session so CMHeadphoneMotionManager can establish the
        // Bluetooth route.  This is required on all supported AirPods models.
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .default,
                                      options: [.allowBluetoothHFP, .allowBluetoothA2DP])
        try? audioSession.setActive(true)

        headphoneMotion.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                print("[HeadphoneMotion] Error: \(error.localizedDescription)")
                return
            }
            guard let motion else { return }

            // Use wall-clock relative to anchor
            let wallTsMs = (Date().timeIntervalSince1970 - anchorTs) * 1000

            self.gestureClassifier.processSample(
                accX:  motion.userAcceleration.x,
                gyroY: motion.rotationRate.y,
                tsMs:  wallTsMs
            )
        }
        isActive = true
    }

    func stopTracking() {
        headphoneMotion.stopDeviceMotionUpdates()
        gestureClassifier.reset()
        isActive = false
    }
}
