import Foundation

// MARK: - Spike Detector
//
// Detects acceleration spikes above a threshold and collects the raw
// magnitude window. Used on both Watch and iPhone.
//
// Watch: calls onSpike → packs window → sends to iPhone via WCSession.
// iPhone: calls onSpike → PHANTOMFusionEngine.receiveIPhoneSpike.
//
// Phase 2: PHANTOMFusionEngine runs DTW on the spike window to classify
// the specific punch label (jab / hook / uppercut).

final class SpikeDetector {

    // MARK: - Configuration

    var triggerThreshold: Double   = 1.2   // g — spike entry
    private let exitRatio: Double  = 0.5   // mag < threshold * exitRatio → spike end
    private let minDurationMs      = 50.0
    private let maxDurationMs      = 300.0

    // MARK: - State

    private var inSpike        = false
    private var spikeStartMs: Double = 0
    private var buffer: [Double]     = []

    // Callback: (rawMagnitudeWindow, sessionRelativeStartTsMs)
    var onSpike: (([Double], Double) -> Void)?

    // MARK: - Public

    /// Feed one sample. tsMs is session-relative milliseconds (Date().timeIntervalSince1970 - anchorTs) * 1000
    func processSample(magnitude: Double, tsMs: Double) {
        if !inSpike {
            guard magnitude > triggerThreshold else { return }
            inSpike      = true
            spikeStartMs = tsMs
            buffer       = [magnitude]
        } else {
            buffer.append(magnitude)
            let duration = tsMs - spikeStartMs
            let ended    = magnitude < triggerThreshold * exitRatio || duration > maxDurationMs
            if ended {
                if duration >= minDurationMs {
                    onSpike?(buffer, spikeStartMs)
                }
                inSpike = false
                buffer  = []
            }
        }
    }

    func reset() {
        inSpike = false
        buffer  = []
    }
}

// MARK: - Magnitude Helper

func accelerationMagnitude(x: Double, y: Double, z: Double) -> Double {
    (x * x + y * y + z * z).squareRoot()
}
