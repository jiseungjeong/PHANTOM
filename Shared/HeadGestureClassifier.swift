import Foundation

// MARK: - Head Gesture Classifier
//
// Detects boxing defensive head gestures from AirPods Pro via
// CMHeadphoneMotionManager (gyroY + accX streams).
//
// Phase 1 (this implementation): gyroY threshold-based anomaly detection.
//   - gyroY > +threshold → rightward head movement → .rightSlip (heuristic)
//   - gyroY < -threshold → leftward head movement  → .leftSlip  (heuristic)
//   - slow, sustained movement → .pullBack (heuristic)
//
// Phase 2: Replace processSample return with DTW against bundled templates
//   from EarableBoxingHeadGestureRecognition/Data/dtw/ for accurate 5-class
//   classification (left_slip, right_slip, left_roll, right_roll, pull_back).
//
// Sepanosian (UbiComp '24) uses gyroY at 50 Hz on OpenEarable.
// AirPods Pro delivers ~100 Hz via CMHeadphoneMotionManager.

final class HeadGestureClassifier {

    // MARK: - Configuration

    private let gyroYThreshold: Double   = 1.5   // rad/s — anomaly entry (Sepanosian Client.py)
    private let accXSlipThreshold: Double = 0.3  // g    — lateral head acceleration for slip vs roll
    private let minDurationSamples: Int  = 5
    private let maxDurationSamples: Int  = 100
    private let cooldownSamples: Int     = 25    // ~0.25 s at 100 Hz

    // MARK: - State

    private var sampleBuffer: [(accX: Double, gyroY: Double)] = []
    private let bufferCapacity = 200

    private var inAnomaly        = false
    private var anomalyStart     = 0
    private var sampleCount      = 0
    private var cooldownRemaining = 0

    // Callback: (DefenseLabel, sessionRelativeTsMs)
    var onGestureDetected: ((DefenseLabel, Double) -> Void)?

    // Secondary callback for data collection — also passes raw gyroY window
    var onGestureDetectedWithWindow: ((DefenseLabel, [Double], Double) -> Void)?

    // MARK: - Public

    func processSample(accX: Double, gyroY: Double, tsMs: Double) {
        sampleBuffer.append((accX: accX, gyroY: gyroY))
        if sampleBuffer.count > bufferCapacity { sampleBuffer.removeFirst() }

        if cooldownRemaining > 0 { cooldownRemaining -= 1; return }

        if !inAnomaly {
            if abs(gyroY) > gyroYThreshold {
                inAnomaly    = true
                anomalyStart = sampleBuffer.count - 1
                sampleCount  = 1
            }
        } else {
            sampleCount += 1

            let anomalyEnded = abs(gyroY) < gyroYThreshold || sampleCount > maxDurationSamples
            if anomalyEnded && sampleCount >= minDurationSamples {
                let window = Array(sampleBuffer.suffix(sampleCount))
                if let label = classify(window: window) {
                    let gyroWindow = window.map(\.gyroY)
                    cooldownRemaining = cooldownSamples
                    onGestureDetected?(label, tsMs)
                    onGestureDetectedWithWindow?(label, gyroWindow, tsMs)
                }
                inAnomaly   = false
                sampleCount = 0
            } else if anomalyEnded {
                inAnomaly   = false
                sampleCount = 0
            }
        }
    }

    func reset() {
        sampleBuffer      = []
        inAnomaly         = false
        sampleCount       = 0
        cooldownRemaining = 0
    }

    // MARK: - Phase 1 Heuristic Classification

    private func classify(window: [(accX: Double, gyroY: Double)]) -> DefenseLabel? {
        guard !window.isEmpty else { return nil }

        let gyroValues = window.map(\.gyroY)
        let accXValues = window.map { abs($0.accX) }

        let avgGyroY = gyroValues.reduce(0, +) / Double(gyroValues.count)
        let maxAccX  = accXValues.max() ?? 0
        let duration = Double(window.count)

        // Pull back: slow, bilateral movement (low gyroY amplitude)
        if abs(avgGyroY) < gyroYThreshold * 1.5 && duration > 40 {
            return .pullBack
        }

        // Slip (fast, lateral): lateral accX dominates
        // Roll (slower, rotational): gyroY dominates without strong accX
        if avgGyroY > 0 {
            return maxAccX > accXSlipThreshold ? .rightSlip : .rightRoll
        } else {
            return maxAccX > accXSlipThreshold ? .leftSlip : .leftRoll
        }
    }
}
