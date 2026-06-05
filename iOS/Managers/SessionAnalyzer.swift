import Foundation

// MARK: - Session Analyzer
//
// Post-session pipeline: takes all raw IMU samples buffered during a 1-2 minute
// session and returns a DetectedMotionSequence with every classified action.
//
// Steps (mirrors build_peak_repetition_dataset.py):
//   1. Build a motion envelope: bin 40 ms, fuse Watch/iPhone/AirPods signals,
//      weight 62% gyro + 38% dynamic acceleration, smooth with a 5-bin kernel.
//   2. Find all peaks in the envelope (min 0.5 s separation).
//   3. For each peak: cut a ±0.5 s / +0.8 s window, classify with KNN.

struct SessionAnalyzer {

    let classifier: MotionRuleClassifier

    // Envelope parameters (must match build_peak_repetition_dataset.py constants)
    private static let binS:           Double = 0.04
    private static let smoothBins:     Int    = 5
    private static let minPeakSepS:    Double = 0.50
    private static let windowPreS:     Double = 0.50
    private static let windowPostS:    Double = 0.80

    // MARK: - Public Entry Point

    func analyze(samples: [TimeSeriesRawSample]) -> DetectedMotionSequence {
        guard !samples.isEmpty else { return DetectedMotionSequence(items: []) }

        let bySource = classifier.group(samples)

        // Global t0: earliest timestamp across all sources
        let globalT0 = bySource.values.compactMap { $0.first?.t }.min() ?? 0.0

        let (times, envelope, _) = motionEnvelope(bySource: bySource)
        guard !envelope.isEmpty else { return DetectedMotionSequence(items: []) }

        let peakIndices = findAllPeaks(envelope: envelope)

        var items: [DetectedMotionSequence.Item] = []
        for peakIdx in peakIndices {
            let peakRelT  = times[peakIdx]
            let peakAbsT  = globalT0 + peakRelT
            let lo        = peakAbsT - Self.windowPreS
            let hi        = peakAbsT + Self.windowPostS
            let window    = samples.filter { $0.timestamp >= lo && $0.timestamp <= hi }
            guard let result = classifier.classifySegment(samples: window) else { continue }
            items.append(DetectedMotionSequence.Item(
                label:      result.label,
                tsMs:       peakRelT * 1000.0,
                confidence: result.confidence
            ))
        }
        return DetectedMotionSequence(items: items)
    }

    /// Calibration helper (Phase 5 d): cut the SINGLE strongest segment from a demo
    /// recording (one action). Picks the highest-envelope peak and returns its
    /// ±0.5/+0.8 s window, same windowing as analyze(). Returns nil if no peak.
    func strongestSegment(samples: [TimeSeriesRawSample]) -> [TimeSeriesRawSample]? {
        guard !samples.isEmpty else { return nil }
        let bySource = classifier.group(samples)
        let globalT0 = bySource.values.compactMap { $0.first?.t }.min() ?? 0.0
        let (times, envelope, _) = motionEnvelope(bySource: bySource)
        guard !envelope.isEmpty else { return nil }

        let peaks = findAllPeaks(envelope: envelope)
        // Choose the peak with the largest envelope value; fall back to global argmax.
        let bestIdx: Int
        if let p = peaks.max(by: { envelope[$0] < envelope[$1] }) {
            bestIdx = p
        } else if let g = envelope.indices.max(by: { envelope[$0] < envelope[$1] }) {
            bestIdx = g
        } else {
            return nil
        }
        let peakAbsT = globalT0 + times[bestIdx]
        let lo = peakAbsT - Self.windowPreS
        let hi = peakAbsT + Self.windowPostS
        let window = samples.filter { $0.timestamp >= lo && $0.timestamp <= hi }
        return window.count >= 5 ? window : nil
    }

    // MARK: - Motion Envelope

    private func motionEnvelope(
        bySource: [String: [MotionRuleClassifier.Row]]
    ) -> (times: [Double], envelope: [Double], duration: Double) {

        // Duration = longest source segment from the beginning
        var duration = 0.0
        for rows in bySource.values {
            guard rows.count >= 5 else { continue }
            let d = rows.last!.t - rows.first!.t
            if d > duration { duration = d }
        }
        guard duration > 0 else { return ([], [], 0) }

        let nBins = max(1, Int(ceil(duration / Self.binS)))
        var perSource: [[Double]] = []

        for source in MotionRuleClassifier.sources {
            guard let rows = bySource[source], rows.count >= 5 else { continue }
            let t0 = rows.first!.t
            let segment = rows.filter { $0.t - t0 >= 0 && $0.t - t0 < duration }
            guard segment.count >= 5 else { continue }

            let t    = segment.map { $0.t - t0 }
            let ax   = segment.map(\.ax), ay = segment.map(\.ay), az = segment.map(\.az)
            let gx   = segment.map(\.gx), gy = segment.map(\.gy), gz = segment.map(\.gz)
            let aM   = mag(ax, ay, az)
            let gM   = mag(gx, gy, gz)
            let aMed = median(aM)
            let aDyn = aM.map { abs($0 - aMed) }

            let score = zip(robustPosZ(gM), robustPosZ(aDyn)).map { g, a in
                min(0.62 * g + 0.38 * a, 14.0)
            }

            var binned = [Double](repeating: 0.0, count: nBins)
            for (tt, val) in zip(t, score) {
                let idx = min(nBins - 1, max(0, Int(tt / Self.binS)))
                binned[idx] = max(binned[idx], val)
            }
            perSource.append(smooth(binned))
        }

        guard !perSource.isEmpty else { return ([], [], duration) }

        // Fuse: 72% per-source max + 28% per-source mean
        var maxV = [Double](repeating: 0.0, count: nBins)
        var sumV = [Double](repeating: 0.0, count: nBins)
        for src in perSource {
            for i in 0..<nBins { maxV[i] = max(maxV[i], src[i]); sumV[i] += src[i] }
        }
        let n = Double(perSource.count)
        let env = smooth((0..<nBins).map { i in 0.72 * maxV[i] + 0.28 * sumV[i] / n })
        let times = (0..<nBins).map { (Double($0) + 0.5) * Self.binS }
        return (times, env, duration)
    }

    // MARK: - Peak Finder

    private func findAllPeaks(envelope: [Double]) -> [Int] {
        guard envelope.count >= 3 else { return [] }

        var candidates: [Int] = []
        for i in 1..<(envelope.count - 1) {
            if envelope[i] >= envelope[i-1] && envelope[i] >= envelope[i+1] && envelope[i] > 0 {
                candidates.append(i)
            }
        }
        guard !candidates.isEmpty else { return [] }

        // Amplitude floor (same formula as choose_five_peaks)
        let med    = median(envelope)
        let madVal = max(1e-6, median(envelope.map { abs($0 - med) }))
        let floor  = max(percentile62(envelope), med + 0.45 * madVal)
        var qualified = candidates.filter { envelope[$0] >= floor }
        if qualified.isEmpty { qualified = candidates }

        // Non-max suppression
        let minSepBins = max(1, Int((Self.minPeakSepS / Self.binS).rounded()))
        var selected: [Int] = []
        for idx in qualified.sorted(by: { envelope[$0] > envelope[$1] }) {
            if selected.allSatisfy({ abs(idx - $0) >= minSepBins }) {
                selected.append(idx)
            }
        }
        return selected.sorted()
    }

    // MARK: - Numeric Helpers

    private func mag(_ x: [Double], _ y: [Double], _ z: [Double]) -> [Double] {
        (0..<x.count).map { i -> Double in
            let xi = x[i], yi = y[i], zi = z[i]
            let sumSq = xi * xi + yi * yi + zi * zi
            return sumSq.squareRoot()
        }
    }

    private func robustPosZ(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        let med   = median(values)
        let mad   = max(1e-6, median(values.map { abs($0 - med) }))
        let scale = 1.4826 * mad
        return values.map { max(0.0, ($0 - med) / scale) }
    }

    private func smooth(_ v: [Double]) -> [Double] {
        let w = Self.smoothBins; guard v.count > 1 && w > 1 else { return v }
        let half = w / 2
        return (0..<v.count).map { i in
            let lo = max(0, i - half); let hi = min(v.count - 1, i + half)
            return v[lo...hi].reduce(0, +) / Double(hi - lo + 1)
        }
    }

    private func median(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted(); let mid = s.count / 2
        return s.count % 2 == 0 ? (s[mid-1] + s[mid]) / 2 : s[mid]
    }

    private func percentile62(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted()
        let rank = 0.62 * Double(s.count - 1)
        let lo = Int(rank); let hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (rank - Double(lo))
    }
}
