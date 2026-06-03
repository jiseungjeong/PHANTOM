import Foundation

// MARK: - Motion KNN Classifier
//
// On-device classifier backed by the active-window KNN model trained on
// JIHO, JUHO, and SHIM data (bundled as knn_params.json).
//
// Pipeline (mirrors Python sklearn pipeline exactly):
//   1. Group samples by source (Watch / iPhone / AirPods).
//   2. Detect the sync jump; features are extracted after jump_peak + 0.55 s.
//      classifySegment(): no jump cut — the caller already cut a tight window.
//   3. Build the 491-feature vector (same keys as rule_base_classifier.add_source_features).
//   4. Apply: SimpleImputer(0) → VarianceThreshold → RobustScaler → Normalizer(l2).
//   5. Compute 1-NN Manhattan distance against 450 stored training vectors; return
//      the label of the nearest neighbour and distance-based confidence.

struct MotionRuleClassifier {

    // MARK: - Public Result

    struct Result {
        let label: String
        let confidence: Double
        let sourceSignature: String
        let ruleSource: String   // "knn"
        let jumpCutS: Double
    }

    // MARK: - KNN Parameters  (decoded from knn_params.json)

    struct KNNParams: Decodable {
        let featureNames: [String]
        let varianceMask: [Bool]
        let scalerCenter: [Double]
        let scalerScale:  [Double]
        let trainVectors: [[Double]]
        let trainLabels:  [String]

        enum CodingKeys: String, CodingKey {
            case featureNames = "feature_names"
            case varianceMask = "variance_mask"
            case scalerCenter = "scaler_center"
            case scalerScale  = "scaler_scale"
            case trainVectors = "train_vectors"
            case trainLabels  = "train_labels"
        }
    }

    private let params: KNNParams

    // MARK: - Loading

    enum LoadError: Error { case resourceMissing, decodeFailed(Error) }

    static func load(bundle: Bundle = .main) throws -> MotionRuleClassifier {
        guard let url = bundle.url(forResource: "knn_params", withExtension: "json") else {
            throw LoadError.resourceMissing
        }
        let data = try Data(contentsOf: url)
        do {
            let p = try JSONDecoder().decode(KNNParams.self, from: data)
            return MotionRuleClassifier(params: p)
        } catch {
            throw LoadError.decodeFailed(error)
        }
    }

    private init(params: KNNParams) { self.params = params }

    // MARK: - Public Classification

    static let singleMotionLabels: Set<String> = [
        "Lead Jab", "Lead Hook", "Lead Uppercut",
        "Rear Jab", "Rear Hook", "Rear Uppercut",
        "Left Slip", "Right Slip", "Left Roll", "Right Roll",
    ]

    /// Classify a full recording: detect the sync jump, compute features from
    /// the post-jump segment, run KNN. Returns nil when data is too short.
    func classify(samples: [TimeSeriesRawSample]) -> Result? {
        let bySource = group(samples)
        let totalRows = bySource.values.reduce(0) { $0 + $1.count }
        guard totalRows >= 5 else { return nil }
        let jumpCutS = detectJumpCut(bySource)
        return classify(bySource: bySource, jumpCutS: jumpCutS)
    }

    /// Classify a tight per-event window (jumpCut = 0, matching the training
    /// pipeline for single-repetition active windows).
    func classifySegment(samples: [TimeSeriesRawSample]) -> Result? {
        let bySource = group(samples)
        let totalRows = bySource.values.reduce(0) { $0 + $1.count }
        guard totalRows >= 5 else { return nil }
        return classify(bySource: bySource, jumpCutS: 0.0)
    }

    // MARK: - Shared classification core

    private func classify(bySource: [String: [Row]], jumpCutS: Double) -> Result? {
        var features: [String: Double] = [
            "jump_cut_s": jumpCutS,
            "total_rows": Double(bySource.values.reduce(0) { $0 + $1.count }),
        ]
        for source in Self.sources {
            addSourceFeatures(into: &features, source: source, rows: bySource[source] ?? [], jumpCutS: jumpCutS)
        }
        let hasSegment = Self.sources.contains { (features["\(prefix($0))_n"] ?? 0) >= 5 }
        guard hasSegment else { return nil }

        let signature = sourceSignature(features)
        let (label, confidence) = predictKNN(features: features)
        return Result(label: label, confidence: confidence, sourceSignature: signature,
                      ruleSource: "knn", jumpCutS: jumpCutS)
    }

    // MARK: - KNN Inference

    private func predictKNN(features: [String: Double]) -> (label: String, confidence: Double) {
        // 1. Build raw vector — missing features default to 0.0 (SimpleImputer)
        var x = params.featureNames.map { features[$0] ?? 0.0 }

        // 2. VarianceThreshold — keep only selected columns
        x = zip(x, params.varianceMask).compactMap { $1 ? $0 : nil }

        // 3. RobustScaler: (x - center) / scale
        for i in 0..<x.count {
            x[i] = (x[i] - params.scalerCenter[i]) / params.scalerScale[i]
        }

        // 4. Normalizer l2
        let norm = x.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        if norm > 1e-9 { x = x.map { $0 / norm } }

        // 5. 1-NN Manhattan distance
        var bestDist = Double.infinity
        var bestLabel = ""
        for (vec, label) in zip(params.trainVectors, params.trainLabels) {
            var dist = 0.0
            for i in 0..<x.count { dist += abs(x[i] - vec[i]) }
            if dist < bestDist { bestDist = dist; bestLabel = label }
        }
        let confidence = bestDist < 1e-9 ? 1.0 : 1.0 / (1.0 + bestDist)
        return (bestLabel, confidence)
    }

    // MARK: - Source Grouping

    typealias Row = (t: Double, ax: Double, ay: Double, az: Double,
                     gx: Double, gy: Double, gz: Double)

    static let sources = ["Watch", "iPhone", "AirPods"]
    private static let jumpPriority = ["iPhone", "AirPods", "Watch"]

    private func prefix(_ source: String) -> String {
        String(source.lowercased().first ?? "?")
    }

    func group(_ samples: [TimeSeriesRawSample]) -> [String: [Row]] {
        var out: [String: [Row]] = [:]
        for s in samples {
            let row: Row = (s.timestamp, s.accX, s.accY, s.accZ, s.gyroX, s.gyroY, s.gyroZ)
            out[s.source, default: []].append(row)
        }
        for key in out.keys { out[key]?.sort { $0.t < $1.t } }
        return out
    }

    // MARK: - Jump Detection

    func detectJumpCut(_ data: [String: [Row]]) -> Double {
        var candidates: [(priority: Int, jumpT: Double)] = []

        for (priority, source) in Self.jumpPriority.enumerated() {
            guard let rows = data[source], rows.count >= 10 else { continue }
            let t0 = rows[0].t
            let duration = rows[rows.count - 1].t - t0
            if duration <= 0 { continue }

            let earlyLimit = min(5.0, max(1.5, duration * 0.50))
            var samples: [(tt: Double, acc: Double)] = []
            for row in rows {
                let tt = row.t - t0
                if tt > earlyLimit { break }
                let accMag = (row.ax*row.ax + row.ay*row.ay + row.az*row.az).squareRoot()
                samples.append((tt, accMag))
            }
            if samples.count < 6 { continue }

            let acc = samples.map(\.acc)
            let medianAcc = median(acc)
            let mad = max(0.05, median(acc.map { abs($0 - medianAcc) }))
            let threshold = max(2.15, medianAcc + 2.5 * mad)

            for i in 1..<(samples.count - 1) {
                if samples[i].acc >= threshold,
                   samples[i].acc >= samples[i-1].acc,
                   samples[i].acc >= samples[i+1].acc {
                    candidates.append((priority, samples[i].tt))
                    break
                }
            }
        }

        guard let best = candidates.min(by: { $0.priority != $1.priority
            ? $0.priority < $1.priority : $0.jumpT < $1.jumpT }) else { return 0.0 }
        return best.jumpT + 0.55
    }

    // MARK: - Feature Extraction

    private func addSourceFeatures(into features: inout [String: Double],
                                   source: String, rows: [Row], jumpCutS: Double) {
        let p = prefix(source)
        features["\(p)_present"] = rows.isEmpty ? 0.0 : 1.0
        features["\(p)_raw_n"]   = Double(rows.count)
        guard rows.count >= 5 else { return }

        let t0 = rows[0].t
        var segment = rows.filter { $0.t - t0 >= jumpCutS }
        if segment.count < 5 { segment = Array(rows.suffix(min(20, rows.count))) }

        let t  = segment.map { $0.t - t0 - jumpCutS }
        let ax = segment.map(\.ax), ay = segment.map(\.ay), az = segment.map(\.az)
        let gx = segment.map(\.gx), gy = segment.map(\.gy), gz = segment.map(\.gz)
        let accMag  = magnitude(ax, ay, az)
        let gyroMag = magnitude(gx, gy, gz)

        features["\(p)_n"]   = Double(segment.count)
        features["\(p)_dur"] = t.count > 1 ? ((t.max() ?? 0) - (t.min() ?? 0)) : 0.0

        let series: [(String, [Double])] = [
            ("ax", ax), ("ay", ay), ("az", az),
            ("gx", gx), ("gy", gy), ("gz", gz),
            ("am", accMag), ("gm", gyroMag),
        ]
        for (name, values) in series {
            features["\(p)_\(name)_mean"]   = mean(values)
            features["\(p)_\(name)_std"]    = std(values)
            features["\(p)_\(name)_min"]    = values.min() ?? 0
            features["\(p)_\(name)_max"]    = values.max() ?? 0
            features["\(p)_\(name)_absmax"] = values.map { abs($0) }.max() ?? 0
            features["\(p)_\(name)_energy"] = mean(values.map { $0 * $0 })
            for q in [5, 25, 50, 75, 95, 99] {
                features["\(p)_\(name)_p\(String(format: "%02d", q))"] = percentile(values, Double(q))
            }
        }

        let axisSeries: [(String, [Double])] = [
            ("ax", ax), ("ay", ay), ("az", az), ("gx", gx), ("gy", gy), ("gz", gz),
        ]
        for (axisName, values) in axisSeries {
            let pos = values.map { max($0, 0.0) }
            let neg = values.map { max(-$0, 0.0) }
            features["\(p)_\(axisName)_pos_p95"]    = percentile(pos, 95)
            features["\(p)_\(axisName)_neg_p95"]    = percentile(neg, 95)
            features["\(p)_\(axisName)_pos_energy"] = mean(pos.map { $0 * $0 })
            features["\(p)_\(axisName)_neg_energy"] = mean(neg.map { $0 * $0 })
        }

        let peakIndices: [(String, Int)] = [
            ("peakg", argmax(gyroMag)), ("peaka", argmax(accMag)),
        ]
        let peakSeries: [(String, [Double])] = [
            ("t", t), ("ax", ax), ("ay", ay), ("az", az),
            ("gx", gx), ("gy", gy), ("gz", gz), ("am", accMag), ("gm", gyroMag),
        ]
        for (peakName, idx) in peakIndices {
            for (name, values) in peakSeries {
                features["\(p)_\(peakName)_\(name)"] = values.isEmpty ? 0 : values[idx]
            }
        }

        let gyroMedian = median(gyroMag)
        let gyroMad    = max(0.05, median(gyroMag.map { abs($0 - gyroMedian) }))
        let accMedian  = median(accMag)
        let accMad     = max(0.05, median(accMag.map { abs($0 - accMedian) }))

        features["\(p)_gm_peak_count_adapt"] =
            Double(localPeaks(t: t, values: gyroMag, threshold: max(3.0, gyroMedian + 3.0 * gyroMad)).count)
        features["\(p)_am_peak_count_adapt"] =
            Double(localPeaks(t: t, values: accMag, threshold: max(2.2, accMedian + 3.0 * accMad)).count)

        for threshold in [2, 3, 5, 8, 12, 16, 20, 25, 30] {
            features["\(p)_gm_peak_count_gt\(threshold)"] =
                Double(localPeaks(t: t, values: gyroMag, threshold: Double(threshold)).count)
        }
        let accThresholds: [(Double, String)] = [
            (1.5, "1p5"), (2, "2"), (3, "3"), (5, "5"), (8, "8"), (12, "12"), (16, "16"), (20, "20"),
        ]
        for (thr, key) in accThresholds {
            features["\(p)_am_peak_count_gt\(key)"] =
                Double(localPeaks(t: t, values: accMag, threshold: thr).count)
        }
    }

    // MARK: - Local Peaks

    private func localPeaks(t: [Double], values: [Double], threshold: Double,
                            minSeparationS: Double = 0.28) -> [(Double, Double)] {
        var peaks: [(Double, Double)] = []
        guard values.count >= 3 else { return peaks }
        for i in 1..<(values.count - 1) {
            if values[i] < threshold { continue }
            if values[i] < values[i-1] || values[i] < values[i+1] { continue }
            if let last = peaks.last, t[i] - last.0 < minSeparationS {
                if values[i] > last.1 { peaks[peaks.count - 1] = (t[i], values[i]) }
            } else {
                peaks.append((t[i], values[i]))
            }
        }
        return peaks
    }

    // MARK: - Source Signature

    private func sourceSignature(_ f: [String: Double]) -> String {
        var present: [String] = []
        if (f["a_raw_n"] ?? 0) >= 5 { present.append("AirPods") }
        if (f["w_raw_n"] ?? 0) >= 5 { present.append("Watch") }
        if (f["i_raw_n"] ?? 0) >= 5 { present.append("iPhone") }
        return present.isEmpty ? "too_short" : present.sorted().joined(separator: "+")
    }

    // MARK: - Numeric Helpers

    private func magnitude(_ x: [Double], _ y: [Double], _ z: [Double]) -> [Double] {
        (0..<x.count).map { i in (x[i]*x[i] + y[i]*y[i] + z[i]*z[i]).squareRoot() }
    }

    private func mean(_ x: [Double]) -> Double {
        x.isEmpty ? 0 : x.reduce(0, +) / Double(x.count)
    }

    private func std(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let m = mean(x)
        return (x.reduce(0) { $0 + ($1-m)*($1-m) } / Double(x.count)).squareRoot()
    }

    func median(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted(); let mid = s.count / 2
        return s.count % 2 == 0 ? (s[mid-1] + s[mid]) / 2 : s[mid]
    }

    private func percentile(_ x: [Double], _ q: Double) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted()
        if s.count == 1 { return s[0] }
        let rank  = q / 100.0 * Double(s.count - 1)
        let lower = Int(rank); let upper = min(lower + 1, s.count - 1)
        return s[lower] + (s[upper] - s[lower]) * (rank - Double(lower))
    }

    private func argmax(_ x: [Double]) -> Int {
        guard !x.isEmpty else { return 0 }
        var best = 0
        for i in 1..<x.count where x[i] > x[best] { best = i }
        return best
    }
}
