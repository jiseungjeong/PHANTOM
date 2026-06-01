import Foundation

// MARK: - Motion Rule Classifier  (single-action if/then decision tree)
//
// On-device classifier backed by the 10-label single-action decision tree from
// analysis/motion_rule_base 2/.../outlier_filtered_rule_summary.json
// (rule_tree_metrics.tree), bundled as single_action_tree.json. Given a window
// of raw IMU samples it predicts one primitive motion:
//   Lead/Rear Jab·Hook·Uppercut, Left/Right Slip, Left/Right Roll.
// (No combinations, no Step, no Guard — those were dropped with the model swap.)
//
// Pipeline:
//   1. Group samples by source (Watch / iPhone / AirPods).
//   2. classify(): detect the sync jump and segment from jump_peak + 0.55 s.
//      classifySegment(): no jump cut — the caller already cut a tight window.
//   3. Extract the same feature dictionary as rule_base_classifier.add_source_features
//      (feature names like w_ax_p95, i_peaka_gz). The tree references a subset.
//   4. Walk the tree: at each split, go left if feature <= threshold, else right;
//      the leaf carries the predicted label and its purity (confidence).
//
// The tree was trained on single-repetition windows with add_source_features(...,
// jumpCut = 0). classifySegment mirrors that exactly (jumpCut = 0 on a tight
// window), so it is the parity-correct entry point for live segmentation.
//
// IMPORTANT numeric-parity notes (the easiest things to get wrong):
//   • percentile() reproduces numpy's default "linear" interpolation.
//   • std() is the population standard deviation (ddof=0), matching np.std.
//   • median() of an even count averages the two middle elements.

struct MotionRuleClassifier {

    // MARK: - Public Result

    struct Result {
        let label: String
        let confidence: Double
        let sourceSignature: String
        let ruleSource: String   // "rule_tree"
        let jumpCutS: Double
    }

    // MARK: - Decoded single_action_tree.json

    private struct RuleTree: Decodable {
        let labels: [String]
        let tree: Node
    }

    /// A node is either a split (has feature/threshold/left/right) or a leaf
    /// (prediction/confidence only). The JSON tags this with `type`.
    private final class Node: Decodable {
        let type: String           // "split" | "leaf"
        let prediction: String
        let confidence: Double
        let feature: String?
        let threshold: Double?
        let left: Node?
        let right: Node?

        var isLeaf: Bool { type == "leaf" || feature == nil }
    }

    private let ruleTree: RuleTree

    // MARK: - Loading

    enum LoadError: Error { case resourceMissing, decodeFailed(Error) }

    /// Load and decode `single_action_tree.json` from the app bundle once;
    /// reuse the returned instance for every classification.
    static func load(bundle: Bundle = .main) throws -> MotionRuleClassifier {
        guard let url = bundle.url(forResource: "single_action_tree", withExtension: "json") else {
            throw LoadError.resourceMissing
        }
        let data = try Data(contentsOf: url)
        do {
            let tree = try JSONDecoder().decode(RuleTree.self, from: data)
            return MotionRuleClassifier(ruleTree: tree)
        } catch {
            throw LoadError.decodeFailed(error)
        }
    }

    private init(ruleTree: RuleTree) { self.ruleTree = ruleTree }

    // MARK: - Public Classification

    /// The 10 single-motion labels the tree can emit. No combinations, no Step,
    /// no Guard.
    static let singleMotionLabels: Set<String> = [
        "Lead Jab", "Lead Hook", "Lead Uppercut",
        "Rear Jab", "Rear Hook", "Rear Uppercut",
        "Left Slip", "Right Slip", "Left Roll", "Right Roll",
    ]

    /// Classify a full recording: detect the sync jump, segment from
    /// jump_peak + 0.55 s, then walk the tree. Returns nil when there is not
    /// enough usable data (fewer than 5 rows, or no source segment >= 5 samples).
    func classify(samples: [TimeSeriesRawSample]) -> Result? {
        let bySource = group(samples)
        let totalRows = bySource.values.reduce(0) { $0 + $1.count }
        guard totalRows >= 5 else { return nil }
        let jumpCutS = detectJumpCut(bySource)
        return classify(bySource: bySource, jumpCutS: jumpCutS)
    }

    /// Classify a short per-event window cut from a live session. The caller
    /// already isolated a tight window, so there is no leading sync clap —
    /// features are computed with jumpCut = 0, mirroring how the tree was
    /// trained on single-repetition windows. Returns nil when too few samples.
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

        // Require at least one source segment with >= 5 samples.
        let hasSegment = Self.sources.contains { (features["\(prefix($0))_n"] ?? 0) >= 5 }
        guard hasSegment else { return nil }

        let signature = sourceSignature(features)
        let (label, confidence) = walk(ruleTree.tree, features: features)
        return Result(label: label, confidence: confidence, sourceSignature: signature,
                      ruleSource: "rule_tree", jumpCutS: jumpCutS)
    }

    /// Walk the decision tree: go left when feature <= threshold, else right.
    /// Missing features default to 0.0 (matches features.get(name, 0.0)).
    private func walk(_ node: Node, features: [String: Double]) -> (label: String, confidence: Double) {
        var current = node
        while !current.isLeaf, let feature = current.feature, let threshold = current.threshold,
              let left = current.left, let right = current.right {
            let value = features[feature] ?? 0.0
            current = value <= threshold ? left : right
        }
        return (current.prediction, current.confidence)
    }

    // MARK: - Source Grouping

    /// One row: (t, accX, accY, accZ, gyroX, gyroY, gyroZ), sorted by t.
    private typealias Row = (t: Double, ax: Double, ay: Double, az: Double,
                             gx: Double, gy: Double, gz: Double)

    private static let sources = ["Watch", "iPhone", "AirPods"]
    private static let jumpPriority = ["iPhone", "AirPods", "Watch"]

    private func prefix(_ source: String) -> String {
        String(source.lowercased().first ?? "?")  // Watch->w, iPhone->i, AirPods->a
    }

    private func group(_ samples: [TimeSeriesRawSample]) -> [String: [Row]] {
        var out: [String: [Row]] = [:]
        for s in samples {
            let row: Row = (s.timestamp, s.accX, s.accY, s.accZ, s.gyroX, s.gyroY, s.gyroZ)
            out[s.source, default: []].append(row)
        }
        for key in out.keys { out[key]?.sort { $0.t < $1.t } }
        return out
    }

    // MARK: - Jump Detection  (detect_jump_cut)

    private func detectJumpCut(_ data: [String: [Row]]) -> Double {
        // candidate: (priority, jumpTime)
        var candidates: [(priority: Int, jumpT: Double)] = []

        for (priority, source) in Self.jumpPriority.enumerated() {
            guard let rows = data[source], rows.count >= 10 else { continue }

            let t0 = rows[0].t
            let duration = rows[rows.count - 1].t - t0
            if duration <= 0 { continue }

            let earlyLimit = min(5.0, max(1.5, duration * 0.50))
            // samples: (tt, accMag, gyroMag)
            var samples: [(tt: Double, acc: Double, gyro: Double)] = []
            for row in rows {
                let tt = row.t - t0
                if tt > earlyLimit { break }
                let accMag = (row.ax * row.ax + row.ay * row.ay + row.az * row.az).squareRoot()
                let gyroMag = (row.gx * row.gx + row.gy * row.gy + row.gz * row.gz).squareRoot()
                samples.append((tt, accMag, gyroMag))
            }
            if samples.count < 6 { continue }

            let acc = samples.map(\.acc)
            let medianAcc = median(acc)
            let madRaw = median(acc.map { abs($0 - medianAcc) })
            let mad = madRaw == 0 ? 0.05 : madRaw
            let threshold = max(2.15, medianAcc + 2.5 * mad)

            for i in 1..<(samples.count - 1) {
                if samples[i].acc >= threshold,
                   samples[i].acc >= samples[i - 1].acc,
                   samples[i].acc >= samples[i + 1].acc {
                    candidates.append((priority, samples[i].tt))
                    break
                }
            }
        }

        guard let best = candidates.min(by: { lhs, rhs in
            lhs.priority != rhs.priority ? lhs.priority < rhs.priority : lhs.jumpT < rhs.jumpT
        }) else {
            return 0.0
        }
        return best.jumpT + 0.55
    }

    // MARK: - Feature Extraction  (add_source_features)

    private func addSourceFeatures(into features: inout [String: Double],
                                   source: String, rows: [Row], jumpCutS: Double) {
        let p = prefix(source)
        features["\(p)_present"] = rows.isEmpty ? 0.0 : 1.0
        features["\(p)_raw_n"] = Double(rows.count)

        guard rows.count >= 5 else { return }

        let t0 = rows[0].t
        var segment = rows.filter { $0.t - t0 >= jumpCutS }
        if segment.count < 5 {
            segment = Array(rows.suffix(min(20, rows.count)))
        }

        // Build per-channel arrays. The relative time uses t - t0 - jumpCutS,
        // matching the Python script.
        let t = segment.map { $0.t - t0 - jumpCutS }
        let ax = segment.map(\.ax), ay = segment.map(\.ay), az = segment.map(\.az)
        let gx = segment.map(\.gx), gy = segment.map(\.gy), gz = segment.map(\.gz)
        let accMag = magnitude(ax, ay, az)
        let gyroMag = magnitude(gx, gy, gz)

        features["\(p)_n"] = Double(segment.count)
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
            let positive = values.map { max($0, 0.0) }
            let negative = values.map { max(-$0, 0.0) }
            features["\(p)_\(axisName)_pos_p95"]    = percentile(positive, 95)
            features["\(p)_\(axisName)_neg_p95"]    = percentile(negative, 95)
            features["\(p)_\(axisName)_pos_energy"] = mean(positive.map { $0 * $0 })
            features["\(p)_\(axisName)_neg_energy"] = mean(negative.map { $0 * $0 })
        }

        // Peak-aligned features at the global gyro/acc magnitude maxima.
        let peakIndices: [(String, Int)] = [
            ("peakg", argmax(gyroMag)),
            ("peaka", argmax(accMag)),
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

        // Adaptive peak counts.
        let gyroMedian = median(gyroMag)
        let gyroMadRaw = median(gyroMag.map { abs($0 - gyroMedian) })
        let gyroMad = gyroMadRaw == 0 ? 0.05 : gyroMadRaw
        let accMedian = median(accMag)
        let accMadRaw = median(accMag.map { abs($0 - accMedian) })
        let accMad = accMadRaw == 0 ? 0.05 : accMadRaw
        features["\(p)_gm_peak_count_adapt"] =
            Double(localPeaks(t: t, values: gyroMag, threshold: max(3.0, gyroMedian + 3.0 * gyroMad)).count)
        features["\(p)_am_peak_count_adapt"] =
            Double(localPeaks(t: t, values: accMag, threshold: max(2.2, accMedian + 3.0 * accMad)).count)

        for threshold in [2, 3, 5, 8, 12, 16, 20, 25, 30] {
            features["\(p)_gm_peak_count_gt\(threshold)"] =
                Double(localPeaks(t: t, values: gyroMag, threshold: Double(threshold)).count)
        }
        // Keys mirror the Python feature names exactly. There 1.5 is a float so
        // str(1.5).replace(".", "p") == "1p5", while 2, 3, ... are ints so their
        // keys are "gt2", "gt3", ... (no "p0" suffix).
        let accThresholds: [(threshold: Double, key: String)] = [
            (1.5, "1p5"), (2, "2"), (3, "3"), (5, "5"),
            (8, "8"), (12, "12"), (16, "16"), (20, "20"),
        ]
        for (threshold, key) in accThresholds {
            features["\(p)_am_peak_count_gt\(key)"] =
                Double(localPeaks(t: t, values: accMag, threshold: threshold).count)
        }
    }

    // MARK: - local_peaks

    private func localPeaks(t: [Double], values: [Double], threshold: Double,
                            minSeparationS: Double = 0.28) -> [(Double, Double)] {
        var peaks: [(Double, Double)] = []
        guard values.count >= 3 else { return peaks }

        for i in 1..<(values.count - 1) {
            if values[i] < threshold { continue }
            if values[i] < values[i - 1] || values[i] < values[i + 1] { continue }

            if let last = peaks.last, t[i] - last.0 < minSeparationS {
                if values[i] > last.1 {
                    peaks[peaks.count - 1] = (t[i], values[i])
                }
            } else {
                peaks.append((t[i], values[i]))
            }
        }
        return peaks
    }

    // MARK: - Source Signature  (source_signature_from_features)

    private func sourceSignature(_ f: [String: Double]) -> String {
        var present: [String] = []
        if (f["a_raw_n"] ?? 0) >= 5 { present.append("AirPods") }
        if (f["w_raw_n"] ?? 0) >= 5 { present.append("Watch") }
        if (f["i_raw_n"] ?? 0) >= 5 { present.append("iPhone") }
        return present.isEmpty ? "too_short" : present.sorted().joined(separator: "+")
    }

    // MARK: - Numeric Helpers  (numpy parity)

    /// Per-sample Euclidean magnitude of three equal-length channels.
    private func magnitude(_ x: [Double], _ y: [Double], _ z: [Double]) -> [Double] {
        let n = x.count
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).squareRoot()
        }
        return out
    }

    private func mean(_ x: [Double]) -> Double {
        x.isEmpty ? 0 : x.reduce(0, +) / Double(x.count)
    }

    /// Population standard deviation (ddof = 0), matching numpy's np.std.
    private func std(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let m = mean(x)
        let variance = x.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(x.count)
        return variance.squareRoot()
    }

    private func median(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let sorted = x.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2.0 : sorted[mid]
    }

    /// numpy np.percentile default ("linear" interpolation). q is in [0, 100].
    private func percentile(_ x: [Double], _ q: Double) -> Double {
        guard !x.isEmpty else { return 0 }
        let sorted = x.sorted()
        if sorted.count == 1 { return sorted[0] }
        let rank = q / 100.0 * Double(sorted.count - 1)
        let lower = Int(rank.rounded(.down))
        let upper = Int(rank.rounded(.up))
        if lower == upper { return sorted[lower] }
        let frac = rank - Double(lower)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * frac
    }

    /// Index of the maximum value (first occurrence), matching np.argmax.
    private func argmax(_ x: [Double]) -> Int {
        guard !x.isEmpty else { return 0 }
        var best = 0
        for i in 1..<x.count where x[i] > x[best] { best = i }
        return best
    }
}
