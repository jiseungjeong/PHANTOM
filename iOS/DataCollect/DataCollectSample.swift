import Foundation

// MARK: - Labeled IMU Sample

struct DataCollectSample: Identifiable, Codable {
    let id:     UUID
    let label:  String    // gesture label (e.g. "Lead Jab", "Left Slip")
    let source: String    // "iPhone" | "Watch"
    let window: [Double]  // raw magnitude values (g), 100 Hz
    let tsMs:   Double    // session-relative start timestamp (ms)

    init(label: String, source: String, window: [Double], tsMs: Double) {
        self.id     = UUID()
        self.label  = label
        self.source = source
        self.window = window
        self.tsMs   = tsMs
    }

    // MARK: - Extracted Features

    var peak: Double {
        window.max() ?? 0
    }

    var mean: Double {
        guard !window.isEmpty else { return 0 }
        return window.reduce(0, +) / Double(window.count)
    }

    /// Root mean square — overall signal power
    var rms: Double {
        guard !window.isEmpty else { return 0 }
        return sqrt(window.map { $0 * $0 }.reduce(0, +) / Double(window.count))
    }

    /// Total signal energy
    var energy: Double {
        window.map { $0 * $0 }.reduce(0, +)
    }

    /// Approximate duration in ms at 100 Hz (10 ms/sample)
    var durationMs: Double {
        Double(window.count) * 10.0
    }

    /// Zero-crossing rate relative to window mean
    var zeroCrossingRate: Double {
        guard window.count > 1 else { return 0 }
        let m = mean
        var crossings = 0
        for i in 1..<window.count {
            if (window[i - 1] - m) * (window[i] - m) < 0 { crossings += 1 }
        }
        return Double(crossings) / Double(window.count - 1)
    }

    /// Number of samples above 50% of peak (impulse spread)
    var impulseWidth: Int {
        window.filter { $0 > peak * 0.5 }.count
    }

    /// Variance of the signal
    var variance: Double {
        guard window.count > 1 else { return 0 }
        let m = mean
        return window.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(window.count - 1)
    }
}

// MARK: - CSV Export

extension [DataCollectSample] {
    func csvString() -> String {
        let header = [
            "id", "label", "source", "tsMs",
            "peak", "mean", "rms", "energy",
            "durationMs", "zeroCrossingRate", "impulseWidth", "variance", "sampleCount"
        ].joined(separator: ",")

        let rows = self.map { s in
            [
                s.id.uuidString,
                "\"\(s.label)\"",
                s.source,
                String(format: "%.1f", s.tsMs),
                String(format: "%.5f", s.peak),
                String(format: "%.5f", s.mean),
                String(format: "%.5f", s.rms),
                String(format: "%.5f", s.energy),
                String(format: "%.1f",  s.durationMs),
                String(format: "%.5f", s.zeroCrossingRate),
                "\(s.impulseWidth)",
                String(format: "%.5f", s.variance),
                "\(s.window.count)"
            ].joined(separator: ",")
        }

        return ([header] + rows).joined(separator: "\n")
    }
}
