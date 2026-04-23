import Foundation

// MARK: - Parsed CSV
//
// Lightweight parser used by FilesView to read exported DataCollect CSVs
// and derive per-label summaries. Handles quoted fields.

struct ParsedCSV {
    let headers: [String]
    let rows:    [[String]]

    init?(url: URL) {
        guard let raw = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var all: [[String]] = []
        for line in raw.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            all.append(ParsedCSV.parseLine(trimmed))
        }
        guard all.count > 1 else { return nil }
        self.headers = all[0]
        self.rows    = Array(all.dropFirst())
    }

    var rowCount: Int { rows.count }

    // MARK: - Label breakdown

    private var labelIdx:  Int? { headers.firstIndex(of: "label") }
    private var peakIdx:   Int? { headers.firstIndex(of: "peak") }
    private var rmsIdx:    Int? { headers.firstIndex(of: "rms") }

    var labelCounts: [(label: String, count: Int)] {
        guard let li = labelIdx else { return [] }
        var counts: [String: Int] = [:]
        for row in rows {
            let l = li < row.count ? row[li] : ""
            counts[l, default: 0] += 1
        }
        return counts.sorted { $0.value > $1.value }.map { (label: $0.key, count: $0.value) }
    }

    var avgPeakPerLabel: [String: Double] { averagePerLabel(valueIdx: peakIdx) }
    var avgRMSPerLabel:  [String: Double] { averagePerLabel(valueIdx: rmsIdx) }

    private func averagePerLabel(valueIdx: Int?) -> [String: Double] {
        guard let li = labelIdx, let vi = valueIdx else { return [:] }
        var groups: [String: [Double]] = [:]
        for row in rows {
            let l = li < row.count ? row[li] : ""
            let v = vi < row.count ? (Double(row[vi]) ?? 0) : 0
            groups[l, default: []].append(v)
        }
        return groups.mapValues { $0.reduce(0, +) / Double($0.count) }
    }

    var globalAvgPeak: Double? { globalAverage(valueIdx: peakIdx) }
    var globalAvgRMS:  Double? { globalAverage(valueIdx: rmsIdx) }

    private func globalAverage(valueIdx: Int?) -> Double? {
        guard let vi = valueIdx else { return nil }
        let vals = rows.compactMap { vi < $0.count ? Double($0[vi]) : nil }
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }

    // MARK: - CSV line parser (handles quoted fields)

    static func parseLine(_ line: String) -> [String] {
        var cells: [String] = []
        var current = ""
        var inQuotes = false
        for char in line {
            if char == "\"" {
                inQuotes.toggle()
            } else if char == "," && !inQuotes {
                cells.append(current)
                current = ""
            } else {
                current.append(char)
            }
        }
        cells.append(current)
        return cells
    }
}
