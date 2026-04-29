import Foundation

// MARK: - File / Timestamp Helpers
//
// Shared utilities for code that writes CSV/JSON into the app's Documents
// folder and timestamps filenames.

enum FileUtils {

    /// The app's Documents directory on the current platform.
    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// `yyyyMMdd_HHmmss` timestamp suitable for filenames.
    static func dateStamp(_ date: Date = Date()) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: date)
    }

    static func normalizedParticipantId(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let token = safeFileToken(trimmed.isEmpty ? "P00" : trimmed)
        return token.isEmpty ? "P00" : token.uppercased()
    }

    static func safeFileToken(_ value: String) -> String {
        let folded = value
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
            .lowercased()
        var scalars: [Character] = []
        scalars.reserveCapacity(folded.count)

        for ch in folded {
            if ch.isLetter || ch.isNumber {
                scalars.append(ch)
            } else {
                scalars.append("_")
            }
        }

        let collapsed = String(scalars)
            .split(separator: "_", omittingEmptySubsequences: true)
            .joined(separator: "_")
        return collapsed.isEmpty ? "unknown" : collapsed
    }

    static func prefixedRawFilename(participantId: String,
                                    label: String,
                                    source: String? = nil,
                                    ext: String = "csv",
                                    timestamp: Date = Date()) -> String {
        let participant = normalizedParticipantId(participantId)
        let sourcePart = source.map { "_\($0)" } ?? ""
        return "\(participant)_\(label)\(sourcePart)_\(dateStamp(timestamp)).\(ext)"
    }
}
