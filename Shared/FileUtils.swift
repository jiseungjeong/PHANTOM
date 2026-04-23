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
}
