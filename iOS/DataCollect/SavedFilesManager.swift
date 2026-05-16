import Foundation

// MARK: - Saved Files Manager
//
// Scans the app's Documents directory for exported .csv and .json files,
// and provides metadata (size, date) and deletion.

@MainActor
final class SavedFilesManager: ObservableObject {

    @Published private(set) var files: [SavedFile] = []

    /// Multi-select state used by FilesView's Edit mode. Allows the view to
    /// drive selection through SwiftUI's `List(selection:)` binding.
    @Published var selectedFileIDs: Set<UUID> = []

    // MARK: - SavedFile

    struct SavedFile: Identifiable {
        let id  = UUID()
        let url: URL

        var name: String { url.lastPathComponent }
        var ext:  String { url.pathExtension.lowercased() }

        var sizeBytes: Int64 {
            let v = try? url.resourceValues(forKeys: [.fileSizeKey])
            return Int64(v?.fileSize ?? 0)
        }

        var modDate: Date {
            let v = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            return v?.contentModificationDate ?? .distantPast
        }
    }

    // MARK: - Selection helpers

    var selectedFiles: [SavedFile] {
        files.filter { selectedFileIDs.contains($0.id) }
    }

    var allFileURLs: [URL] { files.map(\.url) }

    func toggleSelection(_ file: SavedFile) {
        if selectedFileIDs.contains(file.id) {
            selectedFileIDs.remove(file.id)
        } else {
            selectedFileIDs.insert(file.id)
        }
    }

    func clearSelection() {
        selectedFileIDs.removeAll()
    }

    // MARK: - Public

    func refresh() {
        let dir  = FileUtils.documentsURL
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]

        let urls = (try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: keys,
            options: .skipsHiddenFiles
        )) ?? []

        files = urls
            .filter { ["csv", "json"].contains($0.pathExtension.lowercased()) }
            .sorted { a, b in
                let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return aDate > bDate
            }
            .map { SavedFile(url: $0) }
    }

    func delete(_ file: SavedFile) {
        try? FileManager.default.removeItem(at: file.url)
        files.removeAll { $0.id == file.id }
        selectedFileIDs.remove(file.id)
    }

    // MARK: - Helpers

    func formattedSize(_ bytes: Int64) -> String {
        let kb = Double(bytes) / 1024
        if kb < 1024 { return String(format: "%.1f KB", max(kb, 0.1)) }
        return String(format: "%.2f MB", kb / 1024)
    }

    func formattedDate(_ date: Date) -> String {
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: date)
    }
}
