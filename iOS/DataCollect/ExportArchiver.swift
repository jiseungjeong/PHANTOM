import Foundation

// MARK: - Export Archiver
//
// Builds a single .zip archive from a set of files in the app's Documents
// folder using only Foundation APIs (NSFileCoordinator with .forUploading)
// — no external SPM dependency.
//
// Strategy:
//   1. Copy/hard-link selected files into a temporary staging directory.
//   2. NSFileCoordinator with .forUploading zips that directory automatically.
//   3. Move the resulting zip into a stable tmp location and return its URL.
//
// The caller is responsible for sharing the URL (via ShareLink /
// UIActivityViewController) and deciding when to delete it. The OS will
// eventually purge the system tmp directory.

enum ExportArchiver {

    enum ArchiveError: Error, LocalizedError {
        case empty
        case stagingFailed(Error)
        case zipFailed(Error?)

        var errorDescription: String? {
            switch self {
            case .empty:                  return "No files selected."
            case .stagingFailed(let e):   return "Failed to stage files: \(e.localizedDescription)"
            case .zipFailed(let e):       return "Failed to create archive: \(e?.localizedDescription ?? "unknown error")"
            }
        }
    }

    /// Build a single zip archive containing all of `urls`. Throws on empty
    /// input, staging failure, or zip-coordinator failure.
    ///
    /// The returned URL points at a file in `FileManager.default.temporaryDirectory`.
    static func makeArchive(urls: [URL]) throws -> URL {
        guard !urls.isEmpty else { throw ArchiveError.empty }

        let stagingName = "phantom_export_\(FileUtils.dateStamp())"
        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent(stagingName, isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: staging,
                                                    withIntermediateDirectories: true)
            for src in urls {
                let dst = staging.appendingPathComponent(src.lastPathComponent)
                // Same-sandbox hard-link is cheap. Falls back to copy if the
                // filesystem rejects the link (rare on iOS, but defensive).
                do {
                    try FileManager.default.linkItem(at: src, to: dst)
                } catch {
                    try FileManager.default.copyItem(at: src, to: dst)
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw ArchiveError.stagingFailed(error)
        }

        let archive: URL
        do {
            archive = try zipDirectory(staging, archiveName: stagingName)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }

        try? FileManager.default.removeItem(at: staging)
        return archive
    }

    // MARK: - Private

    /// Zip an entire directory using NSFileCoordinator's `.forUploading`
    /// option, which returns a temporary zip representation of the directory.
    /// We copy that zip out to our own tmp location because the coordinator
    /// closure's URL may be invalidated as soon as it returns.
    private static func zipDirectory(_ directory: URL, archiveName: String) throws -> URL {
        let coordinator = NSFileCoordinator()
        var coordError: NSError?
        var resultURL: URL?
        var copyError: Error?

        coordinator.coordinate(readingItemAt: directory,
                               options: [.forUploading],
                               error: &coordError) { tempZipURL in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(archiveName).zip")
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: tempZipURL, to: dest)
                resultURL = dest
            } catch {
                copyError = error
            }
        }

        if let coordError { throw ArchiveError.zipFailed(coordError) }
        if let copyError  { throw ArchiveError.zipFailed(copyError) }
        guard let resultURL else { throw ArchiveError.zipFailed(nil) }
        return resultURL
    }
}
