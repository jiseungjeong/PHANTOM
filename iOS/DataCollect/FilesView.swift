import SwiftUI

// MARK: - Files View
//
// Browses all exported CSV / JSON files saved to the app's Documents folder.
// Tap a file to inspect its contents; share individual files via the row's
// quick-share button or bundle multiple files into a zip via Edit-mode
// multi-select + "Share Selected", or zip the entire folder via "Share All".

struct FilesView: View {
    @StateObject private var manager = SavedFilesManager()

    // MARK: - Local UI state

    /// Archive presented in the bottom card after a successful zip build.
    /// Tapping Share opens UIActivityViewController; tapping the X discards.
    @State private var preparedArchive: PreparedArchive?
    @State private var isArchiving = false
    @State private var showShareSheet = false
    @State private var errorMessage: String?

    struct PreparedArchive: Identifiable {
        let id = UUID()
        let url: URL
        let fileCount: Int
        let totalBytes: Int64
    }

    var body: some View {
        NavigationStack {
            FilesContent(
                manager: manager,
                preparedArchive: $preparedArchive,
                isArchiving: $isArchiving,
                showShareSheet: $showShareSheet,
                errorMessage: $errorMessage,
                shareAll: shareAll,
                shareSelected: shareSelected,
                deleteSelected: deleteSelected,
                discardPreparedArchive: discardPreparedArchive,
                dismissPreparedArchive: dismissPreparedArchive
            )
            .navigationTitle("Saved Files")
            .navigationBarTitleDisplayMode(.large)
        }
        // onAppear must be on NavigationStack (outside it), not on a child view inside,
        // so it fires every time this tab is selected.
        .onAppear { manager.refresh() }
    }

    // MARK: - Actions

    private func shareAll() {
        makeArchiveAndShow(urls: manager.allFileURLs)
    }

    private func shareSelected() {
        makeArchiveAndShow(urls: manager.selectedFiles.map(\.url))
    }

    private func deleteSelected() {
        for file in manager.selectedFiles {
            manager.delete(file)
        }
        manager.clearSelection()
    }

    private func makeArchiveAndShow(urls: [URL]) {
        guard !urls.isEmpty else { return }
        isArchiving = true
        Task.detached(priority: .userInitiated) {
            do {
                let archiveURL = try ExportArchiver.makeArchive(urls: urls)
                let attrs = (try? FileManager.default
                    .attributesOfItem(atPath: archiveURL.path)) ?? [:]
                let size = (attrs[.size] as? Int64) ?? 0
                await MainActor.run {
                    self.preparedArchive = PreparedArchive(
                        url: archiveURL,
                        fileCount: urls.count,
                        totalBytes: size
                    )
                    self.isArchiving = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isArchiving = false
                }
            }
        }
    }

    private func discardPreparedArchive() {
        if let url = preparedArchive?.url {
            try? FileManager.default.removeItem(at: url)
        }
        preparedArchive = nil
    }

    private func dismissPreparedArchive() {
        // After the share sheet closes, drop the archive reference. The tmp
        // zip will be cleaned up by the OS eventually; do not delete it here
        // because Mail/Files/etc. may still be uploading it asynchronously.
        preparedArchive = nil
    }
}

// MARK: - Files Content (reads SwiftUI's environment editMode)

private struct FilesContent: View {
    @ObservedObject var manager: SavedFilesManager
    @Binding var preparedArchive: FilesView.PreparedArchive?
    @Binding var isArchiving: Bool
    @Binding var showShareSheet: Bool
    @Binding var errorMessage: String?
    let shareAll: () -> Void
    let shareSelected: () -> Void
    let deleteSelected: () -> Void
    let discardPreparedArchive: () -> Void
    let dismissPreparedArchive: () -> Void

    @Environment(\.editMode) private var editMode

    private var isEditing: Bool { editMode?.wrappedValue.isEditing ?? false }

    var body: some View {
        ZStack(alignment: .bottom) {
            // Main content
            Group {
                if manager.files.isEmpty {
                    EmptyFilesState()
                } else {
                    List(selection: $manager.selectedFileIDs) {
                        ForEach(manager.files) { file in
                            // In edit mode SwiftUI lays out a selection circle
                            // on each row only when the row label is NOT a
                            // NavigationLink. Switch presentations so the user
                            // can either tap-to-navigate (normal) or
                            // tap-to-select (edit).
                            if isEditing {
                                FileRow(file: file, manager: manager)
                                    .tag(file.id)
                            } else {
                                NavigationLink {
                                    FileDetailView(file: file)
                                } label: {
                                    FileRow(file: file, manager: manager)
                                }
                                .tag(file.id)
                            }
                        }
                        .onDelete { offsets in
                            offsets.map { manager.files[$0] }.forEach { manager.delete($0) }
                        }
                    }
                    .listStyle(.insetGrouped)
                    // Leave room at the bottom so the floating bar doesn't
                    // cover the last row.
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        if showsBottomBar {
                            Color.clear.frame(height: 70)
                        }
                    }
                }
            }

            // Floating bottom bar (only when needed)
            if showsBottomBar {
                bottomBarContent
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isEditing)
        .animation(.easeInOut(duration: 0.2), value: manager.selectedFileIDs)
        .animation(.easeInOut(duration: 0.2), value: isArchiving)
        .animation(.easeInOut(duration: 0.2), value: preparedArchive?.id)
        .toolbar { toolbarContent }
        .sheet(isPresented: $showShareSheet, onDismiss: dismissPreparedArchive) {
            if let archive = preparedArchive {
                ShareSheet(url: archive.url)
            }
        }
        .alert("Export Failed",
               isPresented: alertBinding,
               actions: { Button("OK") { errorMessage = nil } },
               message: { Text(errorMessage ?? "") })
    }

    /// True when any of the three bottom-bar states should render.
    /// Note: we do NOT gate on `isEditing` because iOS 17's
    /// `List(selection:)` with `.insetGrouped` style enables selection
    /// without flipping the SwiftUI environment editMode. Selection set
    /// being non-empty is a sufficient signal that the user is selecting.
    private var showsBottomBar: Bool {
        if isArchiving { return true }
        if preparedArchive != nil { return true }
        if !manager.selectedFileIDs.isEmpty { return true }
        return false
    }

    @ViewBuilder
    private var bottomBarContent: some View {
        if isArchiving {
            archivingBar
        } else if let archive = preparedArchive {
            archiveReadyCard(archive)
        } else if !manager.selectedFileIDs.isEmpty {
            selectionActionBar
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        if !manager.files.isEmpty {
            ToolbarItem(placement: .navigationBarLeading) {
                EditButton()
            }
            if !isEditing {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { shareAll() } label: {
                        Label("Share All", systemImage: "square.and.arrow.up.on.square")
                    }
                    .disabled(isArchiving)
                }
            }
        }
    }

    // MARK: - Bottom bar pieces (composed by bottomBarContent)

    private var archivingBar: some View {
        HStack(spacing: 10) {
            ProgressView()
            Text("Preparing archive…")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding()
        .background(.bar)
    }

    @ViewBuilder
    private func archiveReadyCard(_ archive: FilesView.PreparedArchive) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "doc.zipper")
                .font(.title2)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text("Archive ready")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Text("\(archive.fileCount) files · \(formattedSize(archive.totalBytes))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button { showShareSheet = true } label: {
                Label("Share", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.borderedProminent)
            Button { discardPreparedArchive() } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(.bar)
    }

    private var selectionActionBar: some View {
        HStack {
            Button(role: .destructive) {
                deleteSelected()
            } label: {
                Label("Delete (\(manager.selectedFileIDs.count))",
                      systemImage: "trash")
            }
            Spacer()
            Button {
                shareSelected()
            } label: {
                Label("Share (\(manager.selectedFileIDs.count))",
                      systemImage: "square.and.arrow.up.on.square")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(.bar)
    }

    private var alertBinding: Binding<Bool> {
        Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )
    }

    private func formattedSize(_ bytes: Int64) -> String {
        let kb = Double(bytes) / 1024
        if kb < 1024 { return String(format: "%.1f KB", max(kb, 0.1)) }
        return String(format: "%.2f MB", kb / 1024)
    }
}

// MARK: - File Row

private struct FileRow: View {
    let file:    SavedFilesManager.SavedFile
    let manager: SavedFilesManager

    var body: some View {
        HStack(spacing: 12) {
            // Icon
            Image(systemName: file.ext == "csv" ? "tablecells" : "curlybraces")
                .font(.title2)
                .foregroundStyle(file.ext == "csv" ? Color.blue : Color.orange)
                .frame(width: 36)

            // Info
            VStack(alignment: .leading, spacing: 3) {
                Text(file.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(manager.formattedSize(file.sizeBytes))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(manager.formattedDate(file.modDate))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Quick share
            ShareLink(item: file.url) {
                Image(systemName: "square.and.arrow.up")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Empty State

private struct EmptyFilesState: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "doc.badge.plus")
                .font(.system(size: 52))
                .foregroundStyle(.secondary)
            Text("No saved files yet")
                .font(.headline)
            Text("Go to the Collect tab, record some samples,\nthen tap Save CSV or Save JSON.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}

// MARK: - Share Sheet wrapper

private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

// MARK: - File Detail View

struct FileDetailView: View {
    let file: SavedFilesManager.SavedFile

    var body: some View {
        Group {
            if file.ext == "csv" {
                CSVDetailView(url: file.url)
            } else {
                JSONDetailView(url: file.url)
            }
        }
        .navigationTitle(file.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                ShareLink(item: file.url) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
            }
        }
    }
}

// MARK: - CSV Detail View

private struct CSVDetailView: View {
    let url: URL

    @State private var parsed: ParsedCSV? = nil

    var body: some View {
        Group {
            if let parsed {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        CSVSummarySection(parsed: parsed)
                        Divider()
                        CSVTableSection(parsed: parsed)
                    }
                    .padding()
                }
            } else {
                ProgressView("Loading…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            DispatchQueue.global(qos: .userInitiated).async {
                let result = ParsedCSV(url: url)
                DispatchQueue.main.async { parsed = result }
            }
        }
    }
}

// MARK: - CSV Summary Section

private struct CSVSummarySection: View {
    let parsed: ParsedCSV

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Summary", systemImage: "chart.bar.fill")
                .font(.headline)

            HStack(spacing: 12) {
                SummaryTile(label: "Samples",    value: "\(parsed.rowCount)")
                SummaryTile(label: "Labels",     value: "\(parsed.labelCounts.count)")
                SummaryTile(label: "Avg Peak",
                            value: parsed.globalAvgPeak.map { String(format: "%.3f g", $0) } ?? "—")
                SummaryTile(label: "Avg RMS",
                            value: parsed.globalAvgRMS.map  { String(format: "%.3f g", $0) } ?? "—")
            }

            // Per-label breakdown
            if !parsed.labelCounts.isEmpty {
                VStack(spacing: 6) {
                    ForEach(parsed.labelCounts, id: \.label) { item in
                        LabelBreakdownRow(
                            label: item.label,
                            count: item.count,
                            total: parsed.rowCount,
                            avgPeak: parsed.avgPeakPerLabel[item.label],
                            avgRMS:  parsed.avgRMSPerLabel[item.label]
                        )
                    }
                }
            }
        }
    }
}

private struct SummaryTile: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(.headline, design: .rounded))
                .fontWeight(.bold)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct LabelBreakdownRow: View {
    let label:   String
    let count:   Int
    let total:   Int
    let avgPeak: Double?
    let avgRMS:  Double?

    private var fraction: CGFloat {
        total > 0 ? CGFloat(count) / CGFloat(total) : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                    .font(.caption).fontWeight(.semibold)
                    .foregroundStyle(labelColorFor(label))
                Spacer()
                Text("\(count)")
                    .font(.caption2).foregroundStyle(.secondary)
                if let p = avgPeak {
                    Text("peak \(String(format: "%.3f", p))g")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let r = avgRMS {
                    Text("rms \(String(format: "%.3f", r))g")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 2)
                    .fill(labelColorFor(label).opacity(0.25))
                    .frame(width: geo.size.width)
                    .overlay(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(labelColorFor(label))
                            .frame(width: geo.size.width * fraction)
                    }
            }
            .frame(height: 4)
        }
    }

    private func labelColorFor(_ l: String) -> Color {
        let comboLabels = DataCollectManager.combinationLabels
        if comboLabels.contains(l) { return .indigo }
        let defenseLabels = DataCollectManager.defenseLabels
        if defenseLabels.contains(l) { return .purple }
        if l.hasPrefix("Lead") { return .blue }
        return .orange
    }
}

// MARK: - CSV Table Section

private struct CSVTableSection: View {
    let parsed: ParsedCSV

    // Columns to display (skip id which is a UUID)
    private let displayCols = [
        "label", "source", "peak", "mean", "rms",
        "energy", "durationMs", "zeroCrossingRate",
        "impulseWidth", "variance", "sampleCount"
    ]

    private func colWidth(_ name: String) -> CGFloat {
        switch name {
        case "label":            return 110
        case "source":           return 65
        case "zeroCrossingRate": return 105
        case "durationMs":       return 85
        case "sampleCount":      return 80
        case "impulseWidth":     return 80
        default:                 return 78
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Data Table", systemImage: "tablecells")
                .font(.headline)
            Text("\(parsed.rowCount) rows  ·  scroll horizontally to see all columns")
                .font(.caption2)
                .foregroundStyle(.tertiary)

            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 0) {
                    // Header row
                    HStack(spacing: 0) {
                        ForEach(displayCols.filter { parsed.headers.contains($0) }, id: \.self) { col in
                            Text(col)
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .frame(width: colWidth(col), alignment: .leading)
                                .padding(.vertical, 6)
                                .padding(.horizontal, 4)
                                .background(Color.primary.opacity(0.06))
                        }
                    }

                    // Data rows
                    let colIndices = displayCols.compactMap { parsed.headers.firstIndex(of: $0) }
                    ForEach(parsed.rows.indices, id: \.self) { rowIdx in
                        let row = parsed.rows[rowIdx]
                        HStack(spacing: 0) {
                            ForEach(Array(zip(displayCols, colIndices)), id: \.0) { col, cellIdx in
                                let cellValue = cellIdx < row.count ? row[cellIdx] : ""
                                Text(cellValue)
                                    .font(.system(size: 10, design: .monospaced))
                                    .lineLimit(1)
                                    .frame(width: colWidth(col), alignment: .leading)
                                    .padding(.vertical, 5)
                                    .padding(.horizontal, 4)
                                    .background(rowIdx % 2 == 0
                                                ? Color.clear
                                                : Color.primary.opacity(0.03))
                            }
                        }
                        Divider().opacity(0.3)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.08)))
        }
    }
}

// MARK: - JSON Detail View

private struct JSONDetailView: View {
    let url: URL
    @State private var text: String = ""

    var body: some View {
        ScrollView {
            Text(text.isEmpty ? "Loading…" : text)
                .font(.system(size: 11, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .onAppear {
            DispatchQueue.global(qos: .userInitiated).async {
                let content = (try? String(contentsOf: url, encoding: .utf8)) ?? "Failed to load."
                DispatchQueue.main.async { text = content }
            }
        }
    }
}
