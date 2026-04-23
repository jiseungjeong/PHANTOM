import SwiftUI

// MARK: - Files View
//
// Browses all exported CSV / JSON files saved to the app's Documents folder.
// Tap a file to inspect its contents; use the share button to share it.

struct FilesView: View {
    @StateObject private var manager = SavedFilesManager()

    var body: some View {
        NavigationStack {
            Group {
                if manager.files.isEmpty {
                    EmptyFilesState()
                } else {
                    List {
                        ForEach(manager.files) { file in
                            NavigationLink {
                                FileDetailView(file: file)
                            } label: {
                                FileRow(file: file, manager: manager)
                            }
                        }
                        .onDelete { offsets in
                            offsets.map { manager.files[$0] }.forEach { manager.delete($0) }
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("Saved Files")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                if !manager.files.isEmpty {
                    EditButton()
                }
            }
        }
        // onAppear must be on NavigationStack (outside it), not on a child view inside,
        // so it fires every time this tab is selected.
        .onAppear { manager.refresh() }
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

