import SwiftUI

// MARK: - Nonisolated label arrays
// Defined at file scope (not inside a @MainActor class) to avoid the
// "main actor-isolated static property from nonisolated context" warning.

private let _punchLabels: [String] = PunchLabel.allCases
    .filter { $0 != .leadPunch && $0 != .rearPunch }
    .map(\.rawValue)

private let _defenseLabels: [String] = DefenseLabel.allCases.map(\.rawValue)
private let _combinationLabels: [String] = CombinationLabel.allCases.map(\.rawValue)

// MARK: - Badge color helper

private func tsBadgeColor(_ label: String) -> Color {
    if _combinationLabels.contains(label) { return .indigo }
    if _defenseLabels.contains(label) { return .purple }
    if label.hasPrefix("Lead")        { return .blue }
    return .orange
}

// MARK: - Time Series Collect View

struct TimeSeriesCollectView: View {
    @EnvironmentObject var manager: TimeSeriesCollectManager

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ParticipantPrefixCard(
                        participantId: $manager.participantId,
                        disabled: manager.isRecording
                    )

                    // ── iPhone charts ──────────────────────────────────────
                    SectionHeader(title: "iPhone", icon: "iphone", color: .teal)

                    TSChartCard(
                        title: "Accelerometer",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.accXBuf, manager.accYBuf, manager.accZBuf],
                        yRange: -3.0...3.0
                    )
                    TSChartCard(
                        title: "Gyroscope",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.gyroXBuf, manager.gyroYBuf, manager.gyroZBuf],
                        yRange: -5.0...5.0
                    )

                    // ── AirPods charts ─────────────────────────────────────
                    SectionHeader(title: "AirPods",
                                  icon: "airpodspro",
                                  color: .green,
                                  badge: manager.airPodsConnected ? nil : "not connected")

                    TSChartCard(
                        title: "Accelerometer",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.apAccXBuf, manager.apAccYBuf, manager.apAccZBuf],
                        yRange: -3.0...3.0,
                        dimmed: !manager.airPodsConnected
                    )
                    TSChartCard(
                        title: "Gyroscope",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.apGyroXBuf, manager.apGyroYBuf, manager.apGyroZBuf],
                        yRange: -5.0...5.0,
                        dimmed: !manager.airPodsConnected
                    )

                    // ── Watch charts ───────────────────────────────────────
                    SectionHeader(title: "Apple Watch",
                                  icon: "applewatch",
                                  color: .orange,
                                  badge: manager.watchReachable ? nil : "not reachable")

                    TSChartCard(
                        title: "Accelerometer",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.waAccXBuf, manager.waAccYBuf, manager.waAccZBuf],
                        yRange: -3.0...3.0,
                        dimmed: !manager.watchReachable
                    )
                    TSChartCard(
                        title: "Gyroscope",
                        labels: ["X", "Y", "Z"],
                        colors: [.red, .green, .blue],
                        data: [manager.waGyroXBuf, manager.waGyroYBuf, manager.waGyroZBuf],
                        yRange: -5.0...5.0,
                        dimmed: !manager.watchReachable
                    )

                    // ── Status + controls ──────────────────────────────────
                    RecordingStatusRow(manager: manager)
                    DeviceStatusGrid(manager: manager)

                    TSLabelPicker(selectedLabel: $manager.selectedLabel,
                                  disabled: manager.isRecording)

                    TSRecordButton(isRecording: manager.isRecording) {
                        manager.isRecording ? manager.stopRecording() : manager.startRecording()
                    }

                    if !manager.isRecording, let url = manager.lastSavedURL {
                        SavedBanner(filename: url.lastPathComponent)
                    }

                    Spacer(minLength: 30)
                }
                .padding()
            }
            .navigationTitle("Raw Stream")
            .navigationBarTitleDisplayMode(.large)
            .toolbar { toolbarContent }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        if manager.totalSampleCount > 0 && !manager.isRecording {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Menu("Save CSV") {
                        ForEach(TimeSeriesCollectManager.ExportSource.allCases, id: \.self) { src in
                            Button(src == .all ? "All devices" : src.rawValue) {
                                _ = try? manager.saveCSV(source: src)
                            }
                        }
                    }
                    Menu("Save JSON") {
                        ForEach(TimeSeriesCollectManager.ExportSource.allCases, id: \.self) { src in
                            Button(src == .all ? "All devices" : src.rawValue) {
                                _ = try? manager.saveJSON(source: src)
                            }
                        }
                    }
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
            }
        }
    }
}

// MARK: - Participant Prefix

private struct ParticipantPrefixCard: View {
    @Binding var participantId: String
    let disabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("File Prefix", systemImage: "person.text.rectangle")
                .font(.caption)
                .foregroundStyle(.secondary)

            TextField("P03", text: $participantId)
                .textInputAutocapitalization(.characters)
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
                .disabled(disabled)
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Section Header

private struct SectionHeader: View {
    let title: String
    let icon:  String
    let color: Color
    var badge: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon).foregroundStyle(color)
            Text(title).font(.headline)
            if let badge {
                Text(badge)
                    .font(.system(size: 10))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Color.secondary.opacity(0.15))
                    .foregroundStyle(.secondary)
                    .clipShape(Capsule())
            }
        }
    }
}

// MARK: - Live Chart Card

private struct TSChartCard: View {
    let title:  String
    let labels: [String]
    let colors: [Color]
    let data:   [[Double]]
    let yRange: ClosedRange<Double>
    var dimmed: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline).fontWeight(.semibold)
                Spacer()
                HStack(spacing: 8) {
                    ForEach(Array(zip(labels, colors)), id: \.0) { label, color in
                        HStack(spacing: 3) {
                            Circle().fill(color).frame(width: 6, height: 6)
                            Text(label).font(.system(size: 10)).foregroundStyle(.secondary)
                        }
                    }
                }
            }

            ZStack {
                VStack(spacing: 0) {
                    ForEach(0..<5) { _ in Divider(); Spacer() }
                }
                .opacity(0.25)

                if dimmed {
                    Text("Waiting for data…")
                        .font(.caption).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Canvas { ctx, size in
                        guard !data.isEmpty else { return }
                        let rangeSpan = yRange.upperBound - yRange.lowerBound
                        for (series, color) in zip(data, colors) {
                            guard series.count > 1 else { continue }
                            var path = Path()
                            for (i, v) in series.enumerated() {
                                let x = CGFloat(i) / CGFloat(series.count - 1) * size.width
                                let y = size.height - CGFloat((v - yRange.lowerBound) / rangeSpan) * size.height
                                if i == 0 { path.move(to: .init(x: x, y: y)) }
                                else       { path.addLine(to: .init(x: x, y: y)) }
                            }
                            ctx.stroke(path, with: .color(color), lineWidth: 1.2)
                        }
                    }
                }
            }
            .frame(height: 100)
            .background(Color.primary.opacity(0.03))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .opacity(dimmed ? 0.5 : 1.0)
    }
}

// MARK: - Recording Status Row

private struct RecordingStatusRow: View {
    let manager: TimeSeriesCollectManager

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Duration").font(.caption).foregroundStyle(.secondary)
                Text(formattedTime)
                    .font(.system(.title2, design: .monospaced)).fontWeight(.bold)
                    .foregroundStyle(manager.isRecording ? .red : .primary)
                    .contentTransition(.numericText())
                    .animation(.linear, value: manager.elapsedSeconds)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("Samples").font(.caption).foregroundStyle(.secondary)
                Text("\(manager.totalSampleCount)")
                    .font(.system(.title2, design: .monospaced)).fontWeight(.semibold)
                    .contentTransition(.numericText())
                    .animation(.linear, value: manager.totalSampleCount)
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var formattedTime: String {
        let h = manager.elapsedSeconds / 3600
        let m = (manager.elapsedSeconds % 3600) / 60
        let s = manager.elapsedSeconds % 60
        return h > 0
            ? String(format: "%d:%02d:%02d", h, m, s)
            : String(format: "%02d:%02d", m, s)
    }
}

// MARK: - Device Status Grid

private struct DeviceStatusGrid: View {
    let manager: TimeSeriesCollectManager

    var body: some View {
        HStack(spacing: 8) {
            DeviceStatusTile(name: "iPhone",  icon: "iphone",     count: manager.iPhoneSampleCount,
                             color: .teal,   isRecording: manager.isRecording, connected: true)
            DeviceStatusTile(name: "Watch",   icon: "applewatch", count: manager.watchSampleCount,
                             color: .orange, isRecording: manager.isRecording, connected: manager.watchReachable)
            DeviceStatusTile(name: "AirPods", icon: "airpodspro", count: manager.airPodsSampleCount,
                             color: .green,  isRecording: manager.isRecording, connected: manager.airPodsConnected)
        }
    }
}

private struct DeviceStatusTile: View {
    let name:        String
    let icon:        String
    let count:       Int
    let color:       Color
    let isRecording: Bool
    let connected:   Bool

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon).font(.title3).foregroundStyle(connected ? color : .secondary)
            Text("\(count)")
                .font(.system(.headline, design: .monospaced)).fontWeight(.bold)
                .foregroundStyle(connected ? color : .secondary)
                .contentTransition(.numericText())
                .animation(.linear, value: count)
            Text(name).font(.system(size: 10)).foregroundStyle(.secondary)
            Circle().fill(connected && isRecording ? Color.green : Color.secondary)
                .frame(width: 6, height: 6)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Label Picker

private struct TSLabelPicker: View {
    @Binding var selectedLabel: String
    let disabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Activity Label").font(.headline)

            if disabled {
                Text(selectedLabel)
                    .font(.subheadline)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(tsBadgeColor(selectedLabel).opacity(0.15))
                    .foregroundStyle(tsBadgeColor(selectedLabel))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                LabelGroupSection(
                    title: "Attack",
                    icon: "figure.boxing",
                    labels: _punchLabels,
                    selectedLabel: $selectedLabel
                )

                LabelGroupSection(
                    title: "Defense",
                    icon: "shield.fill",
                    labels: _defenseLabels,
                    selectedLabel: $selectedLabel
                )

                LabelGroupSection(
                    title: "Combination",
                    icon: "point.3.connected.trianglepath.dotted",
                    labels: _combinationLabels,
                    columnCount: 2,
                    selectedLabel: $selectedLabel
                )
            }
        }
    }
}

private struct LabelGroupSection: View {
    let title: String
    let icon: String
    let labels: [String]
    var columnCount: Int = 3
    @Binding var selectedLabel: String

    private var columns: [GridItem] {
        Array(repeating: GridItem(.flexible()), count: columnCount)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(labels, id: \.self) { label in
                    Button { selectedLabel = label } label: {
                        Text(label)
                            .font(.caption)
                            .lineLimit(columnCount == 2 ? 3 : 2)
                            .multilineTextAlignment(.center)
                            .padding(.vertical, 8)
                            .padding(.horizontal, 4)
                            .frame(maxWidth: .infinity, minHeight: columnCount == 2 ? 56 : 44)
                            .background(selectedLabel == label
                                        ? tsBadgeColor(label) : Color(.systemGray5))
                            .foregroundStyle(selectedLabel == label ? .white : .primary)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

// MARK: - Record Button

private struct TSRecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if isRecording { Circle().fill(Color.red).frame(width: 10, height: 10) }
                Text(isRecording ? "Stop Recording" : "Start Recording").font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(isRecording ? Color.red : Color.green)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Saved Banner

private struct SavedBanner: View {
    let filename: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            VStack(alignment: .leading, spacing: 2) {
                Text("Saved to Files").font(.caption).fontWeight(.semibold)
                Text(filename).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(Color.green.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
