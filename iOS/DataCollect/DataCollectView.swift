import SwiftUI
import UniformTypeIdentifiers

// MARK: - Data Collect View
//
// Lets the user select a gesture label, record labeled spike windows from
// iPhone IMU (and Watch if connected), inspect extracted features per sample,
// and save CSV / JSON files to the app's Documents folder (visible in Files tab).

struct DataCollectView: View {
    @EnvironmentObject var collector: DataCollectManager

    @State private var showClearConfirm = false
    @State private var saveToast: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    LabelPickerCard(selectedLabel: $collector.selectedLabel)
                    AirPodsStatusRow(isRecording: collector.isRecording,
                                     airPodsAvailable: collector.airPodsAvailable)
                    RecordButton(isRecording: collector.isRecording) {
                        collector.isRecording ? collector.stopRecording() : collector.startRecording()
                    }
                    if !collector.samples.isEmpty {
                        SummaryBar(collector: collector)
                        SampleList(collector: collector)
                    } else {
                        EmptyStateView(isRecording: collector.isRecording)
                    }
                    Spacer(minLength: 30)
                }
                .padding()
            }
            .navigationTitle("Data Collect")
            .navigationBarTitleDisplayMode(.large)
            .toolbar { toolbarContent }
            .confirmationDialog("Clear all \(collector.samples.count) samples?",
                                isPresented: $showClearConfirm,
                                titleVisibility: .visible) {
                Button("Clear All", role: .destructive) { collector.clearAll() }
            }
            .overlay(alignment: .top) { toastOverlay }
            .animation(.spring(duration: 0.35), value: saveToast)
        }
    }

    // MARK: - Toast

    @ViewBuilder
    private var toastOverlay: some View {
        if let msg = saveToast {
            Label(msg, systemImage: "checkmark.circle.fill")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 9)
                .background(Color.green.opacity(0.92))
                .clipShape(Capsule())
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
                .onAppear {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        withAnimation { saveToast = nil }
                    }
                }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        if !collector.samples.isEmpty {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(role: .destructive) { showClearConfirm = true } label: {
                    Label("Clear", systemImage: "trash")
                }
                .foregroundStyle(.red)
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Button {
                        do {
                            try collector.saveCSV()
                            withAnimation { saveToast = "CSV saved — see Files tab" }
                        } catch {
                            withAnimation { saveToast = "Save failed: \(error.localizedDescription)" }
                        }
                    } label: {
                        Label("Save CSV", systemImage: "tablecells")
                    }
                    Button {
                        do {
                            try collector.saveJSON()
                            withAnimation { saveToast = "JSON saved — see Files tab" }
                        } catch {
                            withAnimation { saveToast = "Save failed: \(error.localizedDescription)" }
                        }
                    } label: {
                        Label("Save JSON (with raw windows)", systemImage: "curlybraces")
                    }
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
            }
        }
    }
}

// MARK: - Label Picker Card

private struct LabelPickerCard: View {
    @Binding var selectedLabel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Next Gesture Label", systemImage: "tag.fill")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("Label", selection: $selectedLabel) {
                Section("Punch") {
                    ForEach(DataCollectManager.punchLabels, id: \.self) { label in
                        Text(label).tag(label)
                    }
                }
                Section("Defense") {
                    ForEach(DataCollectManager.defenseLabels, id: \.self) { label in
                        Text(label).tag(label)
                    }
                }
                Section("Combination") {
                    ForEach(DataCollectManager.combinationLabels, id: \.self) { label in
                        Text(label).tag(label)
                    }
                }
            }
            .pickerStyle(.menu)
            .tint(labelColor(selectedLabel))
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - AirPods Status Row

private struct AirPodsStatusRow: View {
    let isRecording: Bool
    let airPodsAvailable: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "airpodspro")
                .foregroundStyle(color)
            Text(statusText)
                .font(.caption)
                .foregroundStyle(color)
            Spacer()
            if isRecording && !airPodsAvailable {
                Text("Head gestures only work with AirPods Pro connected")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var color: Color {
        if isRecording && airPodsAvailable { return .green }
        if isRecording && !airPodsAvailable { return .orange }
        return .secondary
    }

    private var statusText: String {
        if isRecording && airPodsAvailable { return "AirPods: Tracking" }
        if isRecording && !airPodsAvailable { return "AirPods: Not connected" }
        return "AirPods Pro required for defense labels"
    }
}

// MARK: - Record Button

private struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if isRecording {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 10, height: 10)
                }
                Text(isRecording ? "Stop Recording" : "Start Recording")
                    .font(.headline)
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

// MARK: - Summary Bar

private struct SummaryBar: View {
    let collector: DataCollectManager

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                SummaryChip(value: collector.samples.count,      label: "Total",  color: .primary)
                SummaryChip(value: collector.punchCount,         label: "Punch",  color: .blue)
                SummaryChip(value: collector.defenseCount,       label: "Defense", color: .purple)
                SummaryChip(value: collector.combinationCount,   label: "Combo",  color: .indigo)
            }
            HStack(spacing: 8) {
                SummaryChip(value: collector.iPhoneCount,  label: "iPhone",  color: .teal)
                SummaryChip(value: collector.watchCount,   label: "Watch",   color: .orange)
                SummaryChip(value: collector.airPodsCount, label: "AirPods", color: .green)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

private struct SummaryChip: View {
    let value: Int
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text("\(value)")
                .font(.system(.headline, design: .rounded))
                .fontWeight(.bold)
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Sample List

private struct SampleList: View {
    let collector: DataCollectManager

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Samples")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(collector.samples.count) recorded")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            ForEach(collector.samples.reversed()) { sample in
                SampleCard(sample: sample) {
                    collector.delete(sample)
                }
            }
        }
    }
}

// MARK: - Sample Card

private struct SampleCard: View {
    let sample: DataCollectSample
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                LabelBadge(text: sample.label)
                SourceBadge(source: sample.source)
                Spacer()
                Text(String(format: "%.0f ms", sample.tsMs))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Button(action: onDelete) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                }
            }

            SparklineView(values: sample.window, color: labelColor(sample.label))
                .frame(height: 36)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                FeatureCell(name: "Peak",     value: String(format: "%.3f g",  sample.peak))
                FeatureCell(name: "RMS",      value: String(format: "%.3f g",  sample.rms))
                FeatureCell(name: "Duration", value: String(format: "%.0f ms", sample.durationMs))
                FeatureCell(name: "ZCR",      value: String(format: "%.3f",    sample.zeroCrossingRate))
                FeatureCell(name: "Energy",   value: String(format: "%.2f",    sample.energy))
                FeatureCell(name: "Width",    value: "\(sample.impulseWidth) smp")
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Sparkline

struct SparklineView: View {
    let values: [Double]
    var color: Color = .blue

    var body: some View {
        GeometryReader { _ in
            Canvas { ctx, size in
                guard values.count > 1 else { return }
                let minV  = values.min() ?? 0
                let maxV  = values.max() ?? 1
                let range = max(maxV - minV, 1e-6)
                let w = size.width
                let h = size.height

                var path = Path()
                for (i, v) in values.enumerated() {
                    let x = CGFloat(i) / CGFloat(values.count - 1) * w
                    let y = h - CGFloat((v - minV) / range) * h
                    if i == 0 { path.move(to: .init(x: x, y: y)) }
                    else       { path.addLine(to: .init(x: x, y: y)) }
                }
                ctx.stroke(path, with: .color(color), lineWidth: 1.5)
            }
        }
    }
}

// MARK: - Feature Cell

private struct FeatureCell: View {
    let name: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name)
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.semibold)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - Badges

private struct LabelBadge: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.caption).fontWeight(.semibold)
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(labelColor(text).opacity(0.15))
            .foregroundStyle(labelColor(text))
            .clipShape(Capsule())
    }
}

private struct SourceBadge: View {
    let source: String

    private var color: Color {
        switch source {
        case "Watch":   return .orange
        case "AirPods": return .green
        default:        return .teal   // iPhone
        }
    }

    var body: some View {
        Text(source)
            .font(.system(size: 10))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}

// MARK: - Empty State

private struct EmptyStateView: View {
    let isRecording: Bool
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: isRecording ? "record.circle" : "waveform.path.badge.plus")
                .font(.system(size: 44))
                .foregroundStyle(isRecording ? .red : .secondary)
            Text(isRecording
                 ? "Perform a gesture to capture a sample"
                 : "Select a label and tap Start Recording")
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }
}

// MARK: - Label color helper

@MainActor func labelColor(_ label: String) -> Color {
    if DataCollectManager.combinationLabels.contains(label) { return .indigo }
    if DataCollectManager.defenseLabels.contains(label) { return .purple }
    if label.hasPrefix("Lead") { return .blue }
    return .orange
}
