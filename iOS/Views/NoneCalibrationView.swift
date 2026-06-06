import SwiftUI

// MARK: - None Calibration View (Phase 9 (2) / §8.7)
//
// Teaches the model to REJECT non-boxing motion (walking, raising an arm). The
// user demonstrates each non-boxing motion once; each demo is cut to its
// strongest segment and appended to the KNN reference under the shared "none"
// label. During a session a window that classifies as "none" is dropped from
// stats and the combo timeline (SessionAnalyzer / D3).
//
// NOTE: standing still is NOT calibrated here — that is handled by the idle gate
// (gm_max < idleGmThreshold). This tab only covers HIGH-energy non-boxing motion
// that would otherwise be forced into a boxing class (D1).
//
// The motion list is user-editable (add / rename / delete custom, reset) via
// NoneMotionStore. Progress is tracked by motion id so renaming never loses it.

struct NoneCalibrationView: View {
    @EnvironmentObject var session: BoxingSessionManager
    @StateObject private var store = NoneMotionStore()

    @State private var index = 0
    @State private var captureFailed = false
    /// IDs of motions confirmed this run (M1: id-based so rename keeps progress;
    /// M3: in-memory only, reset each app launch).
    @State private var doneIDs: Set<UUID> = []

    // Add / rename sheets
    @State private var showingAdd = false
    @State private var newName = ""
    @State private var renameTarget: NoneMotion? = nil
    @State private var renameText = ""

    private var motions: [NoneMotion] { store.motions }
    private var isDone: Bool { index >= motions.count }
    private var current: NoneMotion? { index < motions.count ? motions[index] : nil }
    private var reviewing: Bool { session.lastCalibrationPreview != nil }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                header

                if isDone {
                    completionCard
                } else {
                    motionCard
                }

                if !session.isCalibrationCapturing && !reviewing {
                    motionsListCard
                }
            }
            .padding()
        }
        .navigationTitle("Reject Motions")
        .alert("Add motion", isPresented: $showingAdd) {
            TextField("Motion name", text: $newName)
            Button("Add") { store.add(name: newName); newName = "" }
            Button("Cancel", role: .cancel) { newName = "" }
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 6) {
            Image(systemName: "hand.raised.slash")
                .font(.system(size: 36))
                .foregroundStyle(.orange)
            Text("Teach non-boxing motions")
                .font(.title3.bold())
            Text("Demonstrate everyday motions (walking, raising an arm) so the app ignores them during a round. Standing still is already ignored automatically.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Current motion card

    private var motionCard: some View {
        VStack(spacing: 16) {
            Text(current?.name ?? "")
                .font(.title2.bold())
            Text("\(index + 1) of \(motions.count)")
                .font(.caption)
                .foregroundStyle(.secondary)

            if reviewing, let preview = session.lastCalibrationPreview {
                reviewSection(preview)
            } else if session.isCalibrationCapturing {
                capturingSection
            } else {
                idleSection
            }

            if captureFailed {
                Text("Couldn't detect a clear motion — perform it more distinctly and try again.")
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // Before recording
    private var idleSection: some View {
        VStack(spacing: 12) {
            Button {
                captureFailed = false
                session.startCalibrationCapture()
            } label: {
                Label("Start demo", systemImage: "figure.walk")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            Button("Skip this motion") { advance() }
                .font(.footnote)
        }
    }

    // While recording
    private var capturingSection: some View {
        VStack(spacing: 12) {
            Label("Recording… perform the motion now", systemImage: "record.circle")
                .foregroundStyle(.red)
            Button {
                // Commit under the shared "none" label regardless of display name.
                let ok = session.stopCalibrationCapture(label: MotionRuleClassifier.noneLabel)
                captureFailed = !ok
            } label: {
                Text("Done").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
    }

    // After recording: review the detected peak + crop window
    private func reviewSection(_ preview: SessionAnalyzer.SegmentPreview) -> some View {
        VStack(spacing: 12) {
            Text("Detected motion — does this look right?")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            PeakPreviewGraph(preview: preview)
                .frame(height: 120)
                .padding(.vertical, 4)
            HStack(spacing: 12) {
                Button {
                    session.retryCalibration()
                } label: {
                    Label("Retry", systemImage: "arrow.counterclockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                Button {
                    session.confirmCalibration()
                    if let c = current { doneIDs.insert(c.id) }
                    advance()
                } label: {
                    Label("Confirm", systemImage: "checkmark")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
    }

    // MARK: - Motions list

    private var motionsListCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Motions").font(.headline)
                Spacer()
                Button { showingAdd = true } label: {
                    Label("Add", systemImage: "plus.circle.fill")
                        .font(.subheadline)
                }
            }

            // Swipe a row to Rename (any) / Delete (custom only). M4.
            List {
                ForEach(Array(motions.enumerated()), id: \.element.id) { i, motion in
                    motionRow(i: i, motion: motion)
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            if !motion.isDefault {
                                Button(role: .destructive) {
                                    store.delete(motion.id)
                                    doneIDs.remove(motion.id)
                                    if index >= motions.count { index = max(0, motions.count - 1) }
                                } label: { Label("Delete", systemImage: "trash") }
                            }
                            Button {
                                renameText = motion.name
                                renameTarget = motion
                            } label: { Label("Rename", systemImage: "pencil") }
                            .tint(.blue)
                        }
                }
            }
            .listStyle(.plain)
            .frame(height: CGFloat(motions.count) * 48 + 8)
            .scrollDisabled(true)

            Button(role: .destructive) {
                store.resetToDefaults()
                doneIDs.removeAll()
                index = 0
            } label: {
                Label("Reset to defaults", systemImage: "arrow.counterclockwise")
                    .font(.footnote)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .alert("Rename motion", isPresented: Binding(
            get: { renameTarget != nil },
            set: { if !$0 { renameTarget = nil } })) {
            TextField("Motion name", text: $renameText)
            Button("Save") {
                if let t = renameTarget { store.rename(t.id, to: renameText) }
                renameTarget = nil
            }
            Button("Cancel", role: .cancel) { renameTarget = nil }
        }
    }

    private func motionRow(i: Int, motion: NoneMotion) -> some View {
        let isCurrent = (i == index && !isDone)
        let isDoneRow = doneIDs.contains(motion.id)
        // U1: current row shows a FILLED left circle (orange); done = green check;
        // pending = empty circle.
        let symbol = isDoneRow ? "checkmark.circle.fill"
            : (isCurrent ? "largecircle.fill.circle" : "circle")
        let color: Color = isDoneRow ? .green : (isCurrent ? .orange : .secondary)
        return HStack {
            Image(systemName: symbol)
                .foregroundStyle(color)
            Text(motion.name)
            if !motion.isDefault {
                Text("custom").font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if !session.isCalibrationCapturing && !reviewing { index = i }
        }
    }

    // MARK: - Completion

    private var completionCard: some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 44))
                .foregroundStyle(.green)
            Text("Done")
                .font(.title2.bold())
            Text("\(doneIDs.count) non-boxing motion(s) added. The app will ignore them during a round.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Label("Saved on this device — kept after restart", systemImage: "internaldrive")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button {
                index = 0
            } label: {
                Label("Demonstrate again", systemImage: "arrow.clockwise")
            }
            .font(.footnote)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Helpers

    private func advance() {
        captureFailed = false
        if index < motions.count { index += 1 }
    }
}
