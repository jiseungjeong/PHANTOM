import SwiftUI

// MARK: - Calibration View (Phase 5 d / §8.4)
//
// On-device few-shot calibration. The user demonstrates each of the 10 actions
// ONCE; each demo is cut to its strongest segment and, after the user reviews the
// detected peak + crop window, appended (labeled) to the KNN reference. This
// personalizes the model on-device (one demo per action lifts unseen-person
// macro-F1 ~0.57 -> ~0.90 in our experiments).
//
// Per action:
//   [Start demo] -> perform action -> [Done] -> REVIEW (peak/crop graph) ->
//   [Confirm] (append + advance)  OR  [Retry this action] (discard, re-record)
//
// The review graph (envelope curve + red peak line + crop band) lets the user
// catch a bad capture before committing — so one mistake never forces a full reset.

struct CalibrationView: View {
    @EnvironmentObject var session: BoxingSessionManager

    private let actions = MotionRuleClassifier.calibrationOrder
    @State private var index = 0
    @State private var captureFailed = false
    /// Non-nil = single-action mode (re-calibrate just this action, tapped from the
    /// list). nil = sequential mode (the 10-action walkthrough driven by `index`).
    @State private var targetAction: String? = nil

    private var singleMode: Bool { targetAction != nil }
    private var isDone: Bool { !singleMode && index >= actions.count }
    private var currentAction: String { targetAction ?? (index < actions.count ? actions[index] : "") }
    private var reviewing: Bool { session.lastCalibrationPreview != nil }

    /// Timestamp lookup for an action (nil = not yet calibrated).
    private func savedAt(_ action: String) -> Date? {
        session.calibrationEntries.first { $0.label == action }?.savedAt
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                header

                if singleMode {
                    actionCard          // re-calibrating one tapped action
                } else if isDone {
                    completionCard
                } else {
                    actionCard          // sequential walkthrough
                }

                // The per-action list (all 10) is hidden while actively capturing,
                // to keep focus on the current demo.
                if !session.isCalibrationCapturing && !reviewing {
                    actionsListCard
                }

                if session.calibrationCount > 0 && !singleMode {
                    Button(role: .destructive) {
                        session.resetCalibration()
                        index = 0
                        captureFailed = false
                    } label: {
                        Label("Reset all calibration", systemImage: "arrow.counterclockwise")
                    }
                    .padding(.top, 4)
                }
            }
            .padding()
        }
        .navigationTitle("Calibration")
    }

    // MARK: - Per-action list (all 10; tap an action to re-calibrate just it)

    private var actionsListCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Actions — tap one to (re)calibrate")
                .font(.subheadline.bold())
            ForEach(actions, id: \.self) { action in
                Button {
                    targetAction = action      // enter single-action mode
                    captureFailed = false
                } label: {
                    HStack {
                        Text(action)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                        Spacer()
                        if let t = savedAt(action) {
                            VStack(alignment: .trailing, spacing: 1) {
                                Text(t.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption)
                                Text(t.formatted(.relative(presentation: .named)))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        } else {
                            Text("Not yet")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Image(systemName: "chevron.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                .buttonStyle(.plain)
                Divider()
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 6) {
            Text("Personalize to you")
                .font(.headline)
            Text("Demonstrate each action once. The app learns your style on-device.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            ProgressView(value: Double(min(index, actions.count)),
                         total: Double(actions.count))
                .padding(.top, 4)
            Text("\(min(index, actions.count)) / \(actions.count) actions  ·  \(session.calibrationCount) samples added")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Action capture / review card

    private var actionCard: some View {
        VStack(spacing: 16) {
            Text(singleMode ? "Re-calibrate" : "Action \(index + 1) of \(actions.count)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(currentAction)
                .font(.largeTitle.bold())
                .multilineTextAlignment(.center)

            if reviewing, let preview = session.lastCalibrationPreview {
                reviewSection(preview)
            } else if session.isCalibrationCapturing {
                capturingSection
            } else {
                idleSection
            }

            if captureFailed {
                Text("Couldn't detect a clear motion — perform the action more distinctly and try again.")
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
                Label("Start demo", systemImage: "figure.boxing")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            Button(singleMode ? "Cancel" : "Skip this action") { advance() }
                .font(.footnote)
        }
    }

    // While recording
    private var capturingSection: some View {
        VStack(spacing: 12) {
            Label("Recording… perform the action now", systemImage: "record.circle")
                .foregroundStyle(.red)
            Button {
                let ok = session.stopCalibrationCapture(label: currentAction)
                captureFailed = !ok
            } label: {
                Text("Done").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
    }

    // After recording: show the detected peak + crop window for review
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
                    session.retryCalibration()   // discard, re-record same action
                } label: {
                    Label("Retry", systemImage: "arrow.counterclockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                Button {
                    session.confirmCalibration()
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

    // MARK: - Completion

    private var completionCard: some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 44))
                .foregroundStyle(.green)
            Text("Calibration complete")
                .font(.title2.bold())
            Text("\(session.calibrationCount) of your samples added. The classifier is now personalized to you.")
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

    /// After a demo is confirmed/skipped/cancelled: single mode returns to the list,
    /// sequential mode moves to the next action.
    private func advance() {
        captureFailed = false
        if singleMode {
            targetAction = nil          // exit single-action mode -> back to list
        } else {
            index += 1
        }
    }
}

// MARK: - Peak / crop-window mini graph
//
// Mirrors viz_segmentation.py one cell: blue envelope curve, red peak line, and a
// shaded band for the cropped window [windowStartS, windowEndS]. Lets the user see
// where the peak landed and exactly which span was cut before committing.

private struct PeakPreviewGraph: View {
    let preview: SessionAnalyzer.SegmentPreview

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let times = preview.times
            let env = preview.envelope
            let tMin = times.first ?? 0
            let tMax = times.last ?? 1
            let tSpan = max(tMax - tMin, 1e-6)
            let eMax = max(env.max() ?? 1, 1e-6)

            // Closures (not `func`) — a result-builder closure can't contain declarations.
            let x: (Double) -> CGFloat = { CGFloat(($0 - tMin) / tSpan) * w }
            let y: (Double) -> CGFloat = { h - CGFloat($0 / eMax) * (h - 6) - 3 }

            ZStack {
                // crop window band
                let bx0 = x(preview.windowStartS)
                let bx1 = x(preview.windowEndS)
                Rectangle()
                    .fill(Color.accentColor.opacity(0.18))
                    .frame(width: max(bx1 - bx0, 1), height: h)
                    .position(x: (bx0 + bx1) / 2, y: h / 2)

                // envelope curve
                Path { p in
                    guard times.count == env.count, times.count > 1 else { return }
                    p.move(to: CGPoint(x: x(times[0]), y: y(env[0])))
                    for i in 1..<times.count {
                        p.addLine(to: CGPoint(x: x(times[i]), y: y(env[i])))
                    }
                }
                .stroke(Color.blue, lineWidth: 1.6)

                // peak line
                Path { p in
                    let px = x(preview.peakTimeS)
                    p.move(to: CGPoint(x: px, y: 0))
                    p.addLine(to: CGPoint(x: px, y: h))
                }
                .stroke(Color.red, lineWidth: 1.4)
            }
        }
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .bottomLeading) {
            Text("blue = motion · red = peak · band = crop")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(4)
        }
    }
}
