import SwiftUI

// MARK: - Calibration View (Phase 5 d)
//
// On-device few-shot calibration. The user demonstrates each of the 10 actions
// ONCE; each demo is cut to its strongest segment and appended (labeled) to the
// KNN reference (MotionRuleClassifier.addCalibrationSample). This personalizes the
// model to the current user — in our experiments one demo per action lifts
// unseen-person macro-F1 from ~0.57 to ~0.90.
//
// Flow per action: [Start demo] → user performs the action → [Done] → advance.
// Preprocessing stays frozen; only the reference set grows (no retraining).

struct CalibrationView: View {
    @EnvironmentObject var session: BoxingSessionManager

    private let actions = MotionRuleClassifier.calibrationOrder
    @State private var index = 0
    @State private var lastAddFailed = false

    private var isDone: Bool { index >= actions.count }
    private var currentAction: String { isDone ? "" : actions[index] }

    var body: some View {
        VStack(spacing: 20) {
            header

            if isDone {
                completionCard
            } else {
                actionCard
            }

            Spacer(minLength: 12)

            if session.calibrationCount > 0 {
                Button(role: .destructive) {
                    session.resetCalibration()
                    index = 0
                    lastAddFailed = false
                } label: {
                    Label("Reset calibration", systemImage: "arrow.counterclockwise")
                }
                .padding(.bottom, 8)
            }
        }
        .padding()
        .navigationTitle("Calibration")
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

    // MARK: - Action capture card

    private var actionCard: some View {
        VStack(spacing: 16) {
            Text("Action \(index + 1) of \(actions.count)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(currentAction)
                .font(.largeTitle.bold())
                .multilineTextAlignment(.center)

            if session.isCalibrationCapturing {
                Label("Recording… perform the action now", systemImage: "record.circle")
                    .foregroundStyle(.red)
                Button {
                    let ok = session.stopCalibrationCapture(label: currentAction)
                    lastAddFailed = !ok
                    if ok { index += 1 }
                } label: {
                    Text("Done").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else {
                Button {
                    lastAddFailed = false
                    session.startCalibrationCapture()
                } label: {
                    Label("Start demo", systemImage: "figure.boxing")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                Button("Skip this action") { index += 1 }
                    .font(.footnote)
            }

            if lastAddFailed {
                Text("Couldn't read a clear motion — try again (move more distinctly).")
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
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
}
