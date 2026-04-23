import SwiftUI

struct TransitionTimeCard: View {
    let lastTTms:    Double?
    let averageTTms: Double?

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Image(systemName: "arrow.right.circle.fill").foregroundStyle(.cyan)
                Text("Transition Time").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("Offense → Defense")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            HStack(alignment: .lastTextBaseline, spacing: 4) {
                Text(lastTTms.map { "\(Int($0))" } ?? "—")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .foregroundStyle(lastTTms != nil ? Color.cyan : Color.secondary)
                    .contentTransition(.numericText())
                if lastTTms != nil {
                    Text("ms").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let avg = averageTTms {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("avg").font(.caption2).foregroundStyle(.tertiary)
                        Text("\(Int(avg)) ms").font(.caption).fontWeight(.semibold)
                    }
                }
            }
        }
        .padding()
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
