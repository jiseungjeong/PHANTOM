import SwiftUI

struct LastEventBadge: View {
    let event: BoxingEvent

    private var color: Color {
        event.isPunch ? .blue : .purple
    }
    private var icon: String {
        event.isPunch ? "bolt.fill" : "shield.fill"
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon).foregroundStyle(color)
            Text(event.displayLabel)
                .fontWeight(.semibold)
            if let tt = event.transitionTimeMs {
                Spacer()
                Text("TT \(Int(tt)) ms")
                    .font(.caption)
                    .foregroundStyle(.cyan)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.cyan.opacity(0.15))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(color.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
