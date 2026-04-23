import SwiftUI

struct SessionToggleButton: View {
    let isActive: Bool
    let action:   () -> Void

    var body: some View {
        Button(action: action) {
            Text(isActive ? "Stop Session" : "Start Session")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(isActive ? Color.red : Color.blue)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}
