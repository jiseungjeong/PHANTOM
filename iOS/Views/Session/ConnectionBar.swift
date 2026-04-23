import SwiftUI

struct ConnectionBar: View {
    let watchConnected:   Bool
    let airPodsAvailable: Bool

    var body: some View {
        HStack(spacing: 20) {
            DeviceIndicator(label: "Watch",   icon: "applewatch",  connected: watchConnected)
            DeviceIndicator(label: "iPhone",  icon: "iphone",       connected: true)
            DeviceIndicator(label: "AirPods", icon: "airpodspro",   connected: airPodsAvailable)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct DeviceIndicator: View {
    let label:     String
    let icon:      String
    let connected: Bool

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.caption)
            Text(label).font(.caption2)
            Circle()
                .fill(connected ? Color.green : Color.gray.opacity(0.5))
                .frame(width: 6, height: 6)
        }
        .foregroundStyle(connected ? .primary : .secondary)
    }
}
