import Foundation

// MARK: - Raw Sensor Sample  (HARAPP-compatible format)
//
// Columns match HARApp's DataStorage CSV exactly, with two extra columns:
//   source — device that produced the sample ("iPhone" | "Watch" | "AirPods")
//   label  — activity label chosen by the user before recording
//
// accX/Y/Z  = userAcceleration + gravity  (total acceleration, g)
// gyroX/Y/Z = rotationRate               (rad/s)
// timestamp  = absolute Unix time        (seconds, like HARApp)

struct TimeSeriesRawSample: Identifiable, Codable {
    let id:        UUID
    let timestamp: Double  // Unix epoch (s) — matches HARApp column name
    let source:    String  // "iPhone" | "Watch" | "AirPods"
    let label:     String  // activity label
    let accX:      Double  // userAcceleration.x + gravity.x  (g)
    let accY:      Double  // userAcceleration.y + gravity.y  (g)
    let accZ:      Double  // userAcceleration.z + gravity.z  (g)
    let gyroX:     Double  // rotationRate.x  (rad/s)
    let gyroY:     Double  // rotationRate.y  (rad/s)
    let gyroZ:     Double  // rotationRate.z  (rad/s)

    init(timestamp: Double, source: String, label: String,
         accX: Double, accY: Double, accZ: Double,
         gyroX: Double, gyroY: Double, gyroZ: Double) {
        self.id        = UUID()
        self.timestamp = timestamp
        self.source    = source
        self.label     = label
        self.accX      = accX
        self.accY      = accY
        self.accZ      = accZ
        self.gyroX     = gyroX
        self.gyroY     = gyroY
        self.gyroZ     = gyroZ
    }
}

// MARK: - CSV Export  (column order matches HARApp + source column)

extension [TimeSeriesRawSample] {
    static let csvHeader = "timestamp,source,accX,accY,accZ,gyroX,gyroY,gyroZ,label"

    func csvString() -> String {
        let rows = self.map { s in
            [
                String(format: "%.6f",  s.timestamp),
                s.source,
                String(format: "%.8f",  s.accX),
                String(format: "%.8f",  s.accY),
                String(format: "%.8f",  s.accZ),
                String(format: "%.8f",  s.gyroX),
                String(format: "%.8f",  s.gyroY),
                String(format: "%.8f",  s.gyroZ),
                "\"\(s.label)\""
            ].joined(separator: ",")
        }
        return ([Self.csvHeader] + rows).joined(separator: "\n")
    }
}
