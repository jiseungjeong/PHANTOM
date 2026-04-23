import Foundation
import CoreMotion
import AVFoundation

// MARK: - Data Collect Manager
//
// Standalone recording manager for building labeled IMU datasets.
// Runs independently from BoxingSessionManager — the user selects
// a gesture label, taps Record, performs the gesture, and each
// detected spike / gesture window is saved as a DataCollectSample.
//
// Sources:
//   iPhone IMU  → own CMMotionManager + SpikeDetector (100 Hz)       → punch labels
//   Apple Watch → WatchConnectivityManager.onWristSpikeCollect        → punch labels
//   AirPods Pro → own CMHeadphoneMotionManager + HeadGestureClassifier → defense labels
//
// Data collection and live session must not run simultaneously.

@MainActor
final class DataCollectManager: ObservableObject {

    // MARK: - Published State

    @Published var isRecording    = false
    @Published var selectedLabel  = PunchLabel.leadJab.rawValue
    @Published private(set) var samples: [DataCollectSample] = []

    // MARK: - Available Labels (for picker)

    static let punchLabels: [String] = PunchLabel.allCases
        .filter { $0 != .leadPunch && $0 != .rearPunch }
        .map(\.rawValue)

    static let defenseLabels: [String]     = DefenseLabel.allCases.map(\.rawValue)
    static let combinationLabels: [String] = CombinationLabel.allCases.map(\.rawValue)

    // MARK: - Private — Wrist (iPhone + Watch)

    private let motionManager = CMMotionManager()
    private let detector      = SpikeDetector()

    // MARK: - Private — Head (AirPods)

    private let headphoneMotion       = CMHeadphoneMotionManager()
    private let headGestureClassifier = HeadGestureClassifier()

    @Published private(set) var airPodsAvailable: Bool = false

    private var anchorTs: TimeInterval = 0

    // MARK: - Init

    init() {
        // iPhone wrist spike → sample
        detector.onSpike = { [weak self] window, tsMs in
            guard let self, self.isRecording else { return }
            self.record(label: self.selectedLabel, source: "iPhone", window: window, tsMs: tsMs)
        }

        // Watch wrist spike → sample
        WatchConnectivityManager.shared.onWristSpikeCollect = { [weak self] window, tsMs in
            guard let self, self.isRecording else { return }
            self.record(label: self.selectedLabel, source: "Watch", window: window, tsMs: tsMs)
        }

        // AirPods defense gesture → sample (uses selected defense label, ignores auto-classified label
        // so the user's manual label choice always wins — useful for collecting mislabeled edge cases)
        headGestureClassifier.onGestureDetectedWithWindow = { [weak self] _, gyroWindow, tsMs in
            guard let self, self.isRecording else { return }
            self.record(label: self.selectedLabel, source: "AirPods", window: gyroWindow, tsMs: tsMs)
        }
    }

    // MARK: - Recording Control

    func startRecording() {
        guard !isRecording else { return }
        anchorTs = Date().timeIntervalSince1970

        // Apple Watch — start recording and align its clock to the same anchor
        WatchConnectivityManager.shared.sendStartSession(anchorTs: anchorTs)

        // iPhone IMU
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
                guard let self, let motion else { return }
                let mag  = accelerationMagnitude(x: motion.userAcceleration.x,
                                                 y: motion.userAcceleration.y,
                                                 z: motion.userAcceleration.z)
                let tsMs = (Date().timeIntervalSince1970 - self.anchorTs) * 1000
                self.detector.processSample(magnitude: mag, tsMs: tsMs)
            }
        }

        // AirPods — activate AVAudioSession first so CMHeadphoneMotionManager can
        // establish the Bluetooth route.  Required for AirPods Pro, AirPods Max,
        // AirPods 3rd gen, and AirPods 4 (with ANC); without this the callback
        // silently produces no updates even when the AirPods are worn.
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .default,
                                      options: [.allowBluetoothHFP, .allowBluetoothA2DP])
        try? audioSession.setActive(true)

        headphoneMotion.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                print("[DataCollect] AirPods motion error: \(error.localizedDescription)")
                self.airPodsAvailable = false
                return
            }
            guard let motion else { return }
            self.airPodsAvailable = true
            let tsMs = (Date().timeIntervalSince1970 - self.anchorTs) * 1000
            self.headGestureClassifier.processSample(
                accX:  motion.userAcceleration.x,
                gyroY: motion.rotationRate.y,
                tsMs:  tsMs
            )
        }

        isRecording = true
    }

    func stopRecording() {
        guard isRecording else { return }
        motionManager.stopDeviceMotionUpdates()
        headphoneMotion.stopDeviceMotionUpdates()
        detector.reset()
        headGestureClassifier.reset()
        airPodsAvailable = false
        isRecording = false

        // Apple Watch — stop recording
        WatchConnectivityManager.shared.sendStopSession()
    }

    // MARK: - Sample Management

    func delete(_ sample: DataCollectSample) {
        samples.removeAll { $0.id == sample.id }
    }

    func clearAll() {
        samples.removeAll()
    }

    // MARK: - Export (save to app Documents folder)

    @discardableResult
    func saveCSV() throws -> URL {
        let url = documentsURL.appendingPathComponent("phantom_data_\(dateStamp()).csv")
        try csvString().write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    @discardableResult
    func saveJSON() throws -> URL {
        let url = documentsURL.appendingPathComponent("phantom_data_\(dateStamp()).json")
        try jsonData().write(to: url)
        return url
    }

    func csvString() -> String { samples.csvString() }

    func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(samples)
    }

    // MARK: - Summary

    var punchCount:       Int { samples.filter { DataCollectManager.punchLabels.contains($0.label) }.count }
    var defenseCount:     Int { samples.filter { DataCollectManager.defenseLabels.contains($0.label) }.count }
    var combinationCount: Int { samples.filter { DataCollectManager.combinationLabels.contains($0.label) }.count }
    var watchCount:       Int { samples.filter { $0.source == "Watch" }.count }
    var iPhoneCount:      Int { samples.filter { $0.source == "iPhone" }.count }
    var airPodsCount:     Int { samples.filter { $0.source == "AirPods" }.count }

    // MARK: - Private

    private func record(label: String, source: String, window: [Double], tsMs: Double) {
        samples.append(DataCollectSample(label: label, source: source, window: window, tsMs: tsMs))
    }

    private var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private func dateStamp() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: Date())
    }
}
