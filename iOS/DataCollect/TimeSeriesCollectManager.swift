import Foundation
import CoreMotion
import Combine

// MARK: - Time Series Collect Manager  (HARAPP-compatible)
//
// Records raw IMU samples continuously from all connected devices.
// Format matches HARApp's CSV: timestamp (Unix), accX/Y/Z (total), gyroX/Y/Z, label.
//
// Sources:
//   iPhone IMU  → CMMotionManager          @ 50 Hz  (same as HARApp)
//   Apple Watch → WCSession raw batches    @ 50 Hz, sent in 50-sample chunks
//   AirPods Pro → CMHeadphoneMotionManager @ 50 Hz
//
// Device-recognition fix: watchReachable + airPodsConnected are @Published
// so the view reacts to changes without observing WatchConnectivityManager directly.

@MainActor
final class TimeSeriesCollectManager: ObservableObject {

    // MARK: - Labels  (reuse same taxonomy as DataCollectManager)

    static let allLabels: [String] =
        PunchLabel.allCases.filter { $0 != .leadPunch && $0 != .rearPunch }.map(\.rawValue)
        + DefenseLabel.allCases.map(\.rawValue)
        + CombinationLabel.allCases.map(\.rawValue)

    // MARK: - Published State

    @Published var isRecording      = false
    @Published var selectedLabel    = PunchLabel.leadJab.rawValue
    @Published var participantId    = UserDefaults.standard.string(forKey: "phantomParticipantId") ?? "P00" {
        didSet { UserDefaults.standard.set(participantId, forKey: "phantomParticipantId") }
    }
    @Published var elapsedSeconds   = 0

    // Device counts — updated every sample batch
    @Published private(set) var iPhoneSampleCount  = 0
    @Published private(set) var watchSampleCount   = 0
    @Published private(set) var airPodsSampleCount = 0

    // Device availability — the view observes these instead of
    // reading WatchConnectivityManager.shared directly
    @Published private(set) var watchReachable    = false
    @Published private(set) var airPodsConnected  = false

    // Set after auto-save completes; cleared on next startRecording
    @Published private(set) var lastSavedURL: URL?

    // MARK: - Private

    private let motionManager   = CMMotionManager()
    private let headphoneMotion = CMHeadphoneMotionManager()
    private var anchorTs: TimeInterval = 0
    private var timer: Timer?
    private var watchSub: AnyCancellable?

    // Backing storage — separate per source for export flexibility
    private var iPhoneSamples:  [TimeSeriesRawSample] = []
    private var watchSamples:   [TimeSeriesRawSample] = []
    private var airPodsSamples: [TimeSeriesRawSample] = []

    // MARK: - Init

    init() {
        // Mirror WatchConnectivityManager's published reachability into our own
        // @Published so any view observing this manager sees updates.
        watchSub = WatchConnectivityManager.shared.$isWatchReachable
            .receive(on: RunLoop.main)
            .assign(to: \.watchReachable, on: self)
    }

    // MARK: - Recording Control

    func startRecording() {
        guard !isRecording else { return }
        anchorTs           = Date().timeIntervalSince1970
        iPhoneSamples      = []
        watchSamples       = []
        airPodsSamples     = []
        iPhoneSampleCount  = 0
        watchSampleCount   = 0
        airPodsSampleCount = 0
        elapsedSeconds     = 0
        lastSavedURL       = nil

        // Clear all chart buffers
        accXBuf = [];  accYBuf = [];  accZBuf = []
        gyroXBuf = []; gyroYBuf = []; gyroZBuf = []
        apAccXBuf = []; apAccYBuf = []; apAccZBuf = []
        apGyroXBuf = []; apGyroYBuf = []; apGyroZBuf = []
        waAccXBuf = []; waAccYBuf = []; waAccZBuf = []
        waGyroXBuf = []; waGyroYBuf = []; waGyroZBuf = []

        isRecording = true   // set before starting streams so callbacks don't hit a false guard

        startIPhoneStream()
        startAirPodsStream()
        startWatchStream()
        startTimer()
    }

    func stopRecording() {
        guard isRecording else { return }
        motionManager.stopDeviceMotionUpdates()
        headphoneMotion.stopDeviceMotionUpdates()
        WatchConnectivityManager.shared.sendStopRawStream()
        WatchConnectivityManager.shared.onRawWatchBatch = nil
        timer?.invalidate()
        timer          = nil
        airPodsConnected = false
        isRecording    = false

        // Auto-save — mirrors HARApp behavior: file is written immediately on stop
        lastSavedURL = try? saveCSV()
    }

    // MARK: - Export

    var allSamples: [TimeSeriesRawSample] {
        (iPhoneSamples + watchSamples + airPodsSamples)
            .sorted { $0.timestamp < $1.timestamp }
    }

    var totalSampleCount: Int { iPhoneSampleCount + watchSampleCount + airPodsSampleCount }

    enum ExportSource: String, CaseIterable {
        case all = "all", iPhone, Watch, AirPods
    }

    @discardableResult
    func saveCSV(source: ExportSource = .all) throws -> URL {
        let samples = samplesFor(source)
        let url = exportURL(source, ext: "csv")
        try samples.csvString().write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    @discardableResult
    func saveJSON(source: ExportSource = .all) throws -> URL {
        let samples = samplesFor(source)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = exportURL(source, ext: "json")
        try encoder.encode(samples).write(to: url)
        return url
    }

    private func exportURL(_ source: ExportSource, ext: String) -> URL {
        let filename = FileUtils.prefixedRawFilename(
            participantId: participantId,
            label: selectedLabel,
            source: source == .all ? nil : source.rawValue,
            ext: ext
        )
        return FileUtils.documentsURL.appendingPathComponent(filename)
    }

    // MARK: - Live Buffers for Charts (last 200 samples per device)

    // iPhone
    private(set) var accXBuf:  [Double] = []
    private(set) var accYBuf:  [Double] = []
    private(set) var accZBuf:  [Double] = []
    private(set) var gyroXBuf: [Double] = []
    private(set) var gyroYBuf: [Double] = []
    private(set) var gyroZBuf: [Double] = []

    // AirPods
    private(set) var apAccXBuf:  [Double] = []
    private(set) var apAccYBuf:  [Double] = []
    private(set) var apAccZBuf:  [Double] = []
    private(set) var apGyroXBuf: [Double] = []
    private(set) var apGyroYBuf: [Double] = []
    private(set) var apGyroZBuf: [Double] = []

    // Watch
    private(set) var waAccXBuf:  [Double] = []
    private(set) var waAccYBuf:  [Double] = []
    private(set) var waAccZBuf:  [Double] = []
    private(set) var waGyroXBuf: [Double] = []
    private(set) var waGyroYBuf: [Double] = []
    private(set) var waGyroZBuf: [Double] = []

    private let chartBufferSize = 200

    // MARK: - Private — iPhone Stream

    private func startIPhoneStream() {
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 50.0  // 50 Hz, matches HARApp
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, _ in
            guard let self, let motion, self.isRecording else { return }
            let ts   = Date().timeIntervalSince1970
            let ua   = motion.userAcceleration
            let grav = motion.gravity
            let gr   = motion.rotationRate
            let sample = TimeSeriesRawSample(
                timestamp: ts,
                source: "iPhone",
                label: self.selectedLabel,
                accX: ua.x + grav.x, accY: ua.y + grav.y, accZ: ua.z + grav.z,
                gyroX: gr.x, gyroY: gr.y, gyroZ: gr.z
            )
            self.iPhoneSamples.append(sample)
            self.iPhoneSampleCount += 1
            self.pushToChartBuffers(sample: sample)
        }
    }

    // MARK: - Private — AirPods Stream

    private func startAirPodsStream() {
        AudioSessionActivator.activateForHeadphoneMotion()

        // Do NOT gate on isDeviceMotionAvailable — it can return false even when AirPods
        // are worn because the BT handshake or privacy auth may not have completed yet.
        // Always start updates and rely solely on the callback result.
        headphoneMotion.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            // No isRecording guard here — CMHeadphoneMotionManager can fire its first
            // handshake callback before the run loop returns to set isRecording = true,
            // which would block AirPods from ever being recognised.
            // stopDeviceMotionUpdates() in stopRecording() prevents callbacks after stop.
            guard let self else { return }
            if let error {
                print("[TimeSeries] AirPods error: \(error.localizedDescription)")
                self.airPodsConnected = false
                return
            }
            guard let motion else { return }
            if !self.airPodsConnected { self.airPodsConnected = true }
            guard self.isRecording else { return }   // don't append after stop
            let ts   = Date().timeIntervalSince1970
            let ua   = motion.userAcceleration
            let grav = motion.gravity
            let gr   = motion.rotationRate
            let sample = TimeSeriesRawSample(
                timestamp: ts,
                source: "AirPods",
                label: self.selectedLabel,
                accX: ua.x + grav.x, accY: ua.y + grav.y, accZ: ua.z + grav.z,
                gyroX: gr.x, gyroY: gr.y, gyroZ: gr.z
            )
            self.airPodsSamples.append(sample)
            self.airPodsSampleCount += 1
            self.pushToAirPodsChartBuffers(sample: sample)
        }
    }

    // MARK: - Private — Watch Stream

    private func startWatchStream() {
        WatchConnectivityManager.shared.onRawWatchBatch = { [weak self] flat in
            guard let self, self.isRecording else { return }
            // Decode stride-7: [timestamp, totalAccX, totalAccY, totalAccZ, gyroX, gyroY, gyroZ, ...]
            var i = 0
            while i + 6 < flat.count {
                let sample = TimeSeriesRawSample(
                    timestamp: flat[i],
                    source: "Watch",
                    label: self.selectedLabel,
                    accX: flat[i+1], accY: flat[i+2], accZ: flat[i+3],
                    gyroX: flat[i+4], gyroY: flat[i+5], gyroZ: flat[i+6]
                )
                self.watchSamples.append(sample)
                self.watchSampleCount += 1
                self.pushToWatchChartBuffers(sample: sample)
                i += 7
            }
        }
        WatchConnectivityManager.shared.sendStartRawStream(anchorTs: anchorTs,
                                                           label: selectedLabel)

        // The Watch will deliver an authoritative per-sample CSV via transferFile
        // on stop. The streamed batches above are best-effort live preview only;
        // the file is the source of truth and lands in Documents directly,
        // showing up in FilesView automatically.
        WatchConnectivityManager.shared.onWatchRawFileSaved = { [weak self] url in
            guard let self else { return }
            let finalURL = self.renameWatchRawFileIfNeeded(url)
            print("[TimeSeries] received Watch raw file: \(finalURL.lastPathComponent) "
                  + "(recording=\(self.isRecording ? "active" : "stopped"))")
        }
    }

    private func renameWatchRawFileIfNeeded(_ url: URL) -> URL {
        let basename = url.deletingPathExtension().lastPathComponent
        let participant = FileUtils.normalizedParticipantId(participantId)
        if basename.hasPrefix("\(participant)_") {
            return url
        }

        let filename = "\(participant)_\(url.lastPathComponent)"
        let dest = FileUtils.documentsURL.appendingPathComponent(filename)

        do {
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.moveItem(at: url, to: dest)
            return dest
        } catch {
            print("[TimeSeries] failed to rename Watch raw file: \(error)")
            return url
        }
    }

    // MARK: - Private — Chart Buffers

    private func push(_ val: Double, into buf: inout [Double]) {
        buf.append(val)
        if buf.count > chartBufferSize { buf.removeFirst() }
    }

    private func pushToChartBuffers(sample: TimeSeriesRawSample) {
        push(sample.accX,  into: &accXBuf)
        push(sample.accY,  into: &accYBuf)
        push(sample.accZ,  into: &accZBuf)
        push(sample.gyroX, into: &gyroXBuf)
        push(sample.gyroY, into: &gyroYBuf)
        push(sample.gyroZ, into: &gyroZBuf)
    }

    private func pushToAirPodsChartBuffers(sample: TimeSeriesRawSample) {
        push(sample.accX,  into: &apAccXBuf)
        push(sample.accY,  into: &apAccYBuf)
        push(sample.accZ,  into: &apAccZBuf)
        push(sample.gyroX, into: &apGyroXBuf)
        push(sample.gyroY, into: &apGyroYBuf)
        push(sample.gyroZ, into: &apGyroZBuf)
    }

    private func pushToWatchChartBuffers(sample: TimeSeriesRawSample) {
        push(sample.accX,  into: &waAccXBuf)
        push(sample.accY,  into: &waAccYBuf)
        push(sample.accZ,  into: &waAccZBuf)
        push(sample.gyroX, into: &waGyroXBuf)
        push(sample.gyroY, into: &waGyroYBuf)
        push(sample.gyroZ, into: &waGyroZBuf)
    }

    // MARK: - Private — Timer

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.elapsedSeconds += 1
            }
        }
    }

    // MARK: - Helpers

    private func samplesFor(_ source: ExportSource) -> [TimeSeriesRawSample] {
        switch source {
        case .all:     return allSamples
        case .iPhone:  return iPhoneSamples
        case .Watch:   return watchSamples
        case .AirPods: return airPodsSamples
        }
    }

}
