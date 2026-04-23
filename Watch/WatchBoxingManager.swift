import Foundation
import CoreMotion
import WatchConnectivity
import WatchKit
import HealthKit

// MARK: - Watch Boxing Manager  (PHANTOM architecture)
//
// Lead hand wrist sensor. Responsibilities:
//   • Sample wrist motion at 100 Hz via CMMotionManager
//   • Detect acceleration spikes above threshold (SpikeDetector)
//   • Forward raw spike window + session-relative timestamp to iPhone
//     via WCSession — classification is done on iPhone by PHANTOMFusionEngine
//   • Deliver haptic feedback on each spike
//   • Receive session start (with anchor timestamp) / stop from iPhone
//
// The Watch does NOT classify punches locally — it sends raw windows to
// avoid stale heuristics and keep all DTW logic in one place (iPhone).
//
// An HKWorkoutSession is held for the duration of any active session so
// that CoreMotion updates keep flowing when the wrist drops or the screen
// dims — without it, watchOS suspends the app within seconds.

@MainActor
final class WatchBoxingManager: NSObject, ObservableObject {

    private let motionManager = CMMotionManager()
    private let detector      = SpikeDetector()
    private let updateInterval: TimeInterval = 1.0 / 100.0

    // MARK: - Published State

    @Published var isSessionActive  = false
    @Published var isWorkoutActive  = false  // true while HKWorkoutSession is running
    @Published var spikeCount       = 0      // local counter for Watch UI
    @Published var isPhoneReachable = false

    // MARK: - Sync

    private var anchorTs: TimeInterval = 0  // received from iPhone at session start
    private var watchInitiatedSession  = false  // true when session started from Watch UI

    // MARK: - Raw Streaming

    private var isRawStreaming   = false
    private var rawBuffer:  [Double] = []   // stride-7: tsMs,ax,ay,az,gx,gy,gz per sample
    private let rawBatchSize     = 50       // flush every 50 samples (~500 ms at 100 Hz)

    // Authoritative full buffer — every sample collected during raw streaming.
    // Streamed batches are best-effort for live preview only; this buffer is what
    // gets written to a tmp CSV and shipped to iPhone via transferFile on stop.
    private var rawFullBuffer: [Double] = []
    private var currentLabel: String    = ""

    // MARK: - Background keep-alive

    private let healthStore = HKHealthStore()
    private var workoutSession: HKWorkoutSession?
    private var workoutBuilder: HKLiveWorkoutBuilder?

    // MARK: - Persistence keys

    private enum PersistKey {
        static let sessionActive = "phantom.watch.sessionActive"
        static let anchorTs      = "phantom.watch.anchorTs"
        static let spikeCount    = "phantom.watch.spikeCount"
    }

    // MARK: - Init

    override init() {
        super.init()
        activateWCSession()

        detector.onSpike = { [weak self] window, tsMs in
            guard let self else { return }
            self.spikeCount += 1
            UserDefaults.standard.set(self.spikeCount, forKey: PersistKey.spikeCount)
            WKInterfaceDevice.current().play(.click)
            self.sendSpikeToPhone(window: window, tsMs: tsMs)
        }

        restoreSessionIfNeeded()
    }

    /// If the app was killed while a session was active, resume it automatically
    /// using the persisted anchor timestamp and spike counter so the user lands
    /// back in a live session instead of a blank screen.
    private func restoreSessionIfNeeded() {
        let ud = UserDefaults.standard
        guard ud.bool(forKey: PersistKey.sessionActive) else { return }
        let persistedAnchor = ud.double(forKey: PersistKey.anchorTs)
        guard persistedAnchor > 0 else { return }

        spikeCount = ud.integer(forKey: PersistKey.spikeCount)
        startSession(anchorTs: persistedAnchor, notifyPhone: false, resuming: true)
    }

    // MARK: - Session Control

    func startSession(anchorTs: TimeInterval,
                      notifyPhone: Bool = false,
                      resuming: Bool = false) {
        self.anchorTs = anchorTs
        guard !isSessionActive, motionManager.isDeviceMotionAvailable else { return }

        beginWorkoutKeepAlive()

        if !resuming { spikeCount = 0 }
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let mag  = accelerationMagnitude(x: motion.userAcceleration.x,
                                             y: motion.userAcceleration.y,
                                             z: motion.userAcceleration.z)
            let tsMs = (Date().timeIntervalSince1970 - self.anchorTs) * 1000
            self.detector.processSample(magnitude: mag, tsMs: tsMs)
        }
        isSessionActive = true

        let ud = UserDefaults.standard
        ud.set(true,     forKey: PersistKey.sessionActive)
        ud.set(anchorTs, forKey: PersistKey.anchorTs)
        ud.set(spikeCount, forKey: PersistKey.spikeCount)

        if notifyPhone {
            sendToPhone([
                WatchMessageKey.action:   WatchMessageKey.watchSessionStart,
                WatchMessageKey.anchorTs: anchorTs
            ])
        }
    }

    func stopSession(notifyPhone: Bool = false) {
        guard isSessionActive else { return }
        motionManager.stopDeviceMotionUpdates()
        detector.reset()
        isSessionActive = false
        watchInitiatedSession = false

        let ud = UserDefaults.standard
        ud.set(false, forKey: PersistKey.sessionActive)
        ud.removeObject(forKey: PersistKey.anchorTs)
        ud.removeObject(forKey: PersistKey.spikeCount)

        endWorkoutKeepAliveIfIdle()

        if notifyPhone {
            sendToPhone([WatchMessageKey.action: WatchMessageKey.watchSessionStop])
        }
    }

    // MARK: - Raw Streaming Control

    func startRawStream(anchorTs: TimeInterval, label: String = "") {
        guard !isRawStreaming, motionManager.isDeviceMotionAvailable else { return }
        self.anchorTs     = anchorTs
        self.currentLabel = label
        isRawStreaming    = true
        rawBuffer         = []
        rawFullBuffer     = []

        beginWorkoutKeepAlive()

        motionManager.deviceMotionUpdateInterval = 1.0 / 50.0  // 50 Hz, matches HARApp
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            // Use absolute Unix timestamp — aligns with HARApp CSV format
            let ts = Date().timeIntervalSince1970
            // Total acceleration = userAcceleration + gravity (same as HARApp)
            let sample: [Double] = [
                ts,
                motion.userAcceleration.x + motion.gravity.x,
                motion.userAcceleration.y + motion.gravity.y,
                motion.userAcceleration.z + motion.gravity.z,
                motion.rotationRate.x,
                motion.rotationRate.y,
                motion.rotationRate.z
            ]
            self.rawBuffer.append(contentsOf: sample)
            self.rawFullBuffer.append(contentsOf: sample)
            if self.rawBuffer.count >= self.rawBatchSize * 7 {
                self.flushRawBuffer()
            }
        }
    }

    func stopRawStream() {
        guard isRawStreaming else { return }
        if rawBuffer.count >= 7 { flushRawBuffer() }
        motionManager.stopDeviceMotionUpdates()
        rawBuffer      = []
        isRawStreaming = false

        // Write authoritative CSV and ship via guaranteed-delivery file transfer.
        // This is the path that survives WCSession sendMessage throttling.
        transferFullBufferToPhone()
        rawFullBuffer = []

        endWorkoutKeepAliveIfIdle()
    }

    /// Live-preview flush. Best-effort only: if the phone is not reachable
    /// (e.g. wrist down, screen off) we drop the batch instead of queueing it
    /// via transferUserInfo. The authoritative full buffer is shipped via
    /// transferFile at stop, so dropping live batches never loses data.
    private func flushRawBuffer() {
        guard !rawBuffer.isEmpty else { return }
        let payload = rawBuffer
        rawBuffer = []
        guard WCSession.isSupported(), WCSession.default.isReachable else { return }
        WCSession.default.sendMessage(
            [WatchMessageKey.rawBatch: true,
             WatchMessageKey.rawSamples: payload],
            replyHandler: nil,
            errorHandler: { error in
                print("[Watch→Phone] live batch send error: \(error)")
            }
        )
    }

    // MARK: - Reliable raw delivery (transferFile)

    /// Writes the full per-sample buffer to a CSV file in tmp and queues it
    /// for delivery via WCSession.transferFile. transferFile is a queued,
    /// guaranteed-delivery channel — unlike sendMessage it survives throttling
    /// and app suspension.
    private func transferFullBufferToPhone() {
        let buffer = rawFullBuffer
        guard buffer.count >= 7 else { return }
        guard WCSession.isSupported() else { return }

        let safeLabel = currentLabel.isEmpty ? "raw"
            : currentLabel.replacingOccurrences(of: " ", with: "_")
        let basename = "\(safeLabel)_Watch_\(FileUtils.dateStamp()).csv"

        let url = FileManager.default.temporaryDirectory.appendingPathComponent(basename)
        do {
            try makeCSVString(from: buffer, label: currentLabel)
                .write(to: url, atomically: true, encoding: .utf8)
        } catch {
            print("[Watch] failed to write raw CSV: \(error)")
            return
        }

        let metadata: [String: Any] = [
            WatchMessageKey.fileTag:  WatchMessageKey.rawWatchFile,
            WatchMessageKey.label:    currentLabel,
            "basename":               basename,
            "sampleCount":            buffer.count / 7
        ]
        WCSession.default.transferFile(url, metadata: metadata)
        print("[Watch] queued transferFile: \(basename) samples=\(buffer.count / 7)")
    }

    /// Builds a HARApp-compatible CSV from a stride-7 flat buffer.
    /// Column order matches iOS [TimeSeriesRawSample.csvHeader].
    private func makeCSVString(from buffer: [Double], label: String) -> String {
        var lines: [String] = ["timestamp,source,accX,accY,accZ,gyroX,gyroY,gyroZ,label"]
        lines.reserveCapacity(buffer.count / 7 + 1)
        let escapedLabel = "\"\(label)\""
        var i = 0
        while i + 6 < buffer.count {
            let row = String(
                format: "%.6f,Watch,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%@",
                buffer[i],
                buffer[i + 1], buffer[i + 2], buffer[i + 3],
                buffer[i + 4], buffer[i + 5], buffer[i + 6],
                escapedLabel
            )
            lines.append(row)
            i += 7
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Workout keep-alive

    private func beginWorkoutKeepAlive() {
        guard workoutSession == nil else { return }
        guard HKHealthStore.isHealthDataAvailable() else { return }

        let typesToShare: Set = [HKObjectType.workoutType()]
        healthStore.requestAuthorization(toShare: typesToShare, read: []) { [weak self] granted, error in
            Task { @MainActor in
                guard let self else { return }
                if let error {
                    print("[Watch] HK auth error: \(error)")
                }
                guard granted else {
                    print("[Watch] HK auth not granted — Watch will suspend when wrist drops")
                    return
                }
                self.startWorkoutSessionNow()
            }
        }
    }

    @MainActor
    private func startWorkoutSessionNow() {
        guard workoutSession == nil else { return }

        let config = HKWorkoutConfiguration()
        config.activityType  = .boxing
        config.locationType  = .indoor

        do {
            let session = try HKWorkoutSession(healthStore: healthStore, configuration: config)
            let builder = session.associatedWorkoutBuilder()
            builder.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore,
                                                         workoutConfiguration: config)
            session.delegate = self
            builder.delegate = self
            self.workoutSession  = session
            self.workoutBuilder  = builder

            let startDate = Date()
            session.startActivity(with: startDate)
            builder.beginCollection(withStart: startDate) { ok, err in
                if let err { print("[Watch] beginCollection error: \(err)") }
                if !ok   { print("[Watch] beginCollection returned not-ok") }
            }
            isWorkoutActive = true
            print("[Watch] HKWorkoutSession started — keep-alive active")
        } catch {
            print("[Watch] HKWorkoutSession start failed: \(error)")
            workoutSession = nil
            workoutBuilder = nil
            isWorkoutActive = false
        }
    }

    private func endWorkoutKeepAliveIfIdle() {
        guard !isSessionActive, !isRawStreaming else { return }
        guard let session = workoutSession else { return }
        session.end()
        workoutBuilder?.endCollection(withEnd: Date()) { [weak self] _, _ in
            self?.workoutBuilder?.finishWorkout { _, _ in }
        }
        workoutSession = nil
        isWorkoutActive = false
        workoutBuilder = nil
    }

    // MARK: - WCSession

    private func activateWCSession() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    private func sendSpikeToPhone(window: [Double], tsMs: Double) {
        sendToPhone([
            WatchMessageKey.wristSpike: true,
            WatchMessageKey.window:     window,
            WatchMessageKey.tsMs:       tsMs
        ])
    }

    /// Sends a payload that must be delivered eventually (session lifecycle,
    /// spike events). Prefers live sendMessage when reachable, falls back to
    /// transferUserInfo queue when the phone is not reachable so nothing is
    /// lost across wrist-down / phone-out-of-range intervals.
    private func sendToPhone(_ payload: [String: Any]) {
        guard WCSession.isSupported() else { return }
        if WCSession.default.isReachable {
            WCSession.default.sendMessage(payload, replyHandler: nil) { [weak self] error in
                print("[Watch→Phone] send error, falling back to userInfo: \(error)")
                self?.enqueueUserInfo(payload)
            }
        } else {
            enqueueUserInfo(payload)
        }
    }

    private nonisolated func enqueueUserInfo(_ payload: [String: Any]) {
        guard WCSession.isSupported() else { return }
        WCSession.default.transferUserInfo(payload)
    }
}

// MARK: - WCSessionDelegate  (Watch side)

extension WatchBoxingManager: WCSessionDelegate {

    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith state: WCSessionActivationState,
        error: Error?
    ) {
        Task { @MainActor in
            self.isPhoneReachable = (state == .activated) && session.isReachable
        }
    }

    nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
        Task { @MainActor in self.isPhoneReachable = session.isReachable }
    }

    nonisolated func session(_ session: WCSession,
                             didReceiveMessage message: [String: Any]) {
        Task { @MainActor in self.handleCommand(message) }
    }

    nonisolated func session(_ session: WCSession,
                             didReceiveUserInfo userInfo: [String: Any]) {
        Task { @MainActor in self.handleCommand(userInfo) }
    }

    /// Delete the tmp CSV after the system has finished delivering it.
    nonisolated func session(_ session: WCSession,
                             didFinish fileTransfer: WCSessionFileTransfer,
                             error: Error?) {
        if let error {
            print("[Watch] file transfer failed: \(error)")
        }
        try? FileManager.default.removeItem(at: fileTransfer.file.fileURL)
    }

    @MainActor
    private func handleCommand(_ msg: [String: Any]) {
        guard let action = msg[WatchMessageKey.action] as? String else { return }
        switch action {
        case WatchMessageKey.sessionStart:
            let anchor = msg[WatchMessageKey.anchorTs] as? TimeInterval ?? Date().timeIntervalSince1970
            startSession(anchorTs: anchor)
        case WatchMessageKey.sessionStop:
            stopSession()
        case WatchMessageKey.rawStreamStart:
            let anchor = msg[WatchMessageKey.anchorTs] as? TimeInterval ?? Date().timeIntervalSince1970
            let label  = msg[WatchMessageKey.label] as? String ?? ""
            startRawStream(anchorTs: anchor, label: label)
        case WatchMessageKey.rawStreamStop:
            stopRawStream()
        default:
            break
        }
    }
}

// MARK: - HKWorkoutSessionDelegate / HKLiveWorkoutBuilderDelegate

extension WatchBoxingManager: HKWorkoutSessionDelegate, HKLiveWorkoutBuilderDelegate {
    nonisolated func workoutSession(_ workoutSession: HKWorkoutSession,
                                    didChangeTo toState: HKWorkoutSessionState,
                                    from fromState: HKWorkoutSessionState,
                                    date: Date) {
        print("[Watch] HKWorkoutSession state: \(fromState.rawValue) → \(toState.rawValue)")
    }

    nonisolated func workoutSession(_ workoutSession: HKWorkoutSession,
                                    didFailWithError error: Error) {
        print("[Watch] HKWorkoutSession failed: \(error)")
    }

    nonisolated func workoutBuilderDidCollectEvent(_ workoutBuilder: HKLiveWorkoutBuilder) {}

    nonisolated func workoutBuilder(_ workoutBuilder: HKLiveWorkoutBuilder,
                                    didCollectDataOf collectedTypes: Set<HKSampleType>) {}
}
