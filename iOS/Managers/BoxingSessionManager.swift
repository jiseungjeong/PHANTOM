import Foundation
import Combine

// MARK: - Boxing Session Manager
//
// Orchestrates the three sensor streams and the PHANTOMFusionEngine.
// This is the @MainActor ViewModel that SwiftUI views observe.
//
// Device roles (PHANTOM architecture):
//   Apple Watch  — Lead hand wrist spike detector → raw windows → iPhone
//   iPhone IMU   — Rear hand wrist spike detector (always active, not a fallback)
//   AirPods Pro  — Head gesture stream → PHANTOMFusionEngine

@MainActor
final class BoxingSessionManager: ObservableObject {

    // MARK: - Recording Mode

    /// How a session is recorded and reported.
    ///   liveCount — the existing real-time fusion: Lead/Rear punch counts,
    ///               defense gestures, transition time.
    ///   ruleBased — accumulate raw IMU for the whole session, then classify
    ///               it as a single motion label on stop (rule_base_classifier).
    enum SessionMode: String, CaseIterable, Identifiable {
        case liveCount = "Live Count"
        case ruleBased = "Rule-based"
        var id: String { rawValue }
    }

    // MARK: - Dependencies

    let fusionEngine       = PHANTOMFusionEngine()
    private let phoneMotion       = PhoneMotionManager()
    private let headphoneMotion   = HeadphoneMotionManager()
    private let watchConnectivity = WatchConnectivityManager.shared

    /// Reused for the rule-based mode's raw accumulation. We stop it without
    /// auto-saving a CSV — the samples are classified in memory.
    private let rawCollector = TimeSeriesCollectManager()

    /// nil when rules.json is missing or fails to decode; the rule-based mode
    /// is then disabled in the UI.
    let classifier: MotionRuleClassifier?

    // MARK: - Published State (forwarded from fusionEngine)

    @Published var isSessionActive  = false
    @Published var showSafetyAlert  = false     // first-launch warning

    @Published var isWatchConnected   = false
    @Published var isAirPodsAvailable = false

    // MARK: - Rule-based Mode State

    @Published var mode: SessionMode = .liveCount
    /// Populated after a rule-based session stops and classification finishes.
    @Published var classification: MotionRuleClassifier.Result?
    @Published var isClassifying = false

    /// Whether the rule-based mode can be offered (classifier loaded ok).
    var isRuleBasedAvailable: Bool { classifier != nil }

    // MARK: - Live-session single-motion sequence (Feature D)

    /// Ordered single motions extracted from the most recent live-count session.
    /// Current-session state only — not persisted. Shown and cleared in Stats.
    @Published var motionSequence: DetectedMotionSequence?
    @Published var isSegmenting = false

    /// Current-session Defense count, derived from the segmented single-motion
    /// sequence (Feature E): slips, rolls, step, guard — punches excluded. nil
    /// while a session is in progress (sequence not yet computed).
    var detectedDefenseCount: Int? { motionSequence?.defenseCount }

    /// Raw IMU accumulated during a live-count session (all sources), classified
    /// into per-event segments on stop. Only filled when the classifier loaded.
    private var liveRawSamples: [TimeSeriesRawSample] = []
    private var liveSessionAnchor: TimeInterval = 0

    /// Per-event segmentation window, in seconds, around each event timestamp.
    private let segmentPreS  = 0.3
    private let segmentPostS = 0.7

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        classifier = try? MotionRuleClassifier.load()

        wireCallbacks()
        bindPublished()

        // Show safety alert on first run
        if !UserDefaults.standard.bool(forKey: "phantomSafetyAlertShown") {
            showSafetyAlert = true
        }
    }

    // MARK: - Session Control

    func startSession() {
        guard !isSessionActive else { return }

        switch mode {
        case .liveCount:
            startLiveCountSession()
        case .ruleBased:
            startRuleBasedSession()
        }

        isSessionActive = true
    }

    func stopSession() {
        guard isSessionActive else { return }

        switch mode {
        case .liveCount:
            stopLiveCountSession()
        case .ruleBased:
            stopRuleBasedSession()
        }

        isSessionActive = false
    }

    // MARK: - Live Count Mode

    private func startLiveCountSession() {
        fusionEngine.startSession()
        let anchor = fusionEngine.anchorTs

        beginLiveRawCapture(anchor: anchor)
        phoneMotion.startTracking(anchorTs: anchor)
        headphoneMotion.startTracking(anchorTs: anchor)
        watchConnectivity.sendStartSession(anchorTs: anchor, collectRaw: classifier != nil)
    }

    private func stopLiveCountSession() {
        fusionEngine.stopSession()

        phoneMotion.stopTracking()
        headphoneMotion.stopTracking()
        watchConnectivity.sendStopSession()

        endLiveRawCaptureAndSegment()
    }

    // MARK: - Live-session raw capture + segmentation (Feature D)

    /// Begin accumulating raw IMU from all sources during a live-count session.
    /// No-op when the classifier failed to load (no point collecting).
    private func beginLiveRawCapture(anchor: TimeInterval) {
        motionSequence = nil
        isSegmenting = false
        guard classifier != nil else { return }

        liveRawSamples = []
        liveRawSamples.reserveCapacity(8192)
        liveSessionAnchor = anchor

        phoneMotion.onRawSample = { [weak self] sample in
            self?.liveRawSamples.append(sample)
        }
        headphoneMotion.onRawSample = { [weak self] sample in
            self?.liveRawSamples.append(sample)
        }
        // Watch ships raw in stride-7 batches via the same channel the Raw
        // Stream tab uses. Safe to claim here: live-count and rule-based
        // sessions are mutually exclusive.
        watchConnectivity.onRawWatchBatch = { [weak self] flat in
            guard let self else { return }
            var i = 0
            while i + 6 < flat.count {
                self.liveRawSamples.append(TimeSeriesRawSample(
                    timestamp: flat[i], source: "Watch", label: "",
                    accX: flat[i + 1], accY: flat[i + 2], accZ: flat[i + 3],
                    gyroX: flat[i + 4], gyroY: flat[i + 5], gyroZ: flat[i + 6]
                ))
                i += 7
            }
        }
    }

    /// On stop, segment the session around each detected event and classify
    /// each window into a single primitive motion.
    private func endLiveRawCaptureAndSegment() {
        phoneMotion.onRawSample = nil
        headphoneMotion.onRawSample = nil
        watchConnectivity.onRawWatchBatch = nil

        guard let classifier else { return }
        let samples = liveRawSamples
        liveRawSamples = []

        // Event centers as absolute Unix timestamps (events carry session-
        // relative ms; raw carries absolute Unix seconds).
        let anchor = liveSessionAnchor
        let eventCenters: [(tsMs: Double, absTs: TimeInterval)] =
            fusionEngine.stats.events
                .map { ($0.tsMs, anchor + $0.tsMs / 1000.0) }
                .sorted { $0.absTs < $1.absTs }

        guard !eventCenters.isEmpty, !samples.isEmpty else {
            motionSequence = DetectedMotionSequence(items: [])
            return
        }

        let pre = segmentPreS, post = segmentPostS
        isSegmenting = true
        Task.detached(priority: .userInitiated) { [weak self] in
            var items: [DetectedMotionSequence.Item] = []
            for event in eventCenters {
                let lo = event.absTs - pre
                let hi = event.absTs + post
                let window = samples.filter { $0.timestamp >= lo && $0.timestamp <= hi }
                if let result = classifier.classifySegment(samples: window) {
                    items.append(DetectedMotionSequence.Item(
                        label: result.label, tsMs: event.tsMs, confidence: result.confidence))
                }
            }
            await MainActor.run {
                guard let self else { return }
                self.motionSequence = DetectedMotionSequence(items: items)
                self.isSegmenting = false
            }
        }
    }

    /// Clear the detected-motion sequence independently of cumulative records.
    func resetMotionSequence() {
        motionSequence = nil
    }

    // MARK: - Rule-based Mode

    private func startRuleBasedSession() {
        classification = nil
        isClassifying = false
        // rawCollector drives the iPhone IMU, AirPods, and Watch raw streams at
        // 50 Hz, identical to the Raw Stream tab.
        rawCollector.startRecording()
    }

    private func stopRuleBasedSession() {
        // Stop without auto-saving a CSV; classify the in-memory samples.
        rawCollector.stopRecording(autoSave: false)
        let samples = rawCollector.allSamples

        guard let classifier else { return }
        isClassifying = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let result = classifier.classify(samples: samples)
            await MainActor.run {
                guard let self else { return }
                self.classification = result
                self.isClassifying = false
            }
        }
    }

    func dismissSafetyAlert() {
        UserDefaults.standard.set(true, forKey: "phantomSafetyAlertShown")
        showSafetyAlert = false
    }

    // MARK: - Wiring

    private func wireCallbacks() {
        // Watch spike windows → Lead hand punches
        watchConnectivity.onWristSpike = { [weak self] window, tsMs in
            self?.fusionEngine.receiveWristSpike(window: window, tsMs: tsMs)
        }

        // iPhone IMU spike windows → Rear hand punches
        phoneMotion.onSpike = { [weak self] window, tsMs in
            self?.fusionEngine.receiveIPhoneSpike(window: window, tsMs: tsMs)
        }

        // AirPods defense gestures
        headphoneMotion.onDefenseGestureDetected = { [weak self] label, tsMs in
            self?.fusionEngine.receiveDefenseGesture(label: label, tsMs: tsMs)
        }

        // Watch-initiated session: start iPhone side using Watch's anchor for timestamp alignment
        watchConnectivity.onSessionStartFromWatch = { [weak self] anchor in
            guard let self, !self.isSessionActive else { return }
            self.mode = .liveCount   // Watch-initiated sessions are always live count
            self.fusionEngine.startSession(anchor: anchor)
            // The Watch started its own session without our collectRaw flag, so
            // Watch raw won't arrive; iPhone/AirPods raw still feed segmentation.
            self.beginLiveRawCapture(anchor: anchor)
            self.phoneMotion.startTracking(anchorTs: anchor)
            self.headphoneMotion.startTracking(anchorTs: anchor)
            // Do NOT send sessionStart back to Watch — Watch already started
            self.isSessionActive = true
        }

        // Watch-initiated session stop
        watchConnectivity.onSessionStopFromWatch = { [weak self] in
            guard let self, self.isSessionActive else { return }
            self.fusionEngine.stopSession()
            self.phoneMotion.stopTracking()
            self.headphoneMotion.stopTracking()
            self.endLiveRawCaptureAndSegment()
            self.isSessionActive = false
        }
    }

    private func bindPublished() {
        watchConnectivity.$isWatchReachable
            .receive(on: RunLoop.main)
            .assign(to: \.isWatchConnected, on: self)
            .store(in: &cancellables)

        headphoneMotion.$isAvailable
            .receive(on: RunLoop.main)
            .assign(to: \.isAirPodsAvailable, on: self)
            .store(in: &cancellables)
    }
}
