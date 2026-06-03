import Foundation
import Combine

// MARK: - Boxing Session Manager
//
// Orchestrates sensor collection and post-session analysis.
//
// Flow:
//   startSession()  → start all sensor managers, begin buffering raw IMU
//   stopSession()   → stop sensors, hand buffer to SessionAnalyzer on a
//                     background Task; sets motionSequence and fusionEngine.stats
//                     when done, then calls onAnalysisComplete.
//
// onAnalysisComplete is wired in ContentView to WorkoutHistoryManager.recordSession.

@MainActor
final class BoxingSessionManager: ObservableObject {

    // MARK: - Dependencies

    let fusionEngine       = PHANTOMFusionEngine()
    private let phoneMotion       = PhoneMotionManager()
    private let headphoneMotion   = HeadphoneMotionManager()
    private let watchConnectivity = WatchConnectivityManager.shared

    /// nil when knn_params.json is missing or fails to decode.
    let classifier: MotionRuleClassifier?

    // MARK: - Published State

    @Published var isSessionActive    = false
    @Published var showSafetyAlert    = false

    @Published var isWatchConnected   = false
    @Published var isAirPodsAvailable = false

    /// Set while SessionAnalyzer is running after session stop.
    @Published var isSegmenting = false

    /// Populated after analysis completes. Shown in StatsView.
    @Published var motionSequence: DetectedMotionSequence?

    /// Called on the main actor once analysis completes and stats are ready.
    var onAnalysisComplete: ((PHANTOMSessionStats) -> Void)?

    // MARK: - Raw Buffer

    private var liveRawSamples: [TimeSeriesRawSample] = []
    private var liveSessionAnchor: TimeInterval = 0

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        classifier = try? MotionRuleClassifier.load()
        wireCallbacks()
        bindPublished()
        if !UserDefaults.standard.bool(forKey: "phantomSafetyAlertShown") {
            showSafetyAlert = true
        }
    }

    // MARK: - Session Control

    func startSession() {
        guard !isSessionActive else { return }
        fusionEngine.startSession()
        let anchor = fusionEngine.anchorTs
        beginRawCapture(anchor: anchor)
        phoneMotion.startTracking(anchorTs: anchor)
        headphoneMotion.startTracking(anchorTs: anchor)
        watchConnectivity.sendStartSession(anchorTs: anchor, collectRaw: classifier != nil)
        isSessionActive = true
    }

    func stopSession() {
        guard isSessionActive else { return }
        fusionEngine.stopSession()
        phoneMotion.stopTracking()
        headphoneMotion.stopTracking()
        watchConnectivity.sendStopSession()
        isSessionActive = false
        analyzeAndApply()
    }

    func dismissSafetyAlert() {
        UserDefaults.standard.set(true, forKey: "phantomSafetyAlertShown")
        showSafetyAlert = false
    }

    func resetMotionSequence() { motionSequence = nil }

    // MARK: - Raw Capture

    private func beginRawCapture(anchor: TimeInterval) {
        motionSequence = nil
        guard classifier != nil else { return }
        liveRawSamples = []
        liveRawSamples.reserveCapacity(16_000)
        liveSessionAnchor = anchor

        phoneMotion.onRawSample = { [weak self] sample in
            self?.liveRawSamples.append(sample)
        }
        headphoneMotion.onRawSample = { [weak self] sample in
            self?.liveRawSamples.append(sample)
        }
        watchConnectivity.onRawWatchBatch = { [weak self] flat in
            guard let self else { return }
            var i = 0
            while i + 6 < flat.count {
                self.liveRawSamples.append(TimeSeriesRawSample(
                    timestamp: flat[i], source: "Watch", label: "",
                    accX: flat[i+1], accY: flat[i+2], accZ: flat[i+3],
                    gyroX: flat[i+4], gyroY: flat[i+5], gyroZ: flat[i+6]
                ))
                i += 7
            }
        }
    }

    // MARK: - Post-Session Analysis

    private func analyzeAndApply() {
        phoneMotion.onRawSample = nil
        headphoneMotion.onRawSample = nil
        watchConnectivity.onRawWatchBatch = nil

        guard let classifier else { return }
        let samples = liveRawSamples
        liveRawSamples = []
        guard !samples.isEmpty else { return }

        let analyzer = SessionAnalyzer(classifier: classifier)
        isSegmenting = true

        Task.detached(priority: .userInitiated) { [weak self] in
            let sequence = analyzer.analyze(samples: samples)
            await MainActor.run {
                guard let self else { return }
                self.motionSequence = sequence
                self.fusionEngine.applyResults(sequence)
                self.isSegmenting = false
                self.onAnalysisComplete?(self.fusionEngine.stats)
            }
        }
    }

    // MARK: - Wiring

    private func wireCallbacks() {
        // Watch-initiated session start
        watchConnectivity.onSessionStartFromWatch = { [weak self] anchor in
            guard let self, !self.isSessionActive else { return }
            self.fusionEngine.startSession(anchor: anchor)
            self.beginRawCapture(anchor: anchor)
            self.phoneMotion.startTracking(anchorTs: anchor)
            self.headphoneMotion.startTracking(anchorTs: anchor)
            self.isSessionActive = true
        }
        // Watch-initiated session stop
        watchConnectivity.onSessionStopFromWatch = { [weak self] in
            guard let self, self.isSessionActive else { return }
            self.fusionEngine.stopSession()
            self.phoneMotion.stopTracking()
            self.headphoneMotion.stopTracking()
            self.isSessionActive = false
            self.analyzeAndApply()
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
