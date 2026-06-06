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

    /// Two reference sets: single-only (knn_params) and combo-augmented
    /// (knn_params_combo = single + selected combo windows, for fast-transition demo).
    /// `usingCombo` toggles which one `classifier` returns. featureDim is identical
    /// (469) for both, so user calibration stays compatible across the toggle.
    private let singleClassifier: MotionRuleClassifier?
    private let comboClassifier: MotionRuleClassifier?
    /// When true, classification uses the combo-augmented reference (demo toggle).
    @Published var usingCombo = false
    /// Active classifier (nil when the selected reference failed to load).
    var classifier: MotionRuleClassifier? {
        (usingCombo ? comboClassifier : singleClassifier) ?? singleClassifier
    }
    /// True when the combo reference is available (to show/hide the toggle in UI).
    var comboAvailable: Bool { comboClassifier != nil }
    /// Both reference sets — calibration is applied to BOTH so toggling preserves it
    /// (featureDim is identical, so the same user samples are valid for either).
    private var allClassifiers: [MotionRuleClassifier] {
        [singleClassifier, comboClassifier].compactMap { $0 }
    }

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

    // MARK: - Calibration State (Phase 5 d / §8.4)

    /// True while a single calibration demo is being captured.
    @Published var isCalibrationCapturing = false
    /// Number of user calibration samples currently in the classifier reference.
    @Published var calibrationCount = 0
    /// Preview of the just-captured demo (envelope + peak + crop window) shown for
    /// review BEFORE it is committed to the reference. nil until a capture finishes.
    @Published var lastCalibrationPreview: SessionAnalyzer.SegmentPreview?
    /// Per-action record of when each calibration demo was committed (newest last).
    /// Shown as a list ("Lead Jab — 6/5 20:57 (3 days ago)") in CalibrationView.
    @Published var calibrationEntries: [CalibrationEntry] = []
    private var calibrationCapturing = false
    /// Pending segment awaiting Confirm/Retry (captured but not yet appended).
    private var pendingCalibration: (preview: SessionAnalyzer.SegmentPreview, label: String)?

    struct CalibrationEntry: Codable, Identifiable {
        var id: String { label }       // one entry per action (latest demo)
        let label: String
        let savedAt: Date
    }

    // MARK: - Raw Buffer

    private var liveRawSamples: [TimeSeriesRawSample] = []
    private var liveSessionAnchor: TimeInterval = 0

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        singleClassifier = try? MotionRuleClassifier.load(resource: "knn_params")
        comboClassifier  = try? MotionRuleClassifier.load(resource: "knn_params_combo")
        loadCalibrationFromDisk()          // restore saved calibration (Phase 5b)
        wireCallbacks()
        bindPublished()
        if !UserDefaults.standard.bool(forKey: "phantomSafetyAlertShown") {
            showSafetyAlert = true
        }
    }

    // MARK: - Calibration Persistence (Phase 5b — 1-set save/load)

    private struct CalibrationStore: Codable {
        var dim: Int                       // feature dimension (version/dim guard)
        var vectors: [[Double]]
        var labels: [String]
        var entries: [CalibrationEntry]?   // per-action timestamps (optional = old files)
    }

    private var calibrationURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("calibration.json")
    }

    private func saveCalibrationToDisk() {
        guard let classifier else { return }
        let (vectors, labels) = classifier.exportCalibration()
        if vectors.isEmpty {
            try? FileManager.default.removeItem(at: calibrationURL)   // nothing to keep
            return
        }
        let store = CalibrationStore(dim: classifier.featureDim, vectors: vectors,
                                     labels: labels, entries: calibrationEntries)
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601
        if let data = try? enc.encode(store) {
            try? data.write(to: calibrationURL)
        }
    }

    private func loadCalibrationFromDisk() {
        let dec = JSONDecoder(); dec.dateDecodingStrategy = .iso8601
        guard let classifier,
              let data = try? Data(contentsOf: calibrationURL),
              let store = try? dec.decode(CalibrationStore.self, from: data),
              store.dim == classifier.featureDim                        // ignore stale (model changed)
        else { return }
        // load into BOTH references (same featureDim) so the toggle keeps calibration
        var loaded = false
        for c in allClassifiers {
            loaded = c.loadCalibration(vectors: store.vectors, labels: store.labels) || loaded
        }
        if loaded {
            calibrationCount = classifier.calibrationCount
            calibrationEntries = store.entries ?? []                    // old files: no timestamps
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

    // MARK: - Calibration Capture (Phase 5 d)
    //
    // Reuses the same sensor flow as a session, but instead of analyzing the buffer
    // it cuts the strongest single segment and appends it (labeled) to the KNN
    // reference via MotionRuleClassifier.addCalibrationSample. One call per action
    // demo; the user is guided through the 10 actions by the calibration view.

    /// Begin capturing one calibration demo. Sensors start buffering.
    func startCalibrationCapture() {
        guard !isSessionActive, !calibrationCapturing, classifier != nil else { return }
        fusionEngine.startSession()
        let anchor = fusionEngine.anchorTs
        beginRawCapture(anchor: anchor)
        phoneMotion.startTracking(anchorTs: anchor)
        headphoneMotion.startTracking(anchorTs: anchor)
        watchConnectivity.sendStartSession(anchorTs: anchor, collectRaw: true)
        calibrationCapturing = true
        isCalibrationCapturing = true
    }

    /// Stop capturing and PREVIEW the strongest segment (not yet committed).
    /// Sets lastCalibrationPreview for the view to render (envelope + peak + crop
    /// band). The caller then calls confirmCalibration() to keep it or
    /// retryCalibration() to discard. Returns true if a usable segment was found.
    @discardableResult
    func stopCalibrationCapture(label: String) -> Bool {
        guard calibrationCapturing else { return false }
        fusionEngine.stopSession()
        phoneMotion.stopTracking()
        headphoneMotion.stopTracking()
        watchConnectivity.sendStopSession()
        phoneMotion.onRawSample = nil
        headphoneMotion.onRawSample = nil
        watchConnectivity.onRawWatchBatch = nil
        calibrationCapturing = false
        isCalibrationCapturing = false

        let samples = liveRawSamples
        liveRawSamples = []
        lastCalibrationPreview = nil
        pendingCalibration = nil
        guard let classifier, !samples.isEmpty else { return false }

        let analyzer = SessionAnalyzer(classifier: classifier)
        guard let preview = analyzer.strongestSegmentPreview(samples: samples) else { return false }
        lastCalibrationPreview = preview
        pendingCalibration = (preview, label)
        return true
    }

    /// Commit the pending previewed segment to the KNN reference. Returns true on success.
    @discardableResult
    func confirmCalibration() -> Bool {
        guard let classifier, let pending = pendingCalibration else { return false }
        // apply to BOTH references so the toggle keeps the same calibration
        var added = false
        for c in allClassifiers {
            added = c.addCalibrationSample(samples: pending.preview.window,
                                           label: pending.label) || added
        }
        if added {
            calibrationCount = classifier.calibrationCount
            // K-CAL: one entry per action — update timestamp if re-calibrated.
            calibrationEntries.removeAll { $0.label == pending.label }
            calibrationEntries.append(CalibrationEntry(label: pending.label, savedAt: Date()))
            saveCalibrationToDisk()
        }
        pendingCalibration = nil
        lastCalibrationPreview = nil
        return added
    }

    /// Discard the pending preview WITHOUT committing (retry this action).
    func retryCalibration() {
        pendingCalibration = nil
        lastCalibrationPreview = nil
    }

    /// Remove the most recently committed calibration sample (undo last action).
    @discardableResult
    func undoLastCalibration() -> Bool {
        var ok = false
        for c in allClassifiers { ok = c.removeLastCalibrationSample() || ok }
        if ok {
            calibrationCount = classifier?.calibrationCount ?? 0
            if !calibrationEntries.isEmpty { calibrationEntries.removeLast() }   // K-CAL
            saveCalibrationToDisk()
        }
        return ok
    }

    /// Discard all user calibration, reverting to the factory model.
    func resetCalibration() {
        for c in allClassifiers { c.resetCalibration() }
        calibrationCount = classifier?.calibrationCount ?? 0
        calibrationEntries = []            // K-CAL
        pendingCalibration = nil
        lastCalibrationPreview = nil
        saveCalibrationToDisk()            // removes the file (empty calibration)
    }

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
