import Foundation
import WatchConnectivity
import os

private let wcLog = Logger(subsystem: "com.jiseung.phantom.boxingtracker", category: "WC-iOS")

// MARK: - Watch Connectivity Manager  (iOS side)
//
// Activates WCSession and handles two-way communication with Apple Watch:
//
//   iPhone → Watch:
//     • Session start: broadcasts anchor timestamp so Watch and iPhone
//       use the same time origin for all event timestamps.
//     • Session stop.
//
//   Watch → iPhone:
//     • Raw spike windows (wristSpike) forwarded to PHANTOMFusionEngine.

@MainActor
final class WatchConnectivityManager: NSObject, ObservableObject {

    static let shared = WatchConnectivityManager()

    @Published var isWatchReachable = false
    @Published var isWatchPaired    = false

    // Callback: (rawMagnitudeWindow, sessionRelativeTsMs)
    var onWristSpike: (([Double], Double) -> Void)?

    // Secondary callback for DataCollectManager (active regardless of session state)
    var onWristSpikeCollect: (([Double], Double) -> Void)?

    // Callback for TimeSeriesCollectManager — raw sample batch from Watch
    // Delivers flat stride-7 array: [tsMs, accX, accY, accZ, gyroX, gyroY, gyroZ, ...]
    var onRawWatchBatch: (([Double]) -> Void)?

    // Callbacks for Watch-initiated session lifecycle
    var onSessionStartFromWatch: ((TimeInterval) -> Void)?
    var onSessionStopFromWatch:  (() -> Void)?

    // Fired after a Watch-side raw CSV has been moved into the iPhone Documents
    // folder. The argument is the final on-disk URL.
    var onWatchRawFileSaved: ((URL) -> Void)?

    override init() {
        super.init()
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    // MARK: - Commands → Watch

    /// Broadcast session start with anchor timestamp so Watch aligns its clock.
    func sendStartSession(anchorTs: TimeInterval) {
        dispatch([
            WatchMessageKey.action:   WatchMessageKey.sessionStart,
            WatchMessageKey.anchorTs: anchorTs
        ])
    }

    func sendStopSession() {
        dispatch([WatchMessageKey.action: WatchMessageKey.sessionStop])
    }

    func sendStartRawStream(anchorTs: TimeInterval, label: String = "") {
        dispatch([
            WatchMessageKey.action:   WatchMessageKey.rawStreamStart,
            WatchMessageKey.anchorTs: anchorTs,
            WatchMessageKey.label:    label
        ])
    }

    func sendStopRawStream() {
        dispatch([WatchMessageKey.action: WatchMessageKey.rawStreamStop])
    }

    // MARK: - Private

    private func dispatch(_ payload: [String: Any]) {
        guard WCSession.isSupported() else { return }

        let action = payload[WatchMessageKey.action] as? String ?? "?"

        // Always try sendMessage first, even when isReachable reports false.
        //
        // Apple's `isReachable` flag is unreliable on iOS: it can stay stale
        // at `false` for the entire lifetime of a launch even after a clean
        // BT handshake, until a sendMessage is actually attempted. The act of
        // calling sendMessage wakes both sides of the WCSession, after which
        // the reachable flag flips to true and stays accurate.
        //
        // If the message really can't be delivered (counterpart app killed,
        // BT off), the errorHandler fires and we fall back to the queued
        // transferUserInfo channel so no payload is lost.
        WCSession.default.sendMessage(payload, replyHandler: nil) { error in
            wcLog.error("[WC-iOS] sendMessage failed (action=\(action)), falling back to userInfo: \(error.localizedDescription)")
            WCSession.default.transferUserInfo(payload)
        }
    }

    private func handle(_ message: [String: Any]) {
        // Raw time-series batch from Watch
        if message[WatchMessageKey.rawBatch] != nil,
           let flat = message[WatchMessageKey.rawSamples] as? [Double] {
            onRawWatchBatch?(flat)
            return
        }

        // Spike window from Watch
        if message[WatchMessageKey.wristSpike] != nil,
           let tsMs   = message[WatchMessageKey.tsMs]   as? Double,
           let window = message[WatchMessageKey.window] as? [Double] {
            onWristSpike?(window, tsMs)
            onWristSpikeCollect?(window, tsMs)
            return
        }

        // Watch-initiated session lifecycle
        guard let action = message[WatchMessageKey.action] as? String else { return }
        switch action {
        case WatchMessageKey.watchSessionStart:
            let anchor = message[WatchMessageKey.anchorTs] as? TimeInterval ?? Date().timeIntervalSince1970
            onSessionStartFromWatch?(anchor)
        case WatchMessageKey.watchSessionStop:
            onSessionStopFromWatch?()
        default:
            break
        }
    }
}

// MARK: - WCSessionDelegate

extension WatchConnectivityManager: WCSessionDelegate {

    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        let paired      = session.isPaired
        let reachable   = session.isReachable
        let installed   = session.isWatchAppInstalled
        let state       = activationState.rawValue
        let errDesc     = error?.localizedDescription ?? "nil"
        Task { @MainActor in
            self.isWatchPaired    = paired
            self.isWatchReachable = reachable
            wcLog.notice("[WC-iOS] activated: state=\(state) paired=\(paired) reachable=\(reachable) installed=\(installed) error=\(errDesc)")
        }
    }

    nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
        let reachable = session.isReachable
        Task { @MainActor in
            self.isWatchReachable = reachable
            wcLog.notice("[WC-iOS] reachability changed: \(reachable)")
        }
    }

    nonisolated func session(_ session: WCSession,
                             didReceiveMessage message: [String: Any]) {
        Task { @MainActor in self.handle(message) }
    }

    nonisolated func session(_ session: WCSession,
                             didReceiveUserInfo userInfo: [String: Any]) {
        Task { @MainActor in self.handle(userInfo) }
    }

    /// Reliable raw CSV delivery from Watch. The Watch hands us a tmp file
    /// inside the WC inbox; we must move it out before this method returns,
    /// otherwise the system will purge it.
    nonisolated func session(_ session: WCSession, didReceive file: WCSessionFile) {
        let tag      = file.metadata?[WatchMessageKey.fileTag] as? String
        let basename = file.metadata?["basename"] as? String
            ?? file.fileURL.lastPathComponent

        guard tag == WatchMessageKey.rawWatchFile else {
            // Unknown file — leave it. The OS will purge after the delegate returns.
            print("[WatchConnectivity] received unknown file tag=\(tag ?? "nil")")
            return
        }

        let dest = FileUtils.documentsURL.appendingPathComponent(basename)

        do {
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.moveItem(at: file.fileURL, to: dest)
            print("[WatchConnectivity] saved Watch raw CSV → \(dest.lastPathComponent)")
            Task { @MainActor in self.onWatchRawFileSaved?(dest) }
        } catch {
            print("[WatchConnectivity] failed to save received file: \(error)")
        }
    }

    // Required on iOS
    nonisolated func sessionDidBecomeInactive(_ session: WCSession) {}
    nonisolated func sessionDidDeactivate(_ session: WCSession) {
        WCSession.default.activate()
    }
}
