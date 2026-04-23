import Foundation

// MARK: - Watch ↔ iPhone Message Keys
//
// Shared key namespace used by WCSession payloads between iOS and watchOS.
// Kept as a caseless enum purely for namespacing.

enum WatchMessageKey {
    // Session lifecycle – iPhone → Watch
    static let action       = "action"
    static let sessionStart = "sessionStart"
    static let sessionStop  = "sessionStop"
    static let anchorTs     = "anchorTs"      // Unix epoch, used to sync timestamps

    // Session lifecycle – Watch → iPhone
    static let watchSessionStart = "watchSessionStart"  // Watch-initiated session start
    static let watchSessionStop  = "watchSessionStop"   // Watch-initiated session stop

    // Spike window (Message-based)
    static let wristSpike   = "wristSpike"
    static let window       = "window"        // [Double] raw magnitude array
    static let tsMs         = "tsMs"          // Double session-relative ms

    // Raw time-series streaming – iPhone → Watch
    static let rawStreamStart = "rawStreamStart"
    static let rawStreamStop  = "rawStreamStop"
    static let label          = "label"       // activity label forwarded to Watch

    // Raw time-series batch – Watch → iPhone
    // Payload: flat [Double] of stride-8 tuples: tsMs, accX, accY, accZ, gyroX, gyroY, gyroZ, (pad)
    static let rawBatch       = "rawBatch"
    static let rawSamples     = "rawSamples"  // key for the flat [Double] array

    // File transfer metadata – Watch → iPhone
    static let fileTag        = "fileTag"
    static let rawWatchFile   = "rawWatchFile"   // value for fileTag identifying a Watch raw CSV
}
