import Foundation
import AVFoundation

// MARK: - Audio Session Activator
//
// CMHeadphoneMotionManager only delivers samples when AirPods are the active
// AVAudioSession route. This helper activates an AirPods-compatible route
// (supports AirPods Pro, AirPods Max, AirPods 3rd gen, AirPods 4 ANC).
// Without activation the motion callback fires zero samples and surfaces no
// error, making it look like the device isn't connected.

enum AudioSessionActivator {

    /// Activate a Bluetooth-compatible playback route so AirPods deliver
    /// device motion updates. Safe to call repeatedly.
    static func activateForHeadphoneMotion() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .default,
                                    options: [.allowBluetoothHFP, .allowBluetoothA2DP])
            try session.setActive(true)
        } catch {
            print("[AudioSessionActivator] activation error: \(error.localizedDescription)")
        }
    }
}
