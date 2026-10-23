# PHANTOM — Architecture

Boxing training tracker using iPhone + Apple Watch + AirPods Pro.
Detects punches (lead/rear hand) and defensive head gestures, then computes Offense–Defense Transition Time (TT).

---

## Project Layout

```
PHANTOM/
├── iOS/                    # iPhone app target
│   ├── App/                # Entry point (BoxingTrackerApp, ContentView)
│   ├── Managers/           # Business logic / sensor wrappers
│   ├── Models/             # Codable persistence models
│   ├── Views/              # SwiftUI screens (session, calibration, history, stats)
│   │   └── Session/        # Sub-components of the session screen
│   ├── Resources/          # Bundled KNN references (knn_params*.json)
│   └── DataCollect/        # ML dataset collection (separate tab, not core)
├── Watch/                  # Apple Watch app target
│   ├── WatchBoxingApp.swift
│   ├── WatchBoxingManager.swift
│   └── WatchContentView.swift
├── Shared/                 # Compiled into BOTH targets
│   ├── BoxingModels.swift       # All domain types
│   ├── SpikeDetector.swift      # Acceleration spike detector
│   ├── HeadGestureClassifier.swift  # AirPods gesture classifier
│   ├── WatchMessageKey.swift    # WCSession message key constants
│   └── FileUtils.swift          # File path helpers
└── BoxingTracker.xcodeproj  # Two targets: BoxingTracker + BoxingTrackerWatch
```

### Build Targets

| Target | Output | Sources |
|--------|--------|---------|
| `BoxingTracker` | iPhone app (.ipa) | `iOS/` + `Shared/` |
| `BoxingTrackerWatch` | Watch app (embedded inside iPhone app) | `Watch/` + `Shared/` |

The Watch app is bundled inside the iPhone .ipa and installed to the Watch automatically.

---

## Domain Model (`Shared/BoxingModels.swift`)

```
PunchLabel      — leadJab / leadHook / leadUppercut / rearJab / rearHook / rearUppercut
                  (Phase 1: undifferentiated leadPunch / rearPunch)
DefenseLabel    — leftSlip / rightSlip / leftRoll / rightRoll / pullBack / step / guard_
CombinationLabel — multi-punch sequences, used as DATA COLLECTION session labels only
                   (not a real-time classification target)

BoxingEvent     — { tsMs: Double, kind: .punch(PunchLabel) | .defense(DefenseLabel),
                    transitionTimeMs: Double? }

PHANTOMSessionStats — aggregates BoxingEvent[], computes:
                      totalPunches / totalDefenses / leadPunches / rearPunches
                      averageTTms / fatigueDriftMs / duration
```

---

## Sensor Architecture

Three independent sensor streams feed one central hub (PHANTOMFusionEngine) on iPhone.

```
┌─────────────────────────────────────────────────────────────┐
│                        iPhone                               │
│                                                             │
│  PhoneMotionManager          HeadphoneMotionManager         │
│  (CMMotionManager, 100Hz)    (CMHeadphoneMotionManager)     │
│  Rear-hand wrist IMU         AirPods Pro head IMU           │
│        │                           │                        │
│        ▼                           ▼                        │
│  SpikeDetector               HeadGestureClassifier          │
│  onSpike → rearPunch         onGesture → DefenseLabel       │
│        │                           │                        │
│        └──────────┬────────────────┘                        │
│                   ▼                                         │
│         PHANTOMFusionEngine                                 │
│         ├─ receiveWristSpike()   ← Watch (Stream 1)         │
│         ├─ receiveIPhoneSpike()  ← Phone IMU (Stream 2)     │
│         └─ receiveDefenseGesture() ← AirPods (Stream 3)     │
│                   │                                         │
│                   ▼                                         │
│         PHANTOMSessionStats  +  Transition Time             │
└─────────────────────────────────────────────────────────────┘
                   ▲
                   │ WCSession (spike window + timestamp)
                   │
┌─────────────────────────────────────────────────────────────┐
│                      Apple Watch                            │
│                                                             │
│  CMMotionManager (100Hz) → SpikeDetector → sendSpikeToPhone │
│  HKWorkoutSession — keeps app alive when wrist drops        │
└─────────────────────────────────────────────────────────────┘
```

### Device Roles

| Device | Hand | Sensor | Output |
|--------|------|--------|--------|
| Apple Watch | Lead hand (wrist) | CMMotionManager 100Hz | Spike window → iPhone |
| iPhone | Rear hand (held) | CMMotionManager 100Hz | Rear punch events |
| AirPods Pro | Head | CMHeadphoneMotionManager | Defense gesture events |

---

## Key Components

### `PHANTOMFusionEngine` (`iOS/Managers/`)
Central hub on iPhone. Maintains a session-relative timestamp timeline (anchor = Unix epoch at session start). Receives three streams and computes Transition Time (TT = time from last punch to next defense, valid range 150ms–2000ms).

### `WatchBoxingManager` (`Watch/`)
- Samples wrist at 100Hz, detects spikes via `SpikeDetector`
- Does NOT classify punches — sends raw magnitude window to iPhone
- Holds `HKWorkoutSession` (.boxing) as keep-alive so watchOS doesn't suspend the app when the wrist drops or screen dims
- Persists session state in UserDefaults for crash recovery
- Three WCSession channels:
  - `sendMessage` — live spike/batch (best-effort, dropped if unreachable)
  - `transferUserInfo` — session lifecycle fallback queue (guaranteed)
  - `transferFile` — authoritative raw CSV on session stop (guaranteed)

### `SpikeDetector` (`Shared/`)
Used on both Watch and iPhone. Triggers when magnitude exceeds 1.2g, collects raw window until magnitude falls below 0.6g or 300ms elapsed. Fires `onSpike(window, tsMs)`.

### `HeadGestureClassifier` (`Shared/`)
Phase 1 heuristic: gyroY threshold (1.5 rad/s) for anomaly detection, then classifies slip/roll/pullBack from gyroY direction and accX magnitude. 25-sample cooldown prevents double-firing.

### `SessionAnalyzer` + `MotionRuleClassifier` (`iOS/Managers/`)
Post-session action recognition. All raw IMU samples from the three devices are buffered
during a session; on stop, `SessionAnalyzer` builds a fused motion envelope (40 ms bins,
62% gyro + 38% dynamic acceleration, 5-bin smoothing), finds peaks at least 0.5 s apart and
cuts a -0.5 s / +0.8 s window around each one. `MotionRuleClassifier` turns each window into
the same feature vector as the Python pipeline (`analysis/common/`), applies the frozen
preprocessing (impute -> variance filter -> RobustScaler -> L2) and returns the label of the
nearest reference vector (1-NN, Manhattan distance). Windows that are too still are dropped,
and windows closer to the user's `none` demos are rejected.

Two reference sets ship in `iOS/Resources/`: `knn_params.json` (single actions) and
`knn_params_combo.json` (single actions plus selected combo windows, for fast transitions).
On-device calibration appends the user's labeled demo vectors to both and persists them to
`calibration.json` in Documents.

### `BoxingSessionManager` (`iOS/Managers/`)
SwiftUI ViewModel. Wires the three sensor managers to `PHANTOMFusionEngine`. Handles session start/stop from both iPhone UI and Watch UI (Watch-initiated sessions use Watch's anchorTs for clock alignment).

### `WatchConnectivityManager` (`iOS/Managers/`)
iOS-side WCSession singleton. Routes incoming Watch messages to callbacks:
- `onWristSpike` → `PHANTOMFusionEngine`
- `onRawWatchBatch` → `TimeSeriesCollectManager` (data collection mode)
- `onSessionStartFromWatch` / `onSessionStopFromWatch` → `BoxingSessionManager`
- `session(_:didReceive:)` — moves Watch raw CSV from WC inbox to Documents

### `AudioSessionActivator` (`iOS/Managers/`)
`CMHeadphoneMotionManager` only fires when AirPods are the active AVAudioSession route. This helper activates a Bluetooth-compatible playback session. Must be called before starting headphone motion updates.

---

## Session Lifecycle

```
iPhone UI "Start"
    │
    ├─ PHANTOMFusionEngine.startSession()  → records anchorTs
    ├─ PhoneMotionManager.startTracking(anchorTs)
    ├─ HeadphoneMotionManager.startTracking(anchorTs)
    └─ WatchConnectivityManager.sendStartSession(anchorTs)
                                    │
                                    ▼
                           Watch receives anchor
                           WatchBoxingManager.startSession(anchorTs)
                           CMMotionManager starts @ 100Hz
                           HKWorkoutSession starts (keep-alive)

        ── session active ──

iPhone UI "Stop"
    ├─ PHANTOMFusionEngine.stopSession()
    ├─ PhoneMotionManager.stopTracking()
    ├─ HeadphoneMotionManager.stopTracking()
    └─ WatchConnectivityManager.sendStopSession()
                                    │
                                    ▼
                           WatchBoxingManager.stopSession()
                           CMMotionManager stops
                           HKWorkoutSession ends
```

Session can also be initiated from the Watch — in that case the Watch sends its own anchorTs to iPhone so both devices share the same time origin.

---

## Timestamp Alignment

All events use **session-relative milliseconds**:

```
tsMs = (Date().timeIntervalSince1970 - anchorTs) × 1000
```

`anchorTs` is the Unix epoch recorded at session start on iPhone. The Watch receives this via WCSession so its spike timestamps are directly comparable to iPhone and AirPods events.

---

## Transition Time (TT)

```
lastPunchTsMs = tsMs of most recent punch event

On defense event:
  tt = defenseEvent.tsMs - lastPunchTsMs
  if 150ms < tt < 2000ms → valid TT, stored in BoxingEvent.transitionTimeMs
```

Interpretation: how quickly the boxer switched from attacking to defending.

---

## Data Flow — Raw WCSession Channels

| Channel | Payload | When used | Reliability |
|---------|---------|-----------|-------------|
| `sendMessage` | Live spike window (50-sample batch) | reachable only | Drop if unreachable |
| `transferUserInfo` | Spike events, session start/stop | always queued | Guaranteed (delayed) |
| `transferFile` | Full-session raw CSV at stop | always queued | System-guaranteed |

Live batches are best-effort preview only. The `transferFile` CSV is the authoritative data source.

---

## Data Collection Mode (`iOS/DataCollect/`)

Separate tab in the app, independent of the core boxing session. Two sub-modes:

| Tab | Manager | Purpose |
|-----|---------|---------|
| Collect | `DataCollectManager` | Spike-window labeled capture (feature extraction) |
| Raw Stream | `TimeSeriesCollectManager` | 50Hz continuous IMU from all 3 devices simultaneously |

CSV format (HARApp-compatible):
```
timestamp, source, accX, accY, accZ, gyroX, gyroY, gyroZ, label
```
`source` = `"iPhone"` | `"Watch"` | `"AirPods"`

---

## Workout History

`WorkoutHistoryManager` persists session results to two JSON files in Documents:
- `workout_granular.json` — full session detail for last 7 days (`[DailyRecord]`)
- `workout_summaries.json` — compact daily totals, kept indefinitely (`[DailySummary]`)

Records older than 7 days are archived from granular to summary automatically.

---

## Concurrency Model

All managers are `@MainActor final class`. HealthKit and WCSession callbacks arrive on arbitrary threads — they are dispatched back to MainActor via `Task { @MainActor in }`. WCSession delegate methods are `nonisolated` and use the same pattern.

`WatchBoxingManager.endWorkoutKeepAliveIfIdle()` captures `workoutBuilder` as a local constant before passing it into the `@Sendable` `endCollection` completion closure to avoid MainActor isolation violations.

---

## Entitlements

| Entitlement | Target | Purpose |
|-------------|--------|---------|
| `com.apple.developer.healthkit` | Watch | HKWorkoutSession keep-alive |
| `com.apple.developer.watchkit` | iOS | Embed Watch app |
| `NSMotionUsageDescription` | iOS + Watch | CMMotionManager |
| `NSHeadphoneMotionUsageDescription` | iOS | CMHeadphoneMotionManager |

`com.apple.developer.healthkit.background-delivery` is intentionally excluded — requires special Apple approval and causes SIGKILL on Personal Team signing.
