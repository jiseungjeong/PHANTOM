import Foundation

// MARK: - None Motion Store (Phase 9 (2) / §8.7)
//
// The list of non-boxing motions shown in the Reject tab. Three defaults ship in
// code (Walking, Raise Left/Right Arm); the user may add custom ones and rename
// any of them. Only the LIST (names + isDefault) is persisted to UserDefaults —
// the per-run progress (which motion was demonstrated) is intentionally NOT
// persisted, since calibration is redone each session (M3).
//
// Every motion, default or custom, is calibrated under the single
// MotionRuleClassifier.noneLabel ("none"); the name here is display/progress only
// and never reaches the KNN reference (D4).

struct NoneMotion: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    let isDefault: Bool

    init(id: UUID = UUID(), name: String, isDefault: Bool) {
        self.id = id
        self.name = name
        self.isDefault = isDefault
    }
}

final class NoneMotionStore: ObservableObject {
    @Published private(set) var motions: [NoneMotion]

    private let key = "none_calib_motions"

    /// The shipped defaults (kept in sync with display intent; commit label is
    /// always "none"). isDefault motions can be renamed but not deleted.
    static var defaults: [NoneMotion] {
        MotionRuleClassifier.noneCalibrationActions.map {
            NoneMotion(name: $0, isDefault: true)
        }
    }

    init() {
        if let data = UserDefaults.standard.data(forKey: key),
           let saved = try? JSONDecoder().decode([NoneMotion].self, from: data),
           !saved.isEmpty {
            motions = saved
        } else {
            motions = Self.defaults
        }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(motions) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    // MARK: - Mutations

    func add(name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        motions.append(NoneMotion(name: trimmed, isDefault: false))
        persist()
    }

    func rename(_ id: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let i = motions.firstIndex(where: { $0.id == id }) else { return }
        motions[i].name = trimmed
        persist()
    }

    /// Deletes a CUSTOM motion. Default motions are not deletable (use reset).
    func delete(_ id: UUID) {
        guard let m = motions.first(where: { $0.id == id }), !m.isDefault else { return }
        motions.removeAll { $0.id == id }
        persist()
    }

    /// Full reset (M2): drop all custom motions and restore the default names.
    func resetToDefaults() {
        motions = Self.defaults
        persist()
    }
}
