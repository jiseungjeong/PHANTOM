#!/usr/bin/env python3
"""Phase 6 — combo segmentation diagnosis (T1 hit-rate + T2 visualization).

plan §2.1. Goal: BEFORE building combo labeling, find out whether our peak
detector actually finds "one peak per combo action" at the right positions.

Why not reuse choose_real_peaks: it enforces MIN_REPS=4 (single = 5 reps), so
combos (2-3 actions) would all be 'invalid'. Combos need a combo-specific detector:
floor-filtered local peaks, then expect N = combo length (NOT 4-5, NOT equal-gap/
equal-height — combo actions differ, e.g. big jab + weak slip).

T1: for each combo file, detect floor-passing peaks and report how often
    (detected count == combo length), plus defense-miss rate.
T2: grid of envelope + detected peaks + filename-order label assignment, for eyeball
    verification (combos have NO per-action ground truth — only visual check).

Run: python3 analysis/five_subject_active_window_classifier/phase6_combo_probe.py
"""

from __future__ import annotations

import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
DATA_DIR = REPO / "Data"
OUT_DIR = THIS.parent / "reports"; OUT_DIR.mkdir(exist_ok=True)
LEGACY = REPO / "analysis" / "jiho_juho_shim_active_window_classifier_minimal"
sys.path.insert(0, str(LEGACY))
import rule_base_classifier as rbc
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.active_window_classifier_minimal"] = types.ModuleType("analysis.active_window_classifier_minimal")
sys.modules["analysis.active_window_classifier_minimal.rule_base_classifier"] = rbc
import build_single_repetition_dataset as bsr
sys.modules["analysis.active_window_classifier_minimal.build_single_repetition_dataset"] = bsr
import build_peak_repetition_dataset as bp
sys.path.insert(0, str(THIS.parent))
import phase1_segment as p1
import dtw_probe as dtwp              # reuse _window_series, dtw_dist, collect_windows, RESAMPLE_N

SHOE = ["seung", "jiho", "juho", "shim", "jo"]   # shoe cohort (JEONG=barefoot holdout excluded)
DEFENSE = {"Left Slip", "Right Slip", "Left Roll", "Right Roll"}
# ④ amplitude prior: strikes hit ~2.8x harder than defenses (single-action probe:
# strike peak-height median 6.13 vs defense 2.19). So a strike label should prefer
# the BIG peaks and a defense label should tolerate (even prefer) the small ones.
STRIKE = {"Lead Jab", "Lead Hook", "Lead Uppercut",
          "Rear Jab", "Rear Hook", "Rear Uppercut"}
MIN_SEP_S = 0.30                      # drop noise twin-peaks (same as single)
WIN_PRE_S = 0.50                      # candidate window around a peak (match single ±0.5/+0.8)
WIN_POST_S = 0.80
WIN_S = 1.30                          # spotting window length (run_spot redefines locally too)

# ⑥ dominant-sensor groups (single-action probe, peak-bin max-score sensor):
#   Lead strikes -> Watch (60-68:17-21:7-8 W:P:A); Rear strikes -> iPhone (13-18:58-63:5-14);
#   defenses -> AirPods head (Right Slip 27:30:70, Left/Right Roll ~AirPods). Left Slip is
#   ambiguous (33:45:45) -> fall back to the combined envelope.
ACTION_SENSOR = {
    "Lead Jab": "Watch", "Lead Hook": "Watch", "Lead Uppercut": "Watch",
    "Rear Jab": "iPhone", "Rear Hook": "iPhone", "Rear Uppercut": "iPhone",
    "Right Slip": "AirPods", "Left Roll": "AirPods", "Right Roll": "AirPods",
    "Left Slip": None,        # ambiguous -> combined-envelope fallback
}


def per_source_envelopes(data):
    """⑥ Per-sensor envelopes on the SAME time grid as bp.motion_envelope(data, 0.0).

    Mirrors motion_envelope's per-source scoring (gyro+dynamic-acc robust z, binned
    max, smoothed) but returns each source SEPARATELY instead of the max+mean blend.
    Uses jump_cut_s=0.0 so bins align 1:1 with run_spot's t_full/env_full (the cut is
    applied later by time filtering). Returns {source: smoothed_binned_array}; arrays
    share n_bins with the combined envelope."""
    dur = bp.relative_duration(data, 0.0)
    if dur <= 0:
        return {}
    n_bins = max(1, int(np.ceil(dur / bp.BIN_S)))
    out = {}
    for source in rbc.SOURCES:
        rows = data.get(source, [])
        if len(rows) < 5:
            continue
        t0 = rows[0][0]
        seg = [r for r in rows if 0 <= r[0] - t0 < dur]
        if len(seg) < 5:
            continue
        arr = np.array([[r[0] - t0, *r[1:]] for r in seg], dtype=float)
        t = arr[:, 0]
        ax, ay, az, gx, gy, gz = (arr[:, i] for i in range(1, 7))
        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        acc_dyn = np.abs(acc_mag - np.median(acc_mag))
        score = 0.62 * bp.robust_positive_z(gyro_mag) + 0.38 * bp.robust_positive_z(acc_dyn)
        score = np.clip(score, 0.0, 14.0)
        binned = np.zeros(n_bins, dtype=float)
        for tt, v in zip(t, score):
            idx = int(min(n_bins - 1, max(0, int(np.floor(tt / bp.BIN_S)))))
            binned[idx] = max(binned[idx], float(v))
        out[source] = bp.smooth(binned)
    return out


# base_jump thresholds (plan §2.2, final design). base_jump (extract_features'
# jump_cut_s) is NOT a cut position — it sits ~2.67s in (after a real action), so
# cutting there eats the action. It is only a FLAG for "a jump exists".
JUMP_SURE_S = 1.5     # base_jump above this -> jump is certain, force a cut
JUMP_NONE_S = 1.0     # base_jump below this -> ambiguous, keep twin-bump rail


def _rising_edge_start(env, peak_idx):
    """Walk left from a peak to the start of its rising edge (the valley before it)."""
    lo = peak_idx
    while lo > 0 and env[lo - 1] < env[lo]:
        lo -= 1
    return lo


def _combo_jump_cut_base(data, base_jump):
    """Combo-safe jump cut (plan §2.2, final design — base_jump as FLAG + valley).

    History: v1 cut at the first peak's rising edge and ate the first real action
    ('점프컷이 너무 많이 먹는다'). v2 added safety rails but left 51/120 real jumps
    un-cut ('점프한 건 확실한데 안 잘림') — because the rails ignored base_jump,
    which median 2.67s vs first-peak 0.68s clearly signals a jump is present.

    Final rule: base_jump is a FLAG, never a cut position (it lands ~2.67s in,
    after a real action). The jump itself is the FIRST major peak; cut at the rest
    valley just after it (before the first REAL action). base_jump size resolves
    the twin-bump ambiguity:
      - base_jump > JUMP_SURE_S : jump certain -> force a cut (ignore twin-bump rail).
      - base_jump < JUMP_NONE_S : ambiguous   -> keep twin-bump rail (may cut 0).
    Cut position is always clamped to BEFORE the first real action's rising edge,
    so a real action is never eaten. If no clear valley, fall back to that rising
    edge (NOT base_jump — that would eat the action).

    Returns the corrected cut time (seconds, full-signal axis)."""
    t, env, dur, _ = bp.motion_envelope(data, 0.0)     # full incl jump
    if len(env) == 0 or dur <= 0:
        return 0.0
    peaks = bp.local_peak_indices(env)
    median = float(np.median(env)); mad = float(np.median(np.abs(env - median))) or 1e-6
    floor = max(float(np.percentile(env, 62)), median + 0.45 * mad)
    major = [i for i in peaks if env[i] >= floor]
    if not major:
        return 0.0
    jump_idx = major[0]
    jump_t = t[jump_idx]
    jump_sure = base_jump >= JUMP_SURE_S      # base_jump FLAG: jump definitely present

    # Rail 1 (deep first peak = real action, not jump). Skip it when the jump is
    # certain (base_jump large) — a real jump can sit a bit later in long combos.
    if jump_t > 0.30 * dur and not jump_sure:
        return 0.0

    next_major = next((m for m in major if m > jump_idx), None)
    if next_major is None:
        # Only one major blob: if a jump is certain it IS that blob's jump; without
        # a following action to protect, leave it (DTW ignores a stray jump).
        return 0.0
    # Clamp limit: never cut past the first real action's rising-edge start.
    first_action_start_t = max(0.0, t[_rising_edge_start(env, next_major)] - 0.05)

    seg = env[jump_idx:next_major + 1]
    if len(seg) < 2:
        return 0.0
    valley_idx = jump_idx + int(np.argmin(seg))
    valley_t = t[valley_idx]
    valley_deep = env[valley_idx] <= 0.55 * env[jump_idx]   # a real rest gap?

    if valley_deep:
        # Clear rest valley -> cut just before it (clamped before the first action).
        return max(0.0, min(valley_t - 0.05, first_action_start_t))
    # No clear valley (twin-bump look).
    if base_jump < JUMP_NONE_S:
        return 0.0          # ambiguous + weak base_jump -> protect twin-bump action
    if jump_sure:
        # Jump is certain but the valley is shallow -> fall back to cutting just
        # before the first real action's rising edge (base_jump itself would eat it).
        return first_action_start_t
    return 0.0              # middling base_jump, shallow valley -> stay conservative


def combo_jump_cut(data, base_jump):
    """⑩ was attempted (AirPods double-bump cut extension) but ROLLED BACK: it ate
    the first defense in 22/24 defense-first combos (a defense is also AirPods-
    dominant and can sit near base_jump, so neither the AirPods ratio nor the
    base_jump position guard separates it from a jump residue). Protecting real
    actions (plan ①: 'eat 0') outweighs fixing the ~JUHO double-bump residue, so
    we keep the base cut. Double-bump residue is accepted (plan §2.2 / C2)."""
    return _combo_jump_cut_base(data, base_jump)


# ⑫a jump-window END from the front AirPods peak group (plan §2.2). A jump bobs the
# head, so it shows as a CLUSTER of AirPods peaks at the very START of the signal.
# The user's mis-segs (jump read as Left/Right Roll) all stem from that cluster
# surviving the base cut. Height can't separate jump from a real defense (probe:
# jump-AP ≈ defense-AP), but POSITION can: the jump cluster sits at the very front,
# a real defense sits right before a strike. So: walk the front AirPods peaks while
# they stay close together (a contiguous cluster) and return the time just after the
# LAST one — but only while still in the front region (< JUMP_FRONT_FRAC * dur) and
# (when known) within base_jump's neighborhood, so a defense-first action is spared.
AP_FRONT_FRAC = 0.45     # jump cluster must live in the front fraction of the signal
AP_GAP_S = 0.70          # peaks closer than this are one cluster; a bigger gap ends it
AP_MIN = 3.0             # ignore tiny AirPods ripples


def airpods_jump_cut(base_jump, src, t_full, env_full):
    """Return a cut time at the end of the front AirPods peak cluster (0 if none)."""
    ap = src.get("AirPods")
    if ap is None or len(ap) != len(env_full):
        return 0.0
    m = float(np.median(ap)); md = float(np.median(np.abs(ap - m))) or 1e-6
    fl = max(float(np.percentile(ap, 62)), m + 0.45 * md)
    pk = [i for i in bp.local_peak_indices(ap) if ap[i] >= max(fl, AP_MIN)]
    if not pk:
        return 0.0
    dur = t_full[-1] if len(t_full) else 0.0
    # cluster must start near the very front (first AP peak within front region)
    if t_full[pk[0]] > AP_FRONT_FRAC * dur:
        return 0.0
    # extend through contiguous AP peaks (gaps < AP_GAP_S), staying in front region
    last = pk[0]
    gap_bins = AP_GAP_S / bp.BIN_S
    for i in pk[1:]:
        if t_full[i] > AP_FRONT_FRAC * dur:
            break
        if i - last <= gap_bins:
            last = i
        else:
            break
    # cut just after the last cluster peak, at the following envelope valley
    hi = last
    while hi + 1 < len(env_full) and env_full[hi + 1] < env_full[hi]:
        hi += 1
    return max(0.0, float(t_full[hi]) - 0.02)


def combo_peaks(times, envelope, n_expect):
    """Combo-specific peak selection: floor-passing local peaks, min-sep enforced,
    then keep the n_expect STRONGEST (no equal-gap/equal-height assumption)."""
    peaks = bp.local_peak_indices(envelope)
    if not peaks:
        return [], "no_peaks"
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median))) or 1e-6
    floor = max(float(np.percentile(envelope, 62)), median + 0.45 * mad)
    real = sorted(i for i in peaks if envelope[i] >= floor)
    # enforce min separation (keep stronger of two close peaks)
    min_sep = max(1, int(round(MIN_SEP_S / bp.BIN_S)))
    pruned = []
    for i in real:
        if pruned and (i - pruned[-1]) < min_sep:
            if envelope[i] > envelope[pruned[-1]]:
                pruned[-1] = i
        else:
            pruned.append(i)
    return pruned, "ok"


def combo_len(label: str) -> int:
    return label.count("-") + 1


def collect():
    """Return list of (subject, label, n_expect, times, env, peaks_all) for combo files."""
    out = []
    for folder in SHOE:
        subj = folder.upper()
        for path in sorted((DATA_DIR / f"{folder}_data").glob("*.csv")):
            if "_Watch_" in path.name:
                continue
            label = p1.label_from_path(path)
            if label is None or "-" not in label:
                continue
            whole, meta = rbc.extract_features(path)
            if whole is None:
                continue
            jump = float(meta.get("jump_cut_s", 0.0))
            data = rbc.read_csv_sources(path)
            times, env, dur, _ = bp.motion_envelope(data, jump)
            if len(env) == 0:
                continue
            n = combo_len(label)
            peaks, _ = combo_peaks(times, env, n)
            out.append((subj, label, n, times, env, peaks, path.name, data, jump))
    return out


def load_single_templates():
    """Single-action 18ch windows grouped by label (templates for matching)."""
    series, labels, _ = dtwp.collect_windows("all18", dtwp.ALL10)
    by_label = defaultdict(list)
    for s, lab in zip(series, labels):
        by_label[lab].append(s)
    return by_label


def candidate_series(data, jump, peak_time_s):
    """18ch z-scored series for a candidate window centered on a combo peak."""
    return dtwp._window_series(data, peak_time_s - WIN_PRE_S, peak_time_s + WIN_POST_S,
                               "all18", jump)


def dp_order_match(cand_series, actions, templates):
    """Order-preserving assignment (X1): pick, in time order, N candidates for the N
    actions minimizing total DTW. cand_series: list (time-ordered). actions: [label].
    Returns chosen candidate indices [c_0<c_1<...] aligned to actions, or None."""
    M, N = len(cand_series), len(actions)
    if M < N or N == 0:
        return None
    # cost[i][a] = min DTW of candidate i to action a's templates (1-NN over templates)
    INF = float("inf")
    cost = [[INF] * N for _ in range(M)]
    for i, cs in enumerate(cand_series):
        if cs is None:
            continue
        for a, act in enumerate(actions):
            tmpls = templates.get(act, [])
            if tmpls:
                cost[i][a] = min(dtwp.dtw_dist(cs, t) for t in tmpls)
    # DP: dp[i][a] = min total cost assigning actions 0..a using candidates up to i,
    # with candidate i assigned to action a. Enforce strictly increasing candidate idx.
    dp = [[INF] * N for _ in range(M)]
    back = [[-1] * N for _ in range(M)]
    for i in range(M):
        for a in range(N):
            if cost[i][a] == INF:
                continue
            if a == 0:
                dp[i][a] = cost[i][a]
            else:
                best, bj = INF, -1
                for j in range(i):       # previous action a-1 at candidate j<i
                    if dp[j][a - 1] < best:
                        best, bj = dp[j][a - 1], j
                if best < INF:
                    dp[i][a] = best + cost[i][a]; back[i][a] = bj
    # best end candidate for last action
    end, bd = -1, INF
    for i in range(M):
        if dp[i][N - 1] < bd:
            bd, end = dp[i][N - 1], i
    if end < 0:
        return None
    chosen = [0] * N; i, a = end, N - 1
    while a >= 0:
        chosen[a] = i; i = back[i][a]; a -= 1
    return chosen


def run_pattern():
    print("# Phase 6 T3-pattern — DP order-preserving template matching (single -> combo)\n")
    templates = load_single_templates()
    print(f"  templates: {sum(len(v) for v in templates.values())} single windows over {len(templates)} actions")
    rows = collect()

    exact_pos = 0       # all N actions assigned (DP succeeded)
    total = 0
    viz = {}            # label -> (subj, times, env, peaks, chosen, actions)
    for subj, label, n, times, env, peaks, fname, data, jump in rows:
        actions = label.split("-")
        if len(peaks) < n:           # not enough candidates -> can't match all
            total += 1
            continue
        cand = [candidate_series(data, jump, times[pk]) for pk in peaks]
        chosen = dp_order_match(cand, actions, templates)
        total += 1
        if chosen is not None:
            exact_pos += 1
            if label not in viz:
                viz[label] = (subj, times, env, peaks, chosen, actions)

    print(f"  DP assignment succeeded (N actions placed): {exact_pos}/{total} = {exact_pos/total*100:.1f}%")
    print("  (vs T1 naive 'det==len' 21.7% — pattern picks N from over-detected candidates)")

    # Visualization: chosen candidates highlighted + assigned label
    items = list(viz.values())[:12]
    ncol = 3; nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 2.4 * nrow), squeeze=False)
    for k, (subj, times, env, peaks, chosen, actions) in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        ax.plot(times, env, color="#1f4e79", lw=1.2)
        ymax = max(env.max(), 1e-6)
        chosen_pk = {peaks[c] for c in chosen}
        for pk in peaks:                       # all candidates: gray
            ax.axvline(times[pk], color="#bbb", lw=0.8)
        for a, c in enumerate(chosen):         # chosen: red + label
            pk = peaks[c]
            ax.axvline(times[pk], color="#d72638", lw=1.4)
            ax.text(times[pk], ymax * 0.95, actions[a], fontsize=6, rotation=90,
                    va="top", ha="right", color="#333")
        ax.set_title(f"{subj} | {'-'.join(actions)}  (cand={len(peaks)}→pick {len(chosen)})",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(items), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Phase 6 T3-pattern — gray=candidate, red=DP-chosen (matched to single template), label=assigned",
                 fontsize=10, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUT_DIR / "combo_pattern_grid.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved -> {out.name}. Eyeball: do red peaks land on the right actions? (D1, no GT)")
    print("# done.")


def run_jumpcut_all():
    """Visualize ALL combo jump-cuts (full envelope incl. jump) + quantify how often
    the cut eats into the first real action (cut lands after the first major peak)."""
    rows_full = []
    for folder in SHOE:
        subj = folder.upper()
        for path in sorted((DATA_DIR / f"{folder}_data").glob("*.csv")):
            if "_Watch_" in path.name:
                continue
            label = p1.label_from_path(path)
            if label is None or "-" not in label:
                continue
            whole, meta = rbc.extract_features(path)
            if whole is None:
                continue
            jump = float(meta.get("jump_cut_s", 0.0))
            data = rbc.read_csv_sources(path)
            t, env, dur, _ = bp.motion_envelope(data, 0.0)   # FULL incl jump
            if len(env) == 0:
                continue
            rows_full.append((subj, label, jump, t, env))

    # Quantify EAT (cut past first major peak) for BASE cut vs CORRECTED cut.
    def first_major_t(t, env):
        peaks = bp.local_peak_indices(env)
        median = float(np.median(env)); mad = float(np.median(np.abs(env - median))) or 1e-6
        floor = max(float(np.percentile(env, 62)), median + 0.45 * mad)
        major = [t[i] for i in peaks if env[i] >= floor]
        return major[0] if major else None

    eaten_base = eaten_corr = 0
    corrected = {}
    for subj, label, jump, t, env, data in rows_full:
        fm = first_major_t(t, env)
        corr = combo_jump_cut(data, jump)
        corrected[(subj, label, jump)] = corr
        if fm is not None:
            if jump > fm + 0.2: eaten_base += 1
            if corr > fm + 0.2: eaten_corr += 1
    n = len(rows_full)
    print(f"# Combo jump-cut full check ({n} combos)")
    print(f"  BASE cut eats first action:      {eaten_base}/{n} = {eaten_base/n*100:.1f}%")
    print(f"  CORRECTED (combo_jump_cut) eats: {eaten_corr}/{n} = {eaten_corr/n*100:.1f}%\n")

    # Grid: all combos, full envelope + cut line
    ncol = 6
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.4 * nrow), squeeze=False)
    for k, (subj, label, jump, t, env) in enumerate(rows_full):
        ax = axes[k // ncol][k % ncol]
        ax.plot(t, env, color="#1f4e79", lw=0.7)
        ax.axvline(jump, color="red", ls="--", lw=0.9)
        ax.axvspan(0, jump, color="red", alpha=0.07)
        # mark first major peak
        peaks = bp.local_peak_indices(env)
        median = float(np.median(env)); mad = float(np.median(np.abs(env - median))) or 1e-6
        floor = max(float(np.percentile(env, 62)), median + 0.45 * mad)
        major = [t[i] for i in peaks if env[i] >= floor]
        eaten_flag = bool(major and jump > major[0] + 0.2)
        ax.set_title(f"{subj[:2]} {label.count('-')+1}a" + (" EAT" if eaten_flag else ""),
                     fontsize=5, color="red" if eaten_flag else "black")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"All {n} combo jump-cuts (red=cut, shaded=discarded; EAT=cut past 1st action peak {eaten_base}/{n})",
                 fontsize=10, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUT_DIR / "combo_jumpcut_all.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  saved -> {out.name}")
    print("# done.")


def run_spot():
    """Pattern spotting (plan §2.2): NO peak detection, NO jump cut. Slide a window
    over the FULL combo signal; at each position, DTW-match against each action's
    single-action template. DP assigns the N actions to N non-overlapping, time-
    ordered positions minimizing total DTW. Jump auto-ignored (doesn't match any
    action shape); over-detection moot (no peaks)."""
    print("# Phase 6 — pattern spotting (sliding-window DTW, no peak/jump-cut)\n")
    templates = load_single_templates()
    print(f"  templates: {sum(len(v) for v in templates.values())} single windows / {len(templates)} actions")

    WIN_S = 1.30          # window length ≈ single window span (±0.5/+0.8)
    STRIDE_S = 0.20       # slide step
    rows = collect()

    succeeded = 0; total = 0
    first_in_jump = 0          # chosen first action still sits in the jump residue (should drop to ~0)
    viz = []                   # every combo (full census) for eyeball QC
    for subj, label, n, times, env, peaks, fname, data, jump in rows:
        actions = label.split("-")
        # FULL signal axis (no jump cut). Window starts from 0 to end-WIN.
        t_full, env_full, dur, _ = bp.motion_envelope(data, 0.0)
        if len(env_full) == 0 or dur < WIN_S:
            total += 1; continue
        # ① combo-safe jump cut (plan §2.1) — diagnosis (--spot-diag) showed the jump
        # residue is picked as the FIRST action in 75% of combos. Pull the cut back to
        # the first real action's rising edge, then DROP candidate peaks before it.
        cut = combo_jump_cut(data, jump)
        # ⑥ per-sensor envelopes (Lead->Watch, Rear->iPhone, defense->AirPods).
        src_env = per_source_envelopes(data)        # {source: array aligned to t_full}
        # B+⑥ — candidate peaks = UNION of floor-passing peaks across the combined
        # envelope AND each per-source envelope (so an action only loud on its own
        # sensor is not missed), merged by MIN_SEP_S. Combined envelope is the
        # fallback for Left Slip / single-sensor files.
        def _floor(e):
            m = float(np.median(e)); md = float(np.median(np.abs(e - m))) or 1e-6
            return max(float(np.percentile(e, 62)), m + 0.45 * md)
        cand_bins = set()
        for e in [env_full, *src_env.values()]:
            if len(e) != len(env_full):
                continue
            fl = _floor(e)
            for i in bp.local_peak_indices(e):
                if e[i] >= fl and t_full[i] >= cut:
                    cand_bins.add(i)
        # ⑩′ strict jump-residue drop. A jump is a DOUBLE bump; the cut clears the
        # first, but a big AirPods-dominated second bump can survive and get read as
        # the first action (JUHO Lead Jab-Lead Jab: t=2.82 AP=14, and the whole combo
        # shifts by one). GUARD against eating a real defense (also AirPods-dominant):
        # only drop bins STRICTLY INSIDE the jump zone (t < base_jump), since a real
        # defense lands AFTER base_jump. ⑩ failed earlier by using base_jump+0.5 (too
        # wide, ate 22 defenses); base_jump (eps~0) is strict enough to spare them.
        ap = src_env.get("AirPods")
        w = src_env.get("Watch"); ip = src_env.get("iPhone")
        if (jump >= JUMP_SURE_S and ap is not None and len(ap) == len(env_full)
                and w is not None and ip is not None and len(w) == len(env_full)):
            cand_bins = {i for i in cand_bins
                         if not (t_full[i] < jump                       # inside jump zone
                                 and ap[i] >= 3.0                       # head loud
                                 and ap[i] > 1.6 * max(w[i], ip[i]))}   # AirPods-dominant
        # merge bins closer than MIN_SEP_S (keep the one with larger combined env)
        min_sep_bins = max(1, int(round(MIN_SEP_S / bp.BIN_S)))
        merged = []
        for i in sorted(cand_bins):
            if merged and i - merged[-1] < min_sep_bins:
                if env_full[i] > env_full[merged[-1]]:
                    merged[-1] = i
            else:
                merged.append(i)
        starts, energy_by_src, energy_comb = [], [], []
        for pi in merged:
            s = max(cut, t_full[pi] - WIN_PRE_S)
            s = min(s, dur - WIN_S)
            if s < 0:
                continue
            starts.append(s)
            # ⑥+⑨ store the candidate peak height in EACH sensor's envelope (read at
            # the same bin pi), plus the combined value for fallback.
            energy_by_src.append({src: float(e[pi]) for src, e in src_env.items()
                                  if len(e) == len(env_full)})
            energy_comb.append(float(env_full[pi]))
        if len(starts) < n:
            total += 1; continue
        cand = [dtwp._window_series(data, s, s + WIN_S, "all18", 0.0) for s in starts]
        # ②④⑥ LABEL-CONDITIONAL multiplicative energy term (plan §2.2). The energy a
        # label sees comes from ITS dominant sensor (⑥): a strike on its own sensor
        # reads big, a defense on the head sensor reads its true height. e_norm is
        # normalized WITHIN that sensor (⑥ trap-3: never compare sensors' raw values).
        #   strike label  : penalty = 1 + GAMMA·(1 − e_norm)   -> prefers BIG peaks
        #   defense label : penalty = 1 + DEF_GAMMA·e_norm     -> prefers SMALL peaks
        GAMMA = 2.0          # strike: how hard a small candidate is penalized
        DEF_GAMMA = 1.2      # defense: how hard a BIG candidate is penalized (softer)
        # ⑦ⓐ strike amplitude GATE. Single-action probe: strikes hit ~2.8x harder
        # than defenses. A strike that lands on a SMALL peak (its dominant sensor is
        # quiet) is almost surely wrong (an end-tail or ripple mistaken for a punch —
        # the Lead Jab-Lead Jab / Right Roll-...-Lead Hook mis-seg). Add a soft, extra
        # multiplicative penalty when a strike's dominant-sensor e_norm < STK_MIN.
        STK_MIN = 0.5        # a strike should be >= half its sensor's biggest peak
        STK_GATE = 4.0       # strength of the extra below-threshold penalty
        # per-sensor emax (and combined emax) for within-sensor normalization.
        emax_comb = max(energy_comb) or 1.0
        emax_src = {}
        for src in src_env:
            vals = [d.get(src, 0.0) for d in energy_by_src]
            emax_src[src] = max(vals) or 1.0

        def _e_norm(i, act):
            src = ACTION_SENSOR.get(act)            # dominant sensor for this label
            if src is None or src not in emax_src:  # Left Slip / missing -> combined
                return energy_comb[i] / emax_comb
            return energy_by_src[i].get(src, 0.0) / emax_src[src]

        INF = float("inf")
        M = len(starts)
        cost = [[INF] * n for _ in range(M)]
        for i, cs in enumerate(cand):
            if cs is None:
                continue
            for a, act in enumerate(actions):
                tmpls = templates.get(act, [])
                if not tmpls:
                    continue
                e_norm = _e_norm(i, act)
                if act in DEFENSE:
                    # ⑨ defense prefers the BIG peak in ITS sensor (AirPods). The old
                    # '+DEF_GAMMA*e_norm' (small-peak preference) was a leftover from the
                    # combined-envelope era ('defense is small'); once ⑥ normalizes per
                    # sensor, a defense IS the big peak on AirPods, so flip the sign.
                    penalty = 1.0 + DEF_GAMMA * (1.0 - e_norm)  # defense -> big AirPods peak
                else:
                    penalty = 1.0 + GAMMA * (1.0 - e_norm)      # strike -> big peak
                    if e_norm < STK_MIN:                        # ⑦ⓐ strike gate
                        penalty *= 1.0 + STK_GATE * (STK_MIN - e_norm)
                cost[i][a] = min(dtwp.dtw_dist(cs, t) for t in tmpls) * penalty
        # DP: assign actions 0..n-1 to time-ordered, non-overlapping peak windows.
        # ⑧ SENSOR-AWARE MIN_GAP. Base 0.585 (WIN_S*0.45) rejected genuinely-adjacent
        # actions ~0.32s apart (Lead Jab-Rear Hook: cost already pointed at both, but
        # the gap rule pushed Rear Hook to the end-tail). Two cases need relaxing:
        #   ⑧a different dominant sensors (Watch vs iPhone) -> surely two actions even
        #      if close -> relax gap (×0.5).
        #   ⑧b same sensor but a DEEP valley between the two candidate peaks -> two
        #      separate actions (e.g. Lead Jab-Lead Jab) -> relax gap.
        # Else (same sensor + shallow valley = one action's twin bump) -> keep base.
        MIN_GAP_S = WIN_S * 0.45
        RELAX = 0.5                  # gap multiplier when actions are clearly distinct
        VALLEY_RATIO = 0.6           # valley < ratio * min(peak_j, peak_i) = deep dip
        # 0.6 chosen from the valley-ratio distribution (probe_valdist): same-sensor
        # close pairs cluster at 0.74-1.0 (true twin-bumps), and 0.55-0.60 is an empty
        # gap, so 0.5->0.6 rescues the 0.54 Lead Jab-Lead Jab (155044) WITHOUT relaxing
        # any twin-bump (0 extra relaxations). Going to 0.65 would start splitting them.

        def _peak_t(k):              # candidate peak time (window start + pre-roll)
            return starts[k] + WIN_PRE_S

        def _min_gap(j, i, prev_act, cur_act):
            sj, si = ACTION_SENSOR.get(prev_act), ACTION_SENSOR.get(cur_act)
            if sj is not None and si is not None and sj != si:
                return MIN_GAP_S * RELAX                    # ⑧a different sensors
            # ⑧b deep valley between the two candidate peaks (same/ambiguous sensor)
            tj, ti = _peak_t(j), _peak_t(i)
            lo = int(min(len(env_full) - 1, max(0, tj / bp.BIN_S)))
            hi = int(min(len(env_full) - 1, max(0, ti / bp.BIN_S)))
            if hi - lo >= 2:
                valley = float(np.min(env_full[lo:hi + 1]))
                pk_min = min(env_full[lo], env_full[hi])
                if pk_min > 1e-9 and valley < VALLEY_RATIO * pk_min:
                    return MIN_GAP_S * RELAX                # ⑧b deep dip = two actions
            return MIN_GAP_S                                # twin-bump -> keep base

        dp = [[INF] * n for _ in range(M)]
        back = [[-1] * n for _ in range(M)]
        for i in range(M):
            for a in range(n):
                if cost[i][a] == INF:
                    continue
                if a == 0:
                    dp[i][a] = cost[i][a]
                else:
                    best, bj = INF, -1
                    for j in range(i):                       # previous action, earlier & non-overlapping
                        gap = _min_gap(j, i, actions[a - 1], actions[a])
                        if starts[i] - starts[j] >= gap and dp[j][a - 1] < best:
                            best, bj = dp[j][a - 1], j
                    if best < INF:
                        dp[i][a] = best + cost[i][a]; back[i][a] = bj
        end, bd = -1, INF
        for i in range(M):
            if dp[i][n - 1] < bd:
                bd, end = dp[i][n - 1], i
        total += 1
        if end < 0:
            continue
        chosen = [0] * n; i, a = end, n - 1
        while a >= 0:
            chosen[a] = i; i = back[i][a]; a -= 1
        succeeded += 1
        chosen_starts = [starts[c] for c in chosen]
        # first chosen window's center before the cut = jump residue still mistaken (should be ~0)
        if chosen_starts and (chosen_starts[0] + WIN_S / 2) < cut:
            first_in_jump += 1
        # collect EVERY combo (all subjects/files), not just first-per-type, so the
        # grid is a full census for eyeball QC (no ground-truth labels exist).
        viz.append((subj, label, t_full, env_full, chosen_starts, actions, cut))

    print(f"  spotting succeeded (N actions placed, non-overlap): {succeeded}/{total} = {succeeded/total*100:.1f}%")
    print(f"  first chosen action still inside jump residue: {first_in_jump}/{total} = {first_in_jump/total*100:.1f}%"
          f"   (was 75.0% before the jump-cut fix; lower is better)")

    # Visualization: FULL census — every combo across all subjects, paginated.
    # 120 panels don't fit one figure, so write multiple pages of PAGE_ROWS×ncol.
    # Sorted by subject then label so each subject's combos group together.
    items = sorted(viz, key=lambda r: (r[0], r[1]))
    # number repeated (subject,label) takes so identical titles are distinguishable
    # (each combo is recorded 3x): "SHIM | Lead Jab-Lead Jab #1/#2/#3".
    rep_idx = []
    _seen = {}
    for r in items:
        key = (r[0], r[1])
        _seen[key] = _seen.get(key, 0) + 1
        rep_idx.append(_seen[key])
    rep_total = {k: c for k, c in _seen.items()}
    cols = ["#7fb3ff", "#84d68a", "#ffd166", "#c792ea"]
    ncol = 4
    PAGE_ROWS = 6                                 # 24 panels / page
    per_page = ncol * PAGE_ROWS
    n_pages = (len(items) + per_page - 1) // per_page
    saved = []
    for pg in range(n_pages):
        chunk = items[pg * per_page:(pg + 1) * per_page]
        chunk_rep = rep_idx[pg * per_page:(pg + 1) * per_page]
        nrow = (len(chunk) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.2 * nrow), squeeze=False)
        for k, (subj, label, t, env, starts_sel, actions, cut) in enumerate(chunk):
            ax = axes[k // ncol][k % ncol]
            ax.plot(t, env, color="#1f4e79", lw=0.9)
            ymax = max(env.max(), 1e-6)
            if cut > 0:                           # combo-safe jump cut (everything left dropped)
                ax.axvspan(0, cut, color="#ff5555", alpha=0.10)
                ax.axvline(cut, color="#d11", lw=0.9, ls="--")
            for a, s in enumerate(starts_sel):
                ax.axvspan(s, s + WIN_S, color=cols[a % 4], alpha=0.35)
                ax.text(s + WIN_S / 2, ymax * 0.95, actions[a], fontsize=5.5, rotation=90,
                        va="top", ha="center", color="#222")
            rep = chunk_rep[k]
            ntot = rep_total[(subj, label)]
            suffix = f"  #{rep}/{ntot}" if ntot > 1 else ""
            ax.set_title(f"{subj} | {label}{suffix}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        for k in range(len(chunk), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"Phase 6 pattern spotting — ALL combos, page {pg + 1}/{n_pages}  "
                     "(bands=DP-chosen action windows, red dashed=combo-safe jump cut)",
                     fontsize=9, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out = OUT_DIR / f"combo_spotting_all_p{pg + 1}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        saved.append(out.name)
    print(f"\n  saved {len(items)} combos across {n_pages} pages -> {', '.join(saved)}")
    print("  Eyeball each page: do bands land on the right actions (& jump NOT picked)?")
    print("# done.")


def run_spot_diag():
    """Diagnostic for pattern spotting (plan §2.2, before the integrated fix).
    Same candidate/cost/DP logic as run_spot, but instruments WHY bands miss:
      - overlays ALL candidate peak positions (gray ticks) vs DP-chosen (colored bands)
      - marks the base jump-cut time (vertical dashed) and the jump region
      - reports, per combo, whether a JUMP-region window was chosen, and prints the
        chosen action + its match cost so we can tell jump-blame from amplitude-blame.
    Question answered: are missed bands caused by (1) the jump peak being picked as
    the first action, or (2) z-score amplitude loss making small ripples win?"""
    print("# Phase 6 — pattern spotting DIAGNOSIS (jump vs amplitude)\n")
    templates = load_single_templates()
    print(f"  templates: {sum(len(v) for v in templates.values())} single windows / {len(templates)} actions")

    WIN_S = 1.30
    rows = collect()

    total = 0
    jump_picked_first = 0     # a chosen window overlaps the jump region
    jump_is_first_action = 0  # the FIRST chosen action sits in the jump region
    viz = {}
    for subj, label, n, times, env, peaks, fname, data, jump in rows:
        actions = label.split("-")
        t_full, env_full, dur, _ = bp.motion_envelope(data, 0.0)
        if len(env_full) == 0 or dur < WIN_S:
            total += 1; continue
        median = float(np.median(env_full)); mad = float(np.median(np.abs(env_full - median))) or 1e-6
        floor = max(float(np.percentile(env_full, 62)), median + 0.45 * mad)
        peak_idx = [i for i in bp.local_peak_indices(env_full) if env_full[i] >= floor]
        starts, win_energy, peak_times = [], [], []
        for pi in peak_idx:
            s = max(0.0, t_full[pi] - WIN_PRE_S)
            s = min(s, dur - WIN_S)
            if s < 0:
                continue
            starts.append(s)
            peak_times.append(float(t_full[pi]))
            mask = (t_full >= s) & (t_full <= s + WIN_S)
            win_energy.append(float(env_full[mask].mean()) if mask.any() else 0.0)
        if len(starts) < n:
            total += 1; continue
        cand = [dtwp._window_series(data, s, s + WIN_S, "all18", 0.0) for s in starts]
        emax = max(win_energy) or 1.0
        LAMBDA = 4.0
        INF = float("inf")
        M = len(starts)
        cost = [[INF] * n for _ in range(M)]
        for i, cs in enumerate(cand):
            if cs is None:
                continue
            e_norm = win_energy[i] / emax
            for a, act in enumerate(actions):
                tmpls = templates.get(act, [])
                if tmpls:
                    cost[i][a] = min(dtwp.dtw_dist(cs, t) for t in tmpls) - LAMBDA * e_norm
        MIN_GAP_S = WIN_S * 0.6
        dp = [[INF] * n for _ in range(M)]
        back = [[-1] * n for _ in range(M)]
        for i in range(M):
            for a in range(n):
                if cost[i][a] == INF:
                    continue
                if a == 0:
                    dp[i][a] = cost[i][a]
                else:
                    best, bj = INF, -1
                    for j in range(i):
                        if starts[i] - starts[j] >= MIN_GAP_S and dp[j][a - 1] < best:
                            best, bj = dp[j][a - 1], j
                    if best < INF:
                        dp[i][a] = best + cost[i][a]; back[i][a] = bj
        end, bd = -1, INF
        for i in range(M):
            if dp[i][n - 1] < bd:
                bd, end = dp[i][n - 1], i
        total += 1
        if end < 0:
            continue
        chosen = [0] * n; i, a = end, n - 1
        while a >= 0:
            chosen[a] = i; i = back[i][a]; a -= 1

        # jump region = [base jump-cut .. first true action]. A window centered in
        # this region (its peak time < jump) means the jump peak was treated as signal.
        chosen_peak_t = [peak_times[c] for c in chosen]
        in_jump = [pt < jump for pt in chosen_peak_t]      # peak before the jump cut = jump residue
        if any(in_jump):
            jump_picked_first += 1
        if in_jump and in_jump[0]:
            jump_is_first_action += 1

        if label not in viz:
            viz[label] = (subj, t_full, env_full, [starts[c] for c in chosen],
                          chosen_peak_t, actions, peak_times, jump, in_jump)

    print(f"\n  combos diagnosed: {total}")
    print(f"  combos where a JUMP-region window was chosen: {jump_picked_first}/{total} "
          f"= {jump_picked_first/total*100:.1f}%")
    print(f"  combos where the FIRST chosen action is in the jump region: "
          f"{jump_is_first_action}/{total} = {jump_is_first_action/total*100:.1f}%")
    print("  -> high jump-region % = jump-cut is the culprit; near-zero = amplitude (z-score) is.")

    items = list(viz.values())[:12]
    ncol = 3; nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 2.6 * nrow), squeeze=False)
    cols = ["#7fb3ff", "#84d68a", "#ffd166", "#c792ea"]
    for k, (subj, t, env, starts_sel, cpt, actions, all_pt, jump, in_jump) in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        ax.plot(t, env, color="#1f4e79", lw=1.0)
        ymax = max(env.max(), 1e-6)
        # all candidate peaks (gray ticks) — the search space
        for pt in all_pt:
            ax.axvline(pt, color="#bbbbbb", lw=0.6, ls=":")
        # base jump-cut line + shaded jump region [0..jump]
        if jump > 0:
            ax.axvspan(0, jump, color="#ff5555", alpha=0.10)
            ax.axvline(jump, color="#d11", lw=1.0, ls="--")
        # DP-chosen windows (colored bands); red title-mark if a jump-region window chosen
        for a, s in enumerate(starts_sel):
            c = "#ff3333" if in_jump[a] else cols[a % 4]
            ax.axvspan(s, s + WIN_S, color=c, alpha=0.35)
            ax.text(s + WIN_S / 2, ymax * 0.95, actions[a], fontsize=6, rotation=90,
                    va="top", ha="center", color="#222")
        flag = " [JUMP-PICKED]" if any(in_jump) else ""
        ax.set_title(f"{subj} | {'-'.join(actions)}{flag}", fontsize=8,
                     color=("#c00" if any(in_jump) else "#222"))
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(items), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Phase 6 spotting DIAGNOSIS — gray:all candidate peaks  red dashed:base jump-cut  "
                 "shaded red:jump region  bands:DP-chosen (red band=chose jump region)",
                 fontsize=9, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUT_DIR / "combo_spotting_diag.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved -> {out.name}.")
    print("# done.")


def _anchor_floor(e):
    m = float(np.median(e)); md = float(np.median(np.abs(e - m))) or 1e-6
    return max(float(np.percentile(e, 62)), m + 0.45 * md)


def anchor_segment(data, label, jump):
    """⑪⑫ anchor-first placement (shared by run_anchor viz and the labelset builder).

    Returns (windows, cut) where windows = [(action, start_s, end_s, conf), ...] in label
    order, or (None, cut) if placement fails. Stage 1 = strike anchors on dominant-
    sensor peaks (sensor-aware separation ⑫b); Stage 2 = defenses on AirPods peaks in
    the label-implied interval. Cut = max(base jump cut, AirPods front-cluster ⑫a)."""
    actions = label.split("-")
    n = len(actions)
    t_full, env_full, dur, _ = bp.motion_envelope(data, 0.0)
    if len(env_full) == 0 or dur < WIN_S:
        return None, 0.0
    src = per_source_envelopes(data)
    if not src:
        return None, 0.0
    cut = max(combo_jump_cut(data, jump), airpods_jump_cut(jump, src, t_full, env_full))

    def sensor_peaks(sensor):
        e = src.get(sensor)
        if e is None or len(e) != len(env_full):
            return []
        fl = _anchor_floor(e)
        return [i for i in bp.local_peak_indices(e) if e[i] >= fl and t_full[i] >= cut]

    strike_pos = [a for a, act in enumerate(actions) if act in STRIKE]
    cand = {}
    for a in strike_pos:
        pk = sensor_peaks(ACTION_SENSOR.get(actions[a]))
        if not pk:
            return None, cut
        cand[a] = sorted(pk)

    same_sep = max(1, int(round(MIN_SEP_S / bp.BIN_S)))
    diff_sep = max(1, int(round(0.10 / bp.BIN_S)))
    ks = len(strike_pos)
    INF = float("inf")
    best = {(-1, -10**9): (0.0, [])}
    prev_sensor = None
    for ai, a in enumerate(strike_pos):
        sensor = ACTION_SENSOR.get(actions[a]); e = src[sensor]
        sep = diff_sep if (prev_sensor is not None and prev_sensor != sensor) else same_sep
        nxt = {}
        for (pidx, lastbin), (score, path) in best.items():
            for b in cand[a]:
                if b <= lastbin + sep - 1:
                    continue
                nscore = score + e[b]
                key = (ai, b)
                if key not in nxt or nscore > nxt[key][0]:
                    nxt[key] = (nscore, path + [(a, b)])
        if not nxt:
            return None, cut
        best = nxt; prev_sensor = sensor
    final = max((v for k, v in best.items() if k[0] == ks - 1), key=lambda x: x[0], default=None)
    if final is None:
        return None, cut
    chosen_anchor = {a: b for a, b in final[1]}

    ap = src.get("AirPods")
    chosen = dict(chosen_anchor)
    for a, act in enumerate(actions):
        if act not in DEFENSE:
            continue
        prev_anchor = max((chosen_anchor[s] for s in strike_pos if s < a), default=None)
        next_anchor = min((chosen_anchor[s] for s in strike_pos if s > a), default=None)
        lo = prev_anchor if prev_anchor is not None else int(cut / bp.BIN_S)
        hi = next_anchor if next_anchor is not None else len(env_full) - 1
        cands = []
        if ap is not None and len(ap) == len(env_full):
            fl = _anchor_floor(ap)
            cands = [i for i in bp.local_peak_indices(ap)
                     if fl <= ap[i] and lo < i < hi and t_full[i] >= cut and i not in chosen.values()]
        if cands:
            chosen[a] = max(cands, key=lambda i: ap[i])
        else:
            fl = _anchor_floor(env_full)
            cc = [i for i in bp.local_peak_indices(env_full)
                  if env_full[i] >= fl and lo < i < hi and i not in chosen.values()]
            if not cc:
                return None, cut
            chosen[a] = max(cc, key=lambda i: env_full[i])
    if len(set(chosen.values())) < n:
        return None, cut
    windows = []
    # per-window confidence = chosen peak height in the action's dominant sensor,
    # normalized by that sensor's max in this combo (1.0 = the sensor's biggest peak).
    # High = the action sits on a strong, sensor-consistent peak (trustworthy label).
    sensor_max = {}
    for sname, e in src.items():
        if len(e) == len(env_full):
            sensor_max[sname] = float(np.max(e)) or 1.0
    windows = []
    for a in range(n):
        pi = chosen[a]
        s = max(cut, t_full[pi] - WIN_PRE_S); s = min(s, dur - WIN_S)
        s = max(0.0, s)
        sname = ACTION_SENSOR.get(actions[a])
        if sname and sname in src and len(src[sname]) == len(env_full):
            conf = float(src[sname][pi]) / sensor_max.get(sname, 1.0)
        else:
            conf = float(env_full[pi]) / (float(np.max(env_full)) or 1.0)
        windows.append((actions[a], s, s + WIN_S, conf))
    return windows, cut


def run_labelset():
    """Build the combo label set: anchor-segment each combo into per-action windows,
    feature-ize each window with bsr.features_for_window (the SINGLE-action format, so
    it is KNN-compatible), and save combo_windows.npz (X, y, subject, file). This is
    the data product that Phase 7 (a)/(b) consume to test/strengthen the classifier on
    fast action transitions (plan §C2 purpose chain)."""
    print("# Phase 7 — combo label set (anchor windows -> single-format features)\n")
    rows = collect()
    X, y, subjects, files, conf = [], [], [], [], []
    placed = 0; total = 0; feat_fail = 0
    feat_names = None
    for subj, label, n, times, env, peaks, fname, data, jump in rows:
        total += 1
        windows, cut = anchor_segment(data, label, jump)
        if windows is None:
            continue
        placed += 1
        for act, s, e, c in windows:
            # jump_cut_s=0: windows are on the full-signal axis; crop [s,e] directly.
            feats, meta = bsr.features_for_window(data, 0.0, jump > 0, s, e)
            if feats is None:
                feat_fail += 1
                continue
            if feat_names is None:
                feat_names = sorted(feats.keys())
            X.append([float(feats.get(k, 0.0)) for k in feat_names])
            y.append(act); subjects.append(subj); files.append(fname); conf.append(c)
    Xa = np.array(X, dtype=float)
    out = OUT_DIR / "combo_windows.npz"
    np.savez(out, X=Xa, y=np.array(y), subject=np.array(subjects),
             files=np.array(files), feature_names=np.array(feat_names or []),
             conf=np.array(conf, dtype=float))
    print(f"  combos placed: {placed}/{total}")
    print(f"  windows feature-ized: {len(y)}  (feature-fail {feat_fail})")
    print(f"  feature dim: {Xa.shape[1] if Xa.size else 0}")
    from collections import Counter as _C
    print(f"  per-label: {dict(sorted(_C(y).items()))}")
    print(f"  per-subject: {dict(sorted(_C(subjects).items()))}")
    print(f"  saved -> {out.name}")
    print("# done.")


def run_anchor(anon=False):
    """⑪ anchor-first 2-stage segmentation (plan §2.2). Instead of matching every
    label to every peak at once (the combined-DP that z-score C2 keeps fooling),
    exploit that the combo LABEL is known and strikes are loud & sensor-specific:

      Stage 1 (anchors): place the strike actions FIRST, each on its dominant
        sensor's peak (Lead*->Watch, Rear*->iPhone), time-ordered, distinct peaks.
        Strikes are big and unambiguous -> high-confidence skeleton (probe: 89%).
      Stage 2 (defenses): a defense (Slip/Roll) is small and head-driven, so place
        it on an AirPods peak inside the interval its label position implies —
        BETWEEN two anchors (strike-def-strike) or BEFORE the first anchor
        (def-strike-…, jump zone excluded). No 'after all strikes' case exists
        (probe_defpos: 0). Narrowing the search to one interval removes the C2
        ambiguity that the global DP suffered.

    Window placement/visualization mirror run_spot. Writes combo_anchor_all_p*.png
    (or combo_anchor_anon_p*.png when anon=True: subjects -> p1..pN, no action names)."""
    ANON = anon
    print("# Phase 6 — anchor-first 2-stage (strikes->sensor peaks, then defenses)\n")
    templates = load_single_templates()
    print(f"  templates: {sum(len(v) for v in templates.values())} single windows")

    rows = collect()
    succeeded = 0; total = 0
    viz = []

    def _floor(e):
        m = float(np.median(e)); md = float(np.median(np.abs(e - m))) or 1e-6
        return max(float(np.percentile(e, 62)), m + 0.45 * md)

    for subj, label, n, times, env, peaks, fname, data, jump in rows:
        actions = label.split("-")
        t_full, env_full, dur, _ = bp.motion_envelope(data, 0.0)
        if len(env_full) == 0 or dur < WIN_S:
            total += 1; continue
        src = per_source_envelopes(data)
        if not src:
            total += 1; continue
        # ⑫a effective cut = later of the base cut and the front AirPods cluster end.
        # The AirPods cluster catches jumps the base cut misses (SHIM: base_jump small
        # -> base cut 0, but the front AP cluster still marks the jump).
        cut = max(combo_jump_cut(data, jump),
                  airpods_jump_cut(jump, src, t_full, env_full))

        # per-sensor floor-passing peaks (bin indices), after the jump cut
        def sensor_peaks(sensor):
            e = src.get(sensor)
            if e is None or len(e) != len(env_full):
                return []
            fl = _floor(e)
            return [i for i in bp.local_peak_indices(e)
                    if e[i] >= fl and t_full[i] >= cut]

        # ---- Stage 1: strike anchors via time-ordered DP over dominant-sensor peaks
        strike_pos = [a for a, act in enumerate(actions) if act in STRIKE]
        # candidate (bin, sensor_value) per strike action, sorted by time
        cand = {}
        ok = True
        for a in strike_pos:
            sensor = ACTION_SENSOR.get(actions[a])      # Watch / iPhone
            e = src.get(sensor)
            pk = sensor_peaks(sensor)
            if not pk:
                ok = False; break
            cand[a] = sorted(pk)                        # bins ascending in time
        if not ok:
            total += 1; continue

        # DP: pick one peak per strike, strictly increasing in time, maximizing the
        # sum of that strike's dominant-sensor height (prefer the biggest peaks).
        # ⑫b SENSOR-AWARE separation: two SAME-sensor strikes (Lead Jab-Lead Jab) must
        # be MIN_SEP apart (distinct peaks); two DIFFERENT-sensor strikes (Lead Jab on
        # Watch, Rear Jab on iPhone) may sit close — they overlap in time but are
        # distinct because different sensors drive them (JUHO LJ-RJ: t5.54 Watch +
        # t5.70 iPhone, 0.16s apart). Relax the gap to a small floor in that case.
        same_sep = max(1, int(round(MIN_SEP_S / bp.BIN_S)))
        diff_sep = max(1, int(round(0.10 / bp.BIN_S)))   # different sensors -> 0.10s
        ks = len(strike_pos)
        INF = float("inf")
        chosen_anchor = {}
        # state = (strike index, last bin); also remember the previous strike's sensor
        best = {(-1, -10**9): (0.0, [])}
        prev_sensor = None
        for ai, a in enumerate(strike_pos):
            sensor = ACTION_SENSOR.get(actions[a]); e = src[sensor]
            sep = diff_sep if (prev_sensor is not None and prev_sensor != sensor) else same_sep
            nxt = {}
            for (pidx, lastbin), (score, path) in best.items():
                for b in cand[a]:
                    if b <= lastbin + sep - 1:
                        continue                        # must be later & separated
                    nscore = score + e[b]               # prefer big sensor peaks
                    key = (ai, b)
                    if key not in nxt or nscore > nxt[key][0]:
                        nxt[key] = (nscore, path + [(a, b)])
            if not nxt:
                best = {}; break
            best = nxt
            prev_sensor = sensor
        if not best:
            total += 1; continue
        # best full path = max score among states at last strike
        final = max((v for k, v in best.items() if k[0] == ks - 1),
                    key=lambda x: x[0], default=None)
        if final is None:
            total += 1; continue
        for a, b in final[1]:
            chosen_anchor[a] = b                         # action index -> bin

        # ---- Stage 2: defenses into the interval their label position implies
        ap = src.get("AirPods")
        chosen = dict(chosen_anchor)
        place_ok = True
        def ap_peaks_in(lo_bin, hi_bin):
            if ap is None or len(ap) != len(env_full):
                return []
            fl = _floor(ap)
            return [i for i in bp.local_peak_indices(ap)
                    if fl <= ap[i] and lo_bin < i < hi_bin and t_full[i] >= cut
                    and i not in chosen.values()]
        for a, act in enumerate(actions):
            if act not in DEFENSE:
                continue
            prev_anchor = max((chosen_anchor[s] for s in strike_pos if s < a), default=None)
            next_anchor = min((chosen_anchor[s] for s in strike_pos if s > a), default=None)
            lo = prev_anchor if prev_anchor is not None else int(cut / bp.BIN_S)
            hi = next_anchor if next_anchor is not None else len(env_full) - 1
            cands = ap_peaks_in(lo, hi)
            if cands:
                chosen[a] = max(cands, key=lambda i: ap[i])   # strongest head peak
            else:
                # fallback: combined-envelope peak in the interval
                fl = _floor(env_full)
                cc = [i for i in bp.local_peak_indices(env_full)
                      if env_full[i] >= fl and lo < i < hi and i not in chosen.values()]
                if cc:
                    chosen[a] = max(cc, key=lambda i: env_full[i])
                else:
                    place_ok = False; break
        total += 1
        if not place_ok or len(set(chosen.values())) < n:
            continue
        succeeded += 1
        # convert chosen bins -> window starts (centered like run_spot)
        starts_sel = []
        for a in range(n):
            pi = chosen[a]
            s = max(cut, t_full[pi] - WIN_PRE_S); s = min(s, dur - WIN_S)
            starts_sel.append(max(0.0, s))
        viz.append((subj, label, t_full, env_full, src, starts_sel, actions, cut, jump))

    print(f"  anchor-first placed: {succeeded}/{total} = {succeeded/total*100:.1f}%")

    # ---- visualization (numbered, sensor overlay) ----
    items = sorted(viz, key=lambda r: (r[0], r[1]))
    rep = {}; tot = {}
    for r in items:
        tot[(r[0], r[1])] = tot.get((r[0], r[1]), 0) + 1
    cnt = {}; rep_list = []
    for r in items:
        k = (r[0], r[1]); cnt[k] = cnt.get(k, 0) + 1; rep_list.append(cnt[k])
    cols = ["#7fb3ff", "#84d68a", "#ffd166", "#c792ea"]
    sensors = [("Watch", "#1b9e77"), ("iPhone", "#d95f02"), ("AirPods", "#7570b3")]

    # anonymized subject map: sorted real names -> p1..pN (no names, no action text)
    subj_anon = {s: f"p{i+1}" for i, s in enumerate(sorted({r[0] for r in items}))}

    ncol = 4; PAGE_ROWS = 6; per_page = ncol * PAGE_ROWS
    n_pages = (len(items) + per_page - 1) // per_page
    saved = []
    for pg in range(n_pages):
        chunk = items[pg * per_page:(pg + 1) * per_page]
        chunk_rep = rep_list[pg * per_page:(pg + 1) * per_page]
        nrow = (len(chunk) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 2.3 * nrow), squeeze=False)
        for k, (subj, label, t, env, src, starts_sel, actions, cut, jump) in enumerate(chunk):
            ax = axes[k // ncol][k % ncol]
            ax.plot(t, env, color="#1f4e79", lw=1.0)
            for sname, c in sensors:
                e = src.get(sname)
                if e is not None and len(e) == len(t):
                    ax.plot(t, e, color=c, lw=0.6, alpha=0.5)
            ymax = max(env.max(), 1e-6)
            if cut > 0:
                ax.axvspan(0, cut, color="#ff5555", alpha=0.10)
                ax.axvline(cut, color="#d11", lw=0.8, ls="--")
            for a, s in enumerate(starts_sel):
                ax.axvspan(s, s + WIN_S, color=cols[a % 4], alpha=0.32)
                ax.text(s + WIN_S / 2, ymax * 0.95, actions[a], fontsize=5.5,
                        rotation=90, va="top", ha="center", color="#222")
            ntot = tot[(subj, label)]
            suf = f"  #{chunk_rep[k]}/{ntot}" if ntot > 1 else ""
            # subject anonymized (p1..pN); action labels KEPT
            disp_subj = subj_anon[subj] if ANON else subj
            ax.set_title(f"{disp_subj} | {label}{suf}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        for k in range(len(chunk), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        title = ("Combination segmentation across 5 subjects (anonymized) — "
                 "strike on its sensor peak, defense on a head peak in between; "
                 "shaded = placed windows" if ANON else
                 "Phase 6 ANCHOR-FIRST — page %d/%d "
                 "(strike=sensor peak, defense=AirPods in interval; lines=Watch/iPhone/AirPods)"
                 % (pg + 1, n_pages))
        fig.suptitle(title, fontsize=9, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        stem = "combo_anchor_anon_p" if ANON else "combo_anchor_all_p"
        out = OUT_DIR / f"{stem}{pg+1}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        saved.append(out.name)
    print(f"  saved {len(items)} combos / {n_pages} pages -> {', '.join(saved)}")
    print("# done.")


def main() -> None:
    import sys as _sys
    if "--labelset" in _sys.argv:
        run_labelset(); return
    if "--anchor-anon" in _sys.argv:
        run_anchor(anon=True); return
    if "--anchor" in _sys.argv:
        run_anchor(); return
    if "--spot-diag" in _sys.argv:
        run_spot_diag(); return
    if "--spot" in _sys.argv:
        run_spot(); return
    if "--jumpcut" in _sys.argv:
        run_jumpcut_all(); return
    if "--pattern" in _sys.argv:
        run_pattern(); return
    rows = collect()
    print(f"# Phase 6 — combo segmentation diagnosis (shoe-5, {len(rows)} combo files)\n")

    # ── T1: hit rate (detected peaks == combo length) ──────────────────────────
    by_len = defaultdict(lambda: [0, 0])      # len -> [total, exact]
    over = under = exact = 0
    defense_combos = 0; defense_under = 0
    for subj, label, n, times, env, peaks, fname, _data, _jump in rows:
        d = len(peaks)
        by_len[n][0] += 1
        if d == n: by_len[n][1] += 1; exact += 1
        elif d > n: over += 1
        else: under += 1
        if any(a in DEFENSE for a in label.split("-")):
            defense_combos += 1
            if d < n: defense_under += 1

    print("## T1 — detected-peaks vs combo-length (combo-specific detector)")
    tot = len(rows)
    print(f"  exact (det==len): {exact}/{tot} = {exact/tot*100:.1f}%   over: {over}  under: {under}")
    for L in sorted(by_len):
        t, e = by_len[L]
        print(f"    {L}-action: exact {e}/{t} = {e/t*100:.1f}%")
    if defense_combos:
        print(f"  defense-containing combos under-detected: {defense_under}/{defense_combos} "
              f"= {defense_under/defense_combos*100:.1f}%  (peak miss = label-contamination risk)")
    # distribution of detected counts
    dist = Counter(len(r[5]) for r in rows)   # r[5] = peaks
    print(f"  detected-count distribution: {dict(sorted(dist.items()))}")

    # ── T2: visualization grid (one row per combo type, first occurrence) ──────
    seen = {}
    for subj, label, n, times, env, peaks, fname, _data, _jump in rows:
        if label not in seen:
            seen[label] = (subj, label, n, times, env, peaks)
    items = list(seen.values())
    ncol = 3
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 2.4 * nrow), squeeze=False)
    band_colors = ["#7fb3ff", "#84d68a", "#ffd166", "#c792ea"]
    for k, (subj, label, n, times, env, peaks) in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        ax.plot(times, env, color="#1f4e79", lw=1.2)
        ymax = max(env.max(), 1e-6)
        actions = label.split("-")
        for i, pk in enumerate(peaks):
            ax.axvline(times[pk], color="#d72638", lw=1.0)
            # filename-order label assignment (D1): i-th peak -> i-th action
            lab = actions[i] if i < len(actions) else "?"
            ax.text(times[pk], ymax * 0.95, lab, fontsize=6, rotation=90,
                    va="top", ha="right", color="#333")
        ok = len(peaks) == n
        ax.set_title(f"{subj} | {label}  det={len(peaks)}/{n}" + ("" if ok else "  MISMATCH"),
                     fontsize=8, color="black" if ok else "red")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(items), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Phase 6 combo segmentation — envelope + detected peaks + filename-order labels "
                 "(red=peak; MISMATCH = detected != combo length)", fontsize=11, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUT_DIR / "combo_segmentation_grid.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n## T2 — visualization saved -> {out.name}  ({len(items)} combo types)")
    print("  (combos have NO per-action ground truth -> eyeball whether peaks land on actions.)")
    print("# done.")


if __name__ == "__main__":
    main()
