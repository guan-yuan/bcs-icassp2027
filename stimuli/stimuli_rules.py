#!/usr/bin/env python3
"""Stimulus-grid rules, shared by the stimulus generators and their checks.

The three exclusion rules, the seed scheme and the index-based filename scheme
live in this one module, which both the generators and their post-generation
checks import.

Contents
--------
  * anchor-grid constructors, and a check that the literal grids in
    config/stimuli.yaml equal their formulas
  * `seed_for()`      -- 42 + offset + i1*1e6 + i2*1e4 + i3*1e2 + inst
  * `fname_*()`       -- index-based filenames (no float truncation)
  * `temp_eligible()`   -- temporal eligibility  base_ioi <= 1.5/(2+ratio)
  * `temp_overlap_ok()` -- overlap guard  base_ioi >= tone_duration_s
  * `harm_erb_ok()`     -- harmonicity rule-based ERB cap
  * `timbre_nyquist_ok()` -- timbre Nyquist head-room cap
  * `enumerate_all_seeds()` -- whole-grid enumeration (8530 files)
  * `rerender_*()`    -- rules and enumeration of the re-rendered temporal set
                         (1992 files, stimuli/generate_rerender.py)

Everything here is pure python + numpy-free so it can be imported by any env.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Seed scheme: paradigm offsets 100M apart; base-100 positional code.
# ─────────────────────────────────────────────────────────────────────────────

PARADIGM_OFFSETS: Dict[str, int] = {
    "freq_proximity": 100_000_000,
    "temp_proximity": 200_000_000,
    "harmonicity": 300_000_000,
    "onset_sync": 400_000_000,
    "timbre_ablation": 500_000_000,
    # Re-rendered temporal stimuli: their own 100M block.
    "temp_proximity_rerender": 600_000_000,
}

# Positional weights. Every index field must stay < 100 and inst < 100,
# otherwise the base-100 code overflows into the neighbouring field.
_W1, _W2, _W3 = 1_000_000, 10_000, 100
INDEX_LIMIT = 100


def seed_for(paradigm: str, i1: int, i2: int, i3: int, inst: int,
             seed_base: int = 42) -> int:
    """seed = seed_base + offset(paradigm) + i1*1e6 + i2*1e4 + i3*1e2 + inst.

    frequency/temporal/onset: i1 = anchor idx, i2 = predictor idx, i3 = 0.
    harmonicity: i1 = f0 idx,     i2 = rank idx,     i3 = mistuning idx.
    timbre:     i1 = anchor idx, i2 = timbre idx,   i3 = delta_f idx.
    """
    if paradigm not in PARADIGM_OFFSETS:
        raise KeyError(f"unknown paradigm: {paradigm!r}")
    for name, v in (("i1", i1), ("i2", i2), ("i3", i3), ("inst", inst)):
        if not (0 <= int(v) < INDEX_LIMIT):
            raise ValueError(
                f"seed field {name}={v} out of base-100 range for {paradigm}")
    return int(seed_base + PARADIGM_OFFSETS[paradigm]
               + i1 * _W1 + i2 * _W2 + i3 * _W3 + inst)


# ─────────────────────────────────────────────────────────────────────────────
# Index-based filenames (physical values live in metadata only)
# ─────────────────────────────────────────────────────────────────────────────

def fname_freq(ai: int, dfi: int, inst: int) -> str:
    return f"freq_a{ai:02d}_df{dfi:02d}_inst{inst:02d}.wav"


def fname_temp(bi: int, ri: int, inst: int) -> str:
    return f"temp_a{bi:02d}_r{ri:02d}_inst{inst:02d}.wav"


def fname_harm(f0i: int, rki: int, mi: int, inst: int) -> str:
    return f"harm_f{f0i:02d}_rk{rki:02d}_m{mi:02d}_inst{inst:02d}.wav"


def fname_onset(ci: int, asi: int, inst: int) -> str:
    return f"onset_c{ci:02d}_as{asi:02d}_inst{inst:02d}.wav"


def fname_timbre(ai: int, ti: int, dfi: int, inst: int) -> str:
    return f"timbre_a{ai:02d}_t{ti:02d}_df{dfi:02d}_inst{inst:02d}.wav"


# ─────────────────────────────────────────────────────────────────────────────
# Anchor-grid constructors (each formula reproduces a literal list in the config)
# ─────────────────────────────────────────────────────────────────────────────

# frequency: round(200 * 2**(k/6), 1) for k = -3..20, with k=8 -> 500.0, k=14 -> 1000.0
FREQ_SNAP = {8: 500.0, 14: 1000.0}


def freq_anchor_grid() -> List[float]:
    out = []
    for k in range(-3, 21):
        out.append(FREQ_SNAP[k] if k in FREQ_SNAP else round(200.0 * 2.0 ** (k / 6.0), 1))
    return out


# temporal: round(50 * 4**(k/23)) ms for k = 0..23, with k=12 -> 100 ms
TEMP_SNAP = {12: 100}


def temp_anchor_grid_ms() -> List[int]:
    out = []
    for k in range(24):
        out.append(TEMP_SNAP[k] if k in TEMP_SNAP else int(round(50.0 * 4.0 ** (k / 23.0))))
    return out


# onset: 8 roots x 3 intervals; f2 = root * 2**(st/12)
ONSET_ROOTS: List[Tuple[str, float]] = [
    ("A3", 220.00), ("B3", 246.94), ("C4", 261.63), ("D4", 293.66),
    ("E4", 329.63), ("F4", 349.23), ("G4", 392.00), ("A4", 440.00),
]
ONSET_INTERVALS: List[Tuple[str, int, str]] = [
    ("maj3", 4, "maj3_5:4_consonant"),
    ("p4", 5, "p4_4:3_consonant"),
    ("tritone", 6, "tritone_7:5_dissonant"),
]


def onset_tone_configs() -> List[dict]:
    """24 (root x interval) configs."""
    out = []
    for rname, f1 in ONSET_ROOTS:
        for iname, st, label in ONSET_INTERVALS:
            out.append({
                "name": f"{rname}_{iname}",
                "root": rname,
                "interval": iname,
                "semitones": st,
                "f1": f1,
                "f2": round(f1 * 2.0 ** (st / 12.0), 2),
                "label": label,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The three exclusion rules (+ temporal overlap guard)
# ─────────────────────────────────────────────────────────────────────────────

TEMP_MAX_TRIAL_S = 1.5   # base_ioi <= TEMP_MAX_TRIAL_S / (2 + ratio)


def temp_eligible(base_ioi_s: float, ratio: float) -> bool:
    """Temporal eligibility: the 3-tone trial + long gap must fit the window."""
    return float(base_ioi_s) <= TEMP_MAX_TRIAL_S / (2.0 + float(ratio)) + 1e-12


def temp_overlap_ok(base_ioi_s: float, tone_duration_s: float) -> bool:
    """Temporal overlap guard: successive same-frequency tones must not
    overlap (which would create interference artefacts)."""
    return float(base_ioi_s) >= float(tone_duration_s) - 1e-12


def erb_hz(f_hz: float) -> float:
    """Glasberg & Moore (1990) ERB in Hz."""
    return 24.7 * (4.37 * float(f_hz) / 1000.0 + 1.0)


HARM_ERB_FRACTION = 0.75


def harm_erb_ok(f0: float, rank: int, mist_pct: float) -> bool:
    """Harmonicity rule-based ERB cap:
       delta_f = mist/100 * rank * f0 <= 0.75 * ERB(rank * f0)."""
    if float(mist_pct) == 0.0:
        return True
    delta_f = float(mist_pct) / 100.0 * int(rank) * float(f0)
    return delta_f <= HARM_ERB_FRACTION * erb_hz(int(rank) * float(f0)) + 1e-9


# Nyquist head-room rule: n_max * anchor * 2**(dF_max/12)
# <= 0.95 * fs/2, evaluated with the harmonic_8 partial count (the highest of
# any timbre) so the cap applies uniformly to all timbres.
TIMBRE_NMAX = 8
TIMBRE_HEADROOM = 0.95


def timbre_nyquist_limit_hz(delta_f_max_st: float, fs: int,
                        n_max: int = TIMBRE_NMAX) -> float:
    return (TIMBRE_HEADROOM * fs / 2.0) / (n_max * 2.0 ** (float(delta_f_max_st) / 12.0))


def timbre_nyquist_ok(anchor_hz: float, delta_f_max_st: float, fs: int,
                  n_max: int = TIMBRE_NMAX) -> bool:
    return (n_max * float(anchor_hz) * 2.0 ** (float(delta_f_max_st) / 12.0)
            <= TIMBRE_HEADROOM * fs / 2.0 + 1e-9)


def timbre_anchor_grid(delta_f_semitones: Sequence[float], fs: int) -> List[float]:
    """frequency ladder truncated at the Nyquist rule -> 21 anchors (141.4 .. 1425.4)."""
    df_max = max(float(d) for d in delta_f_semitones)
    return [a for a in freq_anchor_grid() if timbre_nyquist_ok(a, df_max, fs)]


# ─────────────────────────────────────────────────────────────────────────────
# Whole-grid seed / filename enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_all_seeds(cfg: dict) -> Dict[str, List[Tuple[str, int]]]:
    """Enumerate every stimulus the grid will generate.

    Returns {paradigm: [(filename, seed), ...]} applying the three exclusion
    rules exactly as `generate_stimuli.py` does. Used by generate_stimuli.py's
    post-generation check.
    """
    st = cfg["stimuli"]
    seed_base = int(cfg["random_seeds"]["stimuli_base"])
    fs = int(cfg["audio"]["sample_rate"])
    out: Dict[str, List[Tuple[str, int]]] = {}

    # ---- frequency -------------------------------------------------------------
    c = st["freq_proximity"]
    rows = []
    for ai, _a in enumerate(c["anchors_hz"]):
        for dfi, _df in enumerate(c["delta_f_semitones"]):
            for inst in range(int(c["n_instances"])):
                rows.append((fname_freq(ai, dfi, inst),
                             seed_for("freq_proximity", ai, dfi, 0, inst, seed_base)))
    out["freq_proximity"] = rows

    # ---- temporal (eligibility + overlap guard) -------------------------------
    c = st["temp_proximity"]
    tone_dur = float(c["tone_duration_s"])
    rows = []
    for bi, b_ms in enumerate(c["base_ioi_ms_anchors"]):
        b_s = float(b_ms) / 1000.0
        if not temp_overlap_ok(b_s, tone_dur):
            continue
        for ri, ratio in enumerate(c["ioi_ratios"]):
            if not temp_eligible(b_s, ratio):
                continue
            for inst in range(int(c["n_instances"])):
                rows.append((fname_temp(bi, ri, inst),
                             seed_for("temp_proximity", bi, ri, 0, inst, seed_base)))
    out["temp_proximity"] = rows

    # ---- harmonicity (rule-based ERB cap) ----------------------------------------
    c = st["harmonicity"]
    rows = []
    for f0i, f0 in enumerate(c["f0_anchors"]):
        for rki, rank in enumerate(c["mistuned_ranks"]):
            for mi, mist in enumerate(c["mistuning_pcts"]):
                if not harm_erb_ok(f0, rank, mist):
                    continue
                for inst in range(int(c["n_instances"])):
                    rows.append((fname_harm(f0i, rki, mi, inst),
                                 seed_for("harmonicity", f0i, rki, mi, inst, seed_base)))
    out["harmonicity"] = rows

    # ---- onset --------------------------------------------------------------
    c = st["onset_sync"]
    rows = []
    for ci, _tc in enumerate(c["tone_configs"]):
        for asi, _a in enumerate(c["async_ms"]):
            for inst in range(int(c["n_instances"])):
                rows.append((fname_onset(ci, asi, inst),
                             seed_for("onset_sync", ci, asi, 0, inst, seed_base)))
    out["onset_sync"] = rows

    # ---- timbre (Nyquist cap already applied to the anchor list) -------------
    c = st["timbre_ablation"]
    df_max = max(float(d) for d in c["delta_f_semitones"])
    rows = []
    for ai, anchor in enumerate(c["base_freq_hz_anchors"]):
        if not timbre_nyquist_ok(anchor, df_max, fs):
            continue
        for ti, _t in enumerate(c["timbres"]):
            for dfi, _df in enumerate(c["delta_f_semitones"]):
                for inst in range(int(c["n_instances"])):
                    rows.append((fname_timbre(ai, ti, dfi, inst),
                                 seed_for("timbre_ablation", ai, ti, dfi, inst, seed_base)))
    out["timbre_ablation"] = rows
    return out


# Expected counts, asserted by the generator.
EXPECTED_COUNTS = {
    "freq_proximity": 2304,
    "temp_proximity": 1992,
    "harmonicity": 1648,
    "onset_sync": 1536,
    "timbre_ablation": 1050,
}
EXPECTED_TOTAL = 8530

# Expected cluster (anchor) counts per cue.
EXPECTED_N_CLUSTERS = {
    "freq_proximity": 24,
    "temp_proximity": 24,
    "harmonicity": 25,
    "onset_sync": 24,
    "timbre_ablation": 21,
}


# ─────────────────────────────────────────────────────────────────────────────
# Re-rendered temporal stimuli (cue key tp_rr_main)
#
# This block does not change the main grid, its seeds or its filenames.
# ─────────────────────────────────────────────────────────────────────────────

RERENDER_PARADIGM = "temp_proximity_rerender"
RERENDER_CUE = "tp_rr_main"

# Start-position counterbalancing: 8 equally spaced, deterministic offsets
# start_k = 0.04 + 0.03*k, k = instance index. Capacity is always judged at
# start_max so the feasible cell set is identical for every instance.
RERENDER_START_BASE_S = 0.04
RERENDER_START_STEP_S = 0.03
RERENDER_N_INSTANCES = 8
RERENDER_START_MAX_S = 0.25          # = 0.04 + 0.03*7

# Fixed per-tone level. R is the per-tone RMS target; the pure-tone
# amplitude it implies is R*sqrt(2). Relative amplitude jitter is +-0.02/0.3,
# as in the main set.
RERENDER_PER_TONE_RMS = 0.300
RERENDER_AMPLITUDE = RERENDER_PER_TONE_RMS * math.sqrt(2.0)     # 0.42426
RERENDER_AMP_JITTER_REL = 0.02 / 0.3
RERENDER_PEAK_CEILING = 0.99         # every rendered peak must stay below this

RERENDER_EXPECTED_COUNT = 1992


def rerender_start_offset_s(inst: int) -> float:
    """Deterministic, equally spaced start offset for instance `inst`."""
    return round(RERENDER_START_BASE_S + RERENDER_START_STEP_S * int(inst), 2)


def rerender_capacity_ok(base_ioi_ms: float, ratio: float, n_tr: int,
                         tone_duration_s: float, seq_duration_s: float,
                         start_max_s: float = RERENDER_START_MAX_S) -> bool:
    """Capacity rule -- the whole n_tr-trial sequence must fit the clip:

        start_max + (n_tr - 1)*b*(2 + r) + 2*b + tone_dur <= clip
    """
    b = float(base_ioi_ms) / 1000.0
    return (float(start_max_s) + (int(n_tr) - 1) * b * (2.0 + float(ratio))
            + 2.0 * b + float(tone_duration_s)
            <= float(seq_duration_s) + 1e-12)


def rerender_max_n_tr(base_ioi_ms: float, ratio: float, tone_duration_s: float,
                      seq_duration_s: float,
                      start_max_s: float = RERENDER_START_MAX_S) -> int:
    """Largest n_tr the capacity rule admits for one (anchor, ratio) cell."""
    n = 1
    while rerender_capacity_ok(base_ioi_ms, ratio, n + 1, tone_duration_s,
                               seq_duration_s, start_max_s):
        n += 1
    return n


def rerender_n_tr(base_ioi_ms: float, ratios: Sequence[float],
                  tone_duration_s: float, seq_duration_s: float,
                  start_max_s: float = RERENDER_START_MAX_S) -> int:
    """Per-anchor trial count:

        n_tr(a) = max{ n : for every eligible ratio r of anchor a,
                       start_max + (n-1)*b*(2+r) + 2*b + tone_dur <= clip }

    i.e. the minimum over the anchor's eligible ratios of `rerender_max_n_tr`.
    It is constant per anchor, not globally, which keeps all 249 eligible cells.
    """
    b = float(base_ioi_ms) / 1000.0
    elig = [float(r) for r in ratios if temp_eligible(b, r)]
    if not elig:
        raise ValueError(f"anchor {base_ioi_ms} ms has no eligible ratio")
    return min(rerender_max_n_tr(base_ioi_ms, r, tone_duration_s, seq_duration_s,
                                 start_max_s) for r in elig)


def fname_rerender(ai: int, ri: int, inst: int) -> str:
    """Index-based filename (same convention as the main set)."""
    return f"tpr_a{ai:02d}_r{ri:02d}_inst{inst:02d}.wav"


def rerender_enumerate(cfg: dict) -> List[dict]:
    """Enumerate every re-render stimulus in render order: the 249 eligible
    (anchor, ratio) cells of temp_proximity x 8 instances, with the recipe
    (seed, start offset, amplitude, n_tr, tone duration) per clip.

    seed = seed_base + 600e6 + anchor*1e6 + ratio*1e4 + inst.
    """
    c = cfg["stimuli"]["temp_proximity"]
    ratios = [float(r) for r in c["ioi_ratios"]]
    tone_dur = float(c["tone_duration_s"])
    seq_dur = float(c["sequence_duration_s"])
    seed_base = int(cfg["random_seeds"]["stimuli_base"])
    if int(c["n_instances"]) != RERENDER_N_INSTANCES:
        raise AssertionError(f"re-render assumes {RERENDER_N_INSTANCES} instances")
    rows: List[dict] = []
    for ai, b_ms in enumerate(int(b) for b in c["base_ioi_ms_anchors"]):
        b_s = b_ms / 1000.0
        if not temp_overlap_ok(b_s, tone_dur):
            raise AssertionError(f"anchor {b_ms} ms violates the overlap guard")
        n_a = rerender_n_tr(b_ms, ratios, tone_dur, seq_dur)
        for ri, r in enumerate(ratios):
            if not temp_eligible(b_s, r):
                continue
            if not rerender_capacity_ok(b_ms, r, n_a, tone_dur, seq_dur):
                raise AssertionError(f"capacity violated: anchor {b_ms} ms ratio {r}")
            for inst in range(RERENDER_N_INSTANCES):
                rows.append({
                    "file": fname_rerender(ai, ri, inst),
                    "seed": seed_for(RERENDER_PARADIGM, ai, ri, 0, inst, seed_base),
                    "cluster": f"ioi_{b_ms:g}ms", "cluster_idx": ai, "plevel": ri,
                    "base_ioi_ms": b_ms, "base_ioi_s": b_s, "ratio": r,
                    "freq_hz": float(c["freq_hz"]), "instance": inst,
                    "start_offset_s": rerender_start_offset_s(inst), "n_tr": int(n_a),
                    "tone_duration_s": tone_dur, "amplitude": RERENDER_AMPLITUDE,
                })
    if len(rows) != RERENDER_EXPECTED_COUNT:
        raise AssertionError(f"enumerated {len(rows)} != {RERENDER_EXPECTED_COUNT}")
    return rows


def assert_config_grids(cfg: dict) -> List[str]:
    """Check that the literal grids in the config equal the formulas above. Returns a list of human-readable confirmations; raises
    AssertionError on any mismatch."""
    notes = []
    st = cfg["stimuli"]
    fs = int(cfg["audio"]["sample_rate"])

    got = [float(a) for a in st["freq_proximity"]["anchors_hz"]]
    want = freq_anchor_grid()
    assert got == want, f"frequency anchors differ from formula: {got} != {want}"
    notes.append(f"frequency anchors_hz == round(200*2**(k/6),1), k=-3..20 (n={len(got)})")

    got = [int(b) for b in st["temp_proximity"]["base_ioi_ms_anchors"]]
    want = temp_anchor_grid_ms()
    assert got == want, f"temporal anchors differ from formula: {got} != {want}"
    notes.append(f"temporal base_ioi_ms_anchors == round(50*4**(k/23)), k=0..23 (n={len(got)})")

    got = [c["name"] for c in st["onset_sync"]["tone_configs"]]
    want = [c["name"] for c in onset_tone_configs()]
    assert got == want, f"onset configs differ from formula: {got} != {want}"
    for a, b in zip(st["onset_sync"]["tone_configs"], onset_tone_configs()):
        assert abs(float(a["f1"]) - b["f1"]) < 1e-6, f"onset f1 mismatch {a}"
        assert abs(float(a["f2"]) - b["f2"]) < 0.011, f"onset f2 mismatch {a} vs {b}"
    notes.append(f"onset tone_configs == 8 roots x 3 intervals (n={len(got)})")

    got = [float(a) for a in st["timbre_ablation"]["base_freq_hz_anchors"]]
    want = timbre_anchor_grid(st["timbre_ablation"]["delta_f_semitones"], fs)
    assert got == want, f"timbre anchors differ from Nyquist rule: {got} != {want}"
    lim = timbre_nyquist_limit_hz(max(st["timbre_ablation"]["delta_f_semitones"]), fs)
    notes.append(f"timbre base_freq_hz_anchors == frequency ladder <= {lim:.1f} Hz (n={len(got)})")
    return notes


if __name__ == "__main__":  # print the derived grids
    print("frequency", freq_anchor_grid())
    print("temporal", temp_anchor_grid_ms())
    print("onset", [c["name"] for c in onset_tone_configs()])
    print("timbre limit @48k, dF=11.35:", round(timbre_nyquist_limit_hz(11.35, 48000), 2))
