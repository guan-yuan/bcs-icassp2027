"""Generate the synthetic stimulus set (four cues plus a timbre set not used in the paper).

  freq_proximity  24 anchors x 12 dF   x 8 inst              = 2304
  temp_proximity  24 anchors x 11 rate x 8 inst - 15 cells*8 = 1992
  harmonicity     25 (f0,rank) x 9 mist x 8 inst - ERB cap   = 1648
  onset_sync      24 configs x 8 async x 8 inst              = 1536
  timbre_ablation 21 anchors x 5 timbre x 5 dF x 2 inst      = 1050
                                                     total = 8530

Seeds: 42 + offset(100M) + i1*1e6 + i2*1e4 + i3*1e2 + inst (stimuli_rules.seed_for);
filenames are index-based (stimuli_rules.fname_*), physical values live in the
metadata; exclusion rules (temporal eligibility, overlap guard, harmonicity ERB
cap, timbre Nyquist cap) are shared with stimuli_rules.

Fully deterministic given `random_seeds.stimuli_base` in config/stimuli.yaml.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))
from extract_common import atomic_json_dump, git_commit
sys.path.insert(0, str(Path(__file__).resolve().parent))   # stimuli/stimuli_rules.py
import stimuli_rules as SR

# ─────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────
import os as _os
CONFIG_PATH = Path(_os.environ.get(
    "EXPERIMENT_CONFIG", Path(__file__).resolve().parents[1] / "config" / "stimuli.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = Path(__file__).parent / CONFIG_PATH
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

if "base_freq_hz_anchors" not in CONFIG["stimuli"]["timbre_ablation"]:
    raise SystemExit(
        f"[generate_stimuli] {CONFIG_PATH} lacks "
        "stimuli.timbre_ablation.base_freq_hz_anchors; "
        "point EXPERIMENT_CONFIG at config/stimuli.yaml.")

SAMPLE_RATE = CONFIG["audio"]["sample_rate"]        # 48000
FADE_MS = CONFIG["audio"]["fade_ms"]                # 20 ms cosine ramp
AMP_BASELINE = CONFIG["audio"]["amplitude_baseline"]
AMP_JITTER = CONFIG["audio"]["amplitude_jitter"]
SEED_BASE = CONFIG["random_seeds"]["stimuli_base"]

# Paradigm seed offsets live in stimuli_rules (100M spacing).
PARADIGM_OFFSETS = SR.PARADIGM_OFFSETS

GIT_COMMIT = git_commit()

# ─────────────────────────────────────────────────────────────────────────
# Atomic WAV write (soundfile-specific)
# ─────────────────────────────────────────────────────────────────────────
def atomic_sf_write(path, audio, sr):
    """POSIX-atomic WAV write: write to .tmp, then rename."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sf.write(str(tmp), audio, sr, format="WAV", subtype="PCM_16")
    tmp.rename(path)

def _metadata_header(paradigm, fade_ms=None, extra=None):
    prov = {
        "paradigm": paradigm,
        "config_version": CONFIG.get("version", "unknown"),
        "config_path": str(CONFIG_PATH),
        "git_commit": GIT_COMMIT,
        "sample_rate": SAMPLE_RATE,
        "fade_ms": FADE_MS if fade_ms is None else fade_ms,
        "seed_base": SEED_BASE,
        "seed_scheme": ("seed_base + offset(100M) + i1*1e6 + i2*1e4 + "
                        "i3*1e2 + inst"),
        "paradigm_offset": SR.PARADIGM_OFFSETS[paradigm],
        "generator_script": str(Path(__file__).relative_to(Path(__file__).parents[2])),
    }
    if extra:
        prov.update(extra)
    return prov

# ─────────────────────────────────────────────────────────────────────────
# Core DSP functions
# ─────────────────────────────────────────────────────────────────────────
def _apply_fade(audio, sr=SAMPLE_RATE, fade_ms=FADE_MS):
    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples > 0 and len(audio) > 2 * fade_samples:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        audio[:fade_samples] *= ramp
        audio[-fade_samples:] *= ramp[::-1]
    return audio

def _rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

def _rms_normalize(audio, target_rms):
    r = _rms(audio)
    if r < 1e-12:
        return audio
    return (audio * (target_rms / r)).astype(audio.dtype)

def generate_tone(freq, duration, sr=SAMPLE_RATE, amplitude=0.3, fade_ms=FADE_MS,
                  phase=0.0):
    """Pure sinusoidal tone with cosine fade."""
    t = np.arange(int(sr * duration)) / sr
    tone = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return _apply_fade(tone, sr, fade_ms)

def generate_complex_tone(f0, n_harmonics, duration, sr=SAMPLE_RATE,
                          amplitude=0.3, mistuning_pct=None, mistuned_rank=3,
                          phase_mode="zero", rng=None):
    """Complex tone with 1/n amplitude rolloff. Optionally mistune one harmonic."""
    t = np.arange(int(sr * duration)) / sr
    tone = np.zeros_like(t)
    if phase_mode == "random" and rng is None:
        rng = np.random.RandomState(0)
    for h in range(1, n_harmonics + 1):
        f = f0 * h
        if mistuning_pct is not None and h == mistuned_rank:
            f *= (1.0 + mistuning_pct / 100.0)
        phase = rng.uniform(0, 2 * np.pi) if phase_mode == "random" else 0.0
        tone += (1.0 / h) * np.sin(2 * np.pi * f * t + phase)
    peak = np.max(np.abs(tone)) + 1e-12
    tone = tone / peak * amplitude
    return _apply_fade(tone, sr)

def generate_harmonic_tone(f0, n_harmonics, duration, sr=SAMPLE_RATE,
                           amplitude=0.3, phase=0.0):
    t = np.arange(int(sr * duration)) / sr
    tone = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        tone += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + phase)
    peak = np.max(np.abs(tone)) + 1e-12
    tone = tone / peak * amplitude
    return _apply_fade(tone, sr)

def generate_stretched_tone(f0, partials, duration, sr=SAMPLE_RATE, amplitude=0.3):
    """Non-harmonic tone with arbitrary partial ratios."""
    t = np.arange(int(sr * duration)) / sr
    tone = np.zeros_like(t)
    for i, p in enumerate(partials):
        tone += (1.0 / (i + 1)) * np.sin(2 * np.pi * f0 * p * t)
    peak = np.max(np.abs(tone)) + 1e-12
    tone = tone / peak * amplitude
    return _apply_fade(tone, sr)

def generate_band_noise_tone(center_freq, bandwidth_octaves, duration,
                             sr=SAMPLE_RATE, amplitude=0.3, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    n_samples = int(sr * duration)
    noise = rng.randn(n_samples)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    f_lo = center_freq * 2 ** (-bandwidth_octaves / 2.0)
    f_hi = center_freq * 2 ** (+bandwidth_octaves / 2.0)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    spec[~mask] = 0
    tone = np.fft.irfft(spec, n=n_samples)
    peak = np.max(np.abs(tone)) + 1e-12
    tone = tone / peak * amplitude
    return _apply_fade(tone, sr)

# ─────────────────────────────────────────────────────────────────────────
# Frequency proximity  (cluster = anchor_hz, 24 clusters)
# ─────────────────────────────────────────────────────────────────────────
def generate_freq_proximity_stimuli(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = CONFIG["stimuli"]["freq_proximity"]
    anchors = cfg["anchors_hz"]
    delta_fs = cfg["delta_f_semitones"]
    tone_dur = cfg["tone_duration_s"]
    ioi = cfg["ioi_s"]
    seq_dur = cfg["sequence_duration_s"]
    n_inst = cfg["n_instances"]
    target_rms = cfg["target_rms"]

    metadata = {"_info": _metadata_header("freq_proximity"),
                "config": cfg, "stimuli": []}
    same_A_within_pattern = bool(cfg.get("same_A_within_pattern", True))

    for ai, f_a in enumerate(anchors):
        for dfi, delta_f in enumerate(delta_fs):
            f_b = f_a * (2.0 ** (delta_f / 12.0))
            for inst in range(n_inst):
                seed = SR.seed_for("freq_proximity", ai, dfi, 0, inst, SEED_BASE)
                rng = np.random.RandomState(seed)

                pattern_dur = 4 * ioi  # A-B-A-silence
                n_patterns = int(seq_dur / pattern_dur)
                audio = np.zeros(int(SAMPLE_RATE * seq_dur), dtype=np.float64)
                onsets = []
                for p in range(n_patterns):
                    base_t = p * pattern_dur
                    if same_A_within_pattern:
                        # both A tones within a pattern share identical phase +
                        # amplitude; B gets independent jitter (van Noorden ABA-)
                        amp_A = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                        phase_A = rng.uniform(0, 2 * np.pi)
                        amp_B = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                        phase_B = rng.uniform(0, 2 * np.pi)
                        tone_params = [
                            (f_a, "A", amp_A, phase_A),
                            (f_b, "B", amp_B, phase_B),
                            (f_a, "A", amp_A, phase_A),
                        ]
                    else:
                        tone_params = [
                            (f_a, "A",
                             AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER),
                             rng.uniform(0, 2 * np.pi)),
                            (f_b, "B",
                             AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER),
                             rng.uniform(0, 2 * np.pi)),
                            (f_a, "A",
                             AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER),
                             rng.uniform(0, 2 * np.pi)),
                        ]
                    for k, (freq, stream, amp, phase) in enumerate(tone_params):
                        t_on = base_t + k * ioi
                        tone = generate_tone(freq, tone_dur, amplitude=amp, phase=phase)
                        idx = int(t_on * SAMPLE_RATE)
                        audio[idx:idx + len(tone)] += tone
                        onsets.append({"t": round(t_on, 4), "freq": round(freq, 2),
                                       "stream": stream})
                audio = _rms_normalize(audio, target_rms)
                peak = np.max(np.abs(audio))
                if peak > 0.99:
                    audio = audio * (0.99 / peak)

                fname = SR.fname_freq(ai, dfi, inst)
                atomic_sf_write(Path(output_dir) / fname, audio.astype(np.float32),
                                SAMPLE_RATE)
                metadata["stimuli"].append({
                    "file": fname, "paradigm": "freq_proximity",
                    "cluster": f"anchor_{f_a:g}", "cluster_idx": ai,
                    "plevel": dfi,
                    "anchor_hz": f_a, "delta_f": delta_f, "f_a": f_a,
                    "f_b": round(f_b, 3), "instance": inst, "seed": seed,
                    "onsets": onsets, "rms": round(_rms(audio), 6),
                    "expected_ratio": round(f_b / f_a, 6),
                })
    atomic_json_dump(Path(output_dir) / "metadata.json", metadata)
    print(f"freq_proximity: {len(metadata['stimuli'])} stimuli → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Temporal proximity  (cluster = base_ioi, 24 clusters)
# ─────────────────────────────────────────────────────────────────────────
def generate_temp_proximity_stimuli(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = CONFIG["stimuli"]["temp_proximity"]
    base_iois_ms = cfg["base_ioi_ms_anchors"]
    ratios = cfg["ioi_ratios"]
    freq = cfg["freq_hz"]
    tone_dur = cfg["tone_duration_s"]
    seq_dur = cfg["sequence_duration_s"]
    n_inst = cfg["n_instances"]
    target_rms = cfg["target_rms"]

    skipped = []
    metadata = {"_info": _metadata_header("temp_proximity"),
                "config": cfg, "stimuli": []}

    for bi, base_ms in enumerate(base_iois_ms):
        base_ioi = float(base_ms) / 1000.0
        # Overlap guard: successive tones at the same frequency must not
        # overlap. The 50 ms lower bound == tone_duration_s makes this
        # hold with equality at the fastest anchor.
        if not SR.temp_overlap_ok(base_ioi, tone_dur):
            raise AssertionError(
                f"[temp_proximity] anchor {base_ms} ms < tone_duration_s {tone_dur*1000:g} ms "
                "violates the overlap guard")
        for ri, ratio in enumerate(ratios):
            # Eligibility: ineligible cells are not generated
            if not SR.temp_eligible(base_ioi, ratio):
                skipped.append({"anchor_ms": base_ms, "ratio": ratio,
                                "limit_ms": round(1000 * SR.TEMP_MAX_TRIAL_S / (2 + ratio), 3),
                                "reason": "temp_eligibility base_ioi > 1.5/(2+ratio)"})
                continue
            long_ioi = base_ioi * ratio
            # One trial = [0, base_ioi, 2*base_ioi] + long gap
            trial_dur = 2 * base_ioi + long_ioi
            n_trials = int(seq_dur / trial_dur)
            if n_trials < 1:
                n_trials = 1
            for inst in range(n_inst):
                seed = SR.seed_for("temp_proximity", bi, ri, 0, inst, SEED_BASE)
                rng = np.random.RandomState(seed)
                audio = np.zeros(int(SAMPLE_RATE * seq_dur), dtype=np.float64)
                onsets = []
                cursor = 0.0
                for _ in range(n_trials):
                    for k, dt in enumerate([0, base_ioi, base_ioi]):
                        cursor += dt if k > 0 else 0
                        if cursor + tone_dur > seq_dur:
                            break
                        amp = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                        phase = rng.uniform(0, 2 * np.pi)
                        tone = generate_tone(freq, tone_dur, amplitude=amp, phase=phase)
                        idx = int(cursor * SAMPLE_RATE)
                        audio[idx:idx + len(tone)] += tone
                        onsets.append({"t": round(cursor, 4)})
                    cursor += long_ioi
                audio = _rms_normalize(audio, target_rms)
                peak = np.max(np.abs(audio))
                if peak > 0.99:
                    audio = audio * (0.99 / peak)
                fname = SR.fname_temp(bi, ri, inst)
                atomic_sf_write(Path(output_dir) / fname, audio.astype(np.float32),
                                SAMPLE_RATE)
                metadata["stimuli"].append({
                    "file": fname, "paradigm": "temp_proximity",
                    "cluster": f"ioi_{base_ms:g}ms", "cluster_idx": bi,
                    "plevel": ri,
                    "base_ioi_ms": base_ms, "base_ioi_s": base_ioi,
                    "ratio": ratio, "freq_hz": freq,
                    "instance": inst, "seed": seed, "n_tones": len(onsets),
                    "n_trials": n_trials,
                    "onsets": onsets, "rms": round(_rms(audio), 6),
                })
    metadata["_info"]["skipped_cells"] = skipped
    metadata["_info"]["n_skipped_cells"] = len(skipped)
    atomic_json_dump(Path(output_dir) / "metadata.json", metadata)
    print(f"temp_proximity: {len(metadata['stimuli'])} stimuli "
          f"({len(skipped)} ineligible cells skipped) → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Harmonicity  (cluster = f0 x rank, 25 clusters; rule-based ERB cap)
# ─────────────────────────────────────────────────────────────────────────
def generate_harmonicity_stimuli(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = CONFIG["stimuli"]["harmonicity"]
    f0s = cfg["f0_anchors"]
    ranks = cfg["mistuned_ranks"]
    mists = cfg["mistuning_pcts"]
    n_harm = cfg["n_harmonics"]
    tone_dur = cfg["tone_duration_s"]
    n_inst = cfg["n_instances"]
    target_rms = cfg["target_rms"]
    phase_mode = cfg.get("phase", "zero")

    skipped = []
    metadata = {"_info": _metadata_header("harmonicity"),
                "config": cfg, "stimuli": []}

    for f0i, f0 in enumerate(f0s):
        for rki, rank in enumerate(ranks):
            for mi, mist in enumerate(mists):
                # Rule-based ERB cap (shared with stimuli_rules)
                if not SR.harm_erb_ok(f0, rank, mist):
                    skipped.append({
                        "f0": f0, "rank": rank, "mist_pct": mist,
                        "delta_f_hz": round(mist / 100.0 * rank * f0, 4),
                        "erb_limit_hz": round(SR.HARM_ERB_FRACTION
                                              * SR.erb_hz(rank * f0), 4),
                        "reason": "harm_erb_cap"})
                    continue
                for inst in range(n_inst):
                    seed = SR.seed_for("harmonicity", f0i, rki, mi, inst, SEED_BASE)
                    rng = np.random.RandomState(seed)
                    tone = generate_complex_tone(
                        f0, n_harm, tone_dur, amplitude=AMP_BASELINE,
                        mistuning_pct=(mist if mist > 0 else None),
                        mistuned_rank=rank, phase_mode=phase_mode, rng=rng,
                    )
                    tone = _rms_normalize(tone, target_rms)
                    peak = np.max(np.abs(tone))
                    if peak > 0.99:
                        tone = tone * (0.99 / peak)
                    fname = SR.fname_harm(f0i, rki, mi, inst)
                    atomic_sf_write(Path(output_dir) / fname,
                                    tone.astype(np.float32), SAMPLE_RATE)
                    metadata["stimuli"].append({
                        "file": fname, "paradigm": "harmonicity",
                        "cluster": f"f0_{f0:g}_rk{rank}", "cluster_idx": f0i * 100 + rki,
                        "plevel": mi,
                        "f0": f0, "mistuned_rank": rank, "mistuning_pct": mist,
                        "n_harmonics": n_harm, "instance": inst, "seed": seed,
                        "phase": phase_mode, "rms": round(_rms(tone), 6),
                    })
    metadata["_info"]["skipped_cells"] = skipped
    metadata["_info"]["n_skipped_cells"] = len(skipped)
    atomic_json_dump(Path(output_dir) / "metadata.json", metadata)
    print(f"harmonicity: {len(metadata['stimuli'])} stimuli "
          f"({len(skipped)} ERB-capped cells skipped) → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Onset synchrony  (cluster = tone_config, 24 clusters)
# ─────────────────────────────────────────────────────────────────────────
def generate_onset_sync_stimuli(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = CONFIG["stimuli"]["onset_sync"]
    configs = cfg["tone_configs"]
    async_ms_list = cfg["async_ms"]
    tone_dur = cfg["tone_duration_s"]
    total_dur = cfg["total_duration_s"]
    n_inst = cfg["n_instances"]
    target_rms = cfg["target_rms"]
    # per-paradigm fade override (2 ms) removes the async x ramp-duration confound
    fade_ms_override = float(cfg.get("fade_ms_override", FADE_MS))

    metadata = {"_info": _metadata_header("onset_sync",
                                                  fade_ms=fade_ms_override),
                "config": cfg, "stimuli": []}

    for ci, tc in enumerate(configs):
        for asi, async_ms in enumerate(async_ms_list):
            async_s = async_ms / 1000.0
            for inst in range(n_inst):
                seed = SR.seed_for("onset_sync", ci, asi, 0, inst, SEED_BASE)
                rng = np.random.RandomState(seed)
                audio = np.zeros(int(SAMPLE_RATE * total_dur), dtype=np.float64)
                phase1 = rng.uniform(0, 2 * np.pi)
                phase2 = rng.uniform(0, 2 * np.pi)
                amp1 = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                amp2 = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                tone1 = generate_tone(tc["f1"], tone_dur, amplitude=amp1, phase=phase1,
                                      fade_ms=fade_ms_override)
                tone2 = generate_tone(tc["f2"], tone_dur, amplitude=amp2, phase=phase2,
                                      fade_ms=fade_ms_override)
                audio[:len(tone1)] += tone1
                idx2 = int(async_s * SAMPLE_RATE)
                audio[idx2:idx2 + len(tone2)] += tone2
                audio = _rms_normalize(audio, target_rms)
                peak = np.max(np.abs(audio))
                if peak > 0.99:
                    audio = audio * (0.99 / peak)
                fname = SR.fname_onset(ci, asi, inst)
                atomic_sf_write(Path(output_dir) / fname, audio.astype(np.float32),
                                SAMPLE_RATE)
                metadata["stimuli"].append({
                    "file": fname, "paradigm": "onset_sync",
                    "cluster": tc["name"], "cluster_idx": ci, "plevel": asi,
                    "config": tc["name"], "f1": tc["f1"], "f2": tc["f2"],
                    "root": tc.get("root"), "interval": tc.get("interval"),
                    "async_ms": async_ms, "label": tc["label"],
                    "instance": inst, "seed": seed, "rms": round(_rms(audio), 6),
                })
    atomic_json_dump(Path(output_dir) / "metadata.json", metadata)
    print(f"onset_sync: {len(metadata['stimuli'])} stimuli → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Timbre x dF  (cluster = base_freq anchor, 21 clusters)
# ─────────────────────────────────────────────────────────────────────────
def _render_timbre(timbre_spec, freq, duration, rng):
    """Render a tone of specified timbre at unit RMS; caller RMS-normalizes."""
    t = timbre_spec["type"]
    if t == "sine":
        return generate_tone(freq, duration, amplitude=1.0)
    if t == "harmonic":
        return generate_harmonic_tone(freq, timbre_spec["n"], duration, amplitude=1.0)
    if t == "stretched":
        return generate_stretched_tone(freq, timbre_spec["partials"], duration,
                                       amplitude=1.0)
    if t == "band_noise":
        return generate_band_noise_tone(freq, timbre_spec["bandwidth_octaves"],
                                        duration, amplitude=1.0, rng=rng)
    raise ValueError(f"Unknown timbre type: {t}")

def generate_timbre_ablation_stimuli(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg = CONFIG["stimuli"]["timbre_ablation"]
    anchors = cfg["base_freq_hz_anchors"]          # anchor list
    timbres = cfg["timbres"]
    delta_fs = cfg["delta_f_semitones"]
    tone_dur = cfg["tone_duration_s"]
    ioi = cfg["ioi_s"]
    seq_dur = cfg["sequence_duration_s"]
    n_inst = cfg["n_instances"]
    target_rms = cfg["target_rms"]
    df_max = float(max(delta_fs))

    metadata = {"_info": _metadata_header("timbre_ablation"),
                "config": cfg, "stimuli": []}
    same_A_within_pattern = bool(cfg.get("same_A_within_pattern", True))
    skipped = []

    for ai, base_freq in enumerate(anchors):
        # Timbre Nyquist cap (applies uniformly to every timbre)
        if not SR.timbre_nyquist_ok(base_freq, df_max, SAMPLE_RATE):
            skipped.append({"anchor_hz": base_freq, "reason": "timbre_nyquist_cap"})
            continue
        for ti, timbre in enumerate(timbres):
            for dfi, delta_f in enumerate(delta_fs):
                f_a = base_freq
                f_b = base_freq * 2.0 ** (delta_f / 12.0)
                for inst in range(n_inst):
                    seed = SR.seed_for("timbre_ablation", ai, ti, dfi, inst, SEED_BASE)
                    rng = np.random.RandomState(seed)
                    pattern_dur = 4 * ioi
                    n_patterns = int(seq_dur / pattern_dur)
                    audio = np.zeros(int(SAMPLE_RATE * seq_dur), dtype=np.float64)
                    onsets = []
                    for p in range(n_patterns):
                        base_t = p * pattern_dur
                        if same_A_within_pattern:
                            amp_scale_A = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                            amp_scale_B = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                            tone_A_raw = _render_timbre(timbre, f_a, tone_dur, rng)
                            tone_B_raw = _render_timbre(timbre, f_b, tone_dur, rng)
                            tone_A = _rms_normalize(tone_A_raw, AMP_BASELINE) * (amp_scale_A / AMP_BASELINE)
                            tone_B = _rms_normalize(tone_B_raw, AMP_BASELINE) * (amp_scale_B / AMP_BASELINE)
                            pattern_tones = [
                                (f_a, "A", tone_A),
                                (f_b, "B", tone_B),
                                (f_a, "A", tone_A),  # reuse identical A waveform
                            ]
                        else:
                            pattern_tones = []
                            for (freq, stream) in [(f_a, "A"), (f_b, "B"), (f_a, "A")]:
                                amp_scale = AMP_BASELINE + rng.uniform(-AMP_JITTER, AMP_JITTER)
                                raw = _render_timbre(timbre, freq, tone_dur, rng)
                                tone = _rms_normalize(raw, AMP_BASELINE) * (amp_scale / AMP_BASELINE)
                                pattern_tones.append((freq, stream, tone))
                        for k, (freq, stream, tone) in enumerate(pattern_tones):
                            t_on = base_t + k * ioi
                            idx = int(t_on * SAMPLE_RATE)
                            audio[idx:idx + len(tone)] += tone
                            onsets.append({"t": round(t_on, 4), "freq": round(freq, 2),
                                           "stream": stream})
                    audio = _rms_normalize(audio, target_rms)
                    peak = np.max(np.abs(audio))
                    if peak > 0.99:
                        audio = audio * (0.99 / peak)
                    fname = SR.fname_timbre(ai, ti, dfi, inst)
                    atomic_sf_write(Path(output_dir) / fname,
                                    audio.astype(np.float32), SAMPLE_RATE)
                    metadata["stimuli"].append({
                        "file": fname, "paradigm": "timbre_ablation",
                        "cluster": f"anchor_{base_freq:g}", "cluster_idx": ai,
                        "plevel": dfi,
                        "base_freq_hz": base_freq, "anchor_hz": base_freq,
                        "timbre": timbre["id"], "timbre_spec": timbre,
                        "delta_f": delta_f, "f_a": f_a, "f_b": round(f_b, 2),
                        "instance": inst, "seed": seed, "rms": round(_rms(audio), 6),
                        "onsets": onsets,
                    })
    metadata["_info"]["skipped_cells"] = skipped
    metadata["_info"]["nyquist_limit_hz"] = round(
        SR.timbre_nyquist_limit_hz(df_max, SAMPLE_RATE), 3)
    atomic_json_dump(Path(output_dir) / "metadata.json", metadata)
    print(f"timbre_ablation: {len(metadata['stimuli'])} stimuli → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Post-generation assertions (seeds, filenames, counts)
# ─────────────────────────────────────────────────────────────────────────
def assert_grid_integrity(out_root: Path, targets) -> dict:
    """Re-run the whole-grid enumeration and compare against what is on disk."""
    expected = SR.enumerate_all_seeds(CONFIG)
    report = {"paradigms": {}, "failures": []}
    all_seeds, all_names = [], []
    for name in targets:
        exp_rows = expected[name]
        exp_names = {n for n, _ in exp_rows}
        exp_seeds = [s for _, s in exp_rows]
        meta_path = out_root / name / "metadata.json"
        got_rows = json.loads(meta_path.read_text())["stimuli"]
        got_names = {r["file"] for r in got_rows}
        got_seeds = [int(r["seed"]) for r in got_rows]
        wavs = {p.name for p in (out_root / name).glob("*.wav")}
        entry = {
            "n_metadata": len(got_rows), "n_wav": len(wavs),
            "n_expected": len(exp_rows),
            "n_unique_seeds": len(set(got_seeds)),
        }
        if len(got_rows) != len(exp_rows):
            report["failures"].append(
                f"[{name}] metadata rows {len(got_rows)} != enumerated {len(exp_rows)}")
        if got_names != exp_names:
            miss = sorted(exp_names - got_names)[:3]
            extra = sorted(got_names - exp_names)[:3]
            report["failures"].append(
                f"[{name}] filename set mismatch (missing {miss}, extra {extra})")
        if wavs != got_names:
            report["failures"].append(
                f"[{name}] WAVs on disk {len(wavs)} != metadata {len(got_names)}")
        if sorted(got_seeds) != sorted(exp_seeds):
            report["failures"].append(f"[{name}] seed set differs from enumeration")
        if len(set(got_seeds)) != len(got_seeds):
            report["failures"].append(f"[{name}] duplicate seeds on disk")
        if name in SR.EXPECTED_COUNTS and len(got_rows) != SR.EXPECTED_COUNTS[name]:
            report["failures"].append(
                f"[{name}] count {len(got_rows)} != expected "
                f"{SR.EXPECTED_COUNTS[name]}")
        report["paradigms"][name] = entry
        all_seeds += got_seeds
        all_names += list(got_names)
    report["total"] = len(all_seeds)
    report["n_unique_seeds_global"] = len(set(all_seeds))
    report["n_unique_names_global"] = len(set(all_names))
    if len(set(all_seeds)) != len(all_seeds):
        report["failures"].append("GLOBAL duplicate seeds")
    if len(set(all_names)) != len(all_names):
        report["failures"].append("GLOBAL duplicate filenames")
    if len(targets) == len(PARADIGMS) and report["total"] != SR.EXPECTED_TOTAL:
        report["failures"].append(
            f"GLOBAL total {report['total']} != {SR.EXPECTED_TOTAL}")
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    return report

# ─────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────
PARADIGMS = {
    "freq_proximity": generate_freq_proximity_stimuli,
    "temp_proximity": generate_temp_proximity_stimuli,
    "harmonicity": generate_harmonicity_stimuli,
    "onset_sync": generate_onset_sync_stimuli,
    "timbre_ablation": generate_timbre_ablation_stimuli,
}

def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic ASA cue stimuli.")
    parser.add_argument("--out", required=True, help="output root")
    parser.add_argument("--only", nargs="*", default=None,
                        help=f"Run only these paradigms. Options: {list(PARADIGMS)}")
    parser.add_argument("--all", action="store_true", help="Run all paradigms")
    parser.add_argument("--assert-only", action="store_true",
                        help="Skip generation; only re-run the integrity assertions")
    parser.add_argument("--report", default=None,
                        help="Write the integrity report JSON here")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = [(k, PARADIGMS[k]) for k in args.only if k in PARADIGMS]
    else:
        targets = list(PARADIGMS.items())

    print(f"[generate_stimuli] config={CONFIG_PATH} version={CONFIG.get('version')}")
    print(f"Generating {len(targets)} paradigm(s) at SR={SAMPLE_RATE}, fade={FADE_MS}ms")
    SR.assert_config_grids(CONFIG)
    if not args.assert_only:
        for name, fn in targets:
            fn(str(out_root / name))

    rep = assert_grid_integrity(out_root, [k for k, _ in targets])
    print(json.dumps(rep, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2))
    if rep["failures"]:
        raise SystemExit("STIMULUS GRID INTEGRITY: FAIL")
    print("\nStimulus generation complete; grid integrity PASS.")

if __name__ == "__main__":
    main()
