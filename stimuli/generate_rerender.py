"""Generate the re-rendered temporal-proximity stimuli (Sec. 3.3, cue tp_rr_main).

1,992 clips: the 249 eligible (anchor, ratio) cells of temp_proximity x 8
instances. The tones, the cosine ramp, the RNG draw order (amplitude, then
phase, per tone) and the PCM_16 write are those of generate_stimuli.py's
temp_proximity generator. What differs is defined in stimuli_rules.py
(rerender_*): a fixed trial count per anchor (the capacity rule), a fixed
per-tone level, start offsets 0.04 + 0.03*k s, no whole-file RMS
normalisation or peak rescaling, and onsets placed at round(t * sample_rate).

Output: <out>/tp_rr_main/*.wav and <out>/tp_rr_main/metadata.json, the layout
that extraction/extract_musicgen.py (--stim-root) and bcs/score.py (--stimuli)
read.

Usage
  python3 stimuli/generate_rerender.py --out <re-rendered stimulus root>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_stimuli as G  # noqa: E402
import stimuli_rules as SR    # noqa: E402

SAMPLE_RATE = int(G.CONFIG["audio"]["sample_rate"])
TP = G.CONFIG["stimuli"]["temp_proximity"]
SEQ_DUR = float(TP["sequence_duration_s"])
FREQ_HZ = float(TP["freq_hz"])


def render_clip(spec):
    """One clip: n_tr trials of three tones at [0, b, 2b] followed by a gap of b*ratio."""
    b = float(spec["base_ioi_ms"]) / 1000.0
    tone_s = float(spec["tone_duration_s"])
    rng = np.random.RandomState(int(spec["seed"]))
    n_samples = int(SAMPLE_RATE * SEQ_DUR)
    audio = np.zeros(n_samples, dtype=np.float64)
    start_idx = int(round(float(spec["start_offset_s"]) * SAMPLE_RATE))
    n_tone = int(SAMPLE_RATE * tone_s)
    idxs = []
    cursor = 0.0
    for _ in range(int(spec["n_tr"])):
        for k, dt in enumerate([0.0, b, b]):
            cursor += dt if k > 0 else 0.0
            idx = start_idx + int(round(cursor * SAMPLE_RATE))
            if idx + n_tone > n_samples:
                raise AssertionError(f"{spec['file']}: tone at sample {idx} overruns the clip")
            amp = float(spec["amplitude"]) * (
                1.0 + rng.uniform(-SR.RERENDER_AMP_JITTER_REL, SR.RERENDER_AMP_JITTER_REL))
            phase = rng.uniform(0, 2 * np.pi)
            tone = G.generate_tone(FREQ_HZ, tone_s, sr=SAMPLE_RATE, amplitude=amp, phase=phase)
            audio[idx:idx + len(tone)] += tone
            idxs.append(idx)
        cursor += b * float(spec["ratio"])
    for a, c in zip(idxs, idxs[1:]):
        if c - a < n_tone:
            raise AssertionError(f"{spec['file']}: tone windows overlap")
    return audio, idxs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="output root")
    a = ap.parse_args()
    rows = SR.rerender_enumerate(G.CONFIG)
    main_seeds = {s for v in SR.enumerate_all_seeds(G.CONFIG).values() for _, s in v}
    seeds = {r["seed"] for r in rows}
    if len(seeds) != len(rows) or seeds & main_seeds:
        raise SystemExit("re-render seeds are not unique or collide with the main grid")
    out_dir = a.out / SR.RERENDER_CUE
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = []
    for spec in rows:
        audio, idxs = render_clip(spec)
        peak = float(np.abs(audio).max())
        if peak >= SR.RERENDER_PEAK_CEILING:
            raise AssertionError(f"{spec['file']}: peak {peak:.4f} >= {SR.RERENDER_PEAK_CEILING}")
        G.atomic_sf_write(out_dir / spec["file"], audio.astype(np.float32), SAMPLE_RATE)
        e = dict(spec, cue=SR.RERENDER_CUE, amplitude=round(spec["amplitude"], 9))
        e["onsets"] = [{"t": round(i / SAMPLE_RATE, 6), "idx": int(i)} for i in idxs]
        e["peak"] = round(peak, 6)
        stimuli.append(e)
    info = {"cue": SR.RERENDER_CUE, "sample_rate": SAMPLE_RATE,
            "sequence_duration_s": SEQ_DUR, "freq_hz": FREQ_HZ,
            "seed_scheme": "seed_base + 600000000 + anchor*1e6 + ratio*1e4 + inst",
            "per_tone_rms_target": SR.RERENDER_PER_TONE_RMS,
            "amplitude_jitter_relative": SR.RERENDER_AMP_JITTER_REL,
            "normalisation": "none"}
    G.atomic_json_dump(out_dir / "metadata.json", {"_info": info, "stimuli": stimuli})
    if {p.name for p in out_dir.glob("*.wav")} != {r["file"] for r in rows}:
        raise SystemExit(f"{out_dir}: WAV files differ from the enumeration")
    print(f"{SR.RERENDER_CUE}: {len(stimuli)} stimuli -> {out_dir}")


if __name__ == "__main__":
    main()
