#!/usr/bin/env python3
"""Extract the fixed DSP front-end references (log-mel, CQT, gammatone).

Produces, on the identical ASA probe stimuli that the learned models are scored
on, three fixed (non-learned) spectro-temporal front-ends:

  * ``logmel``    128-band log-mel spectrogram (power_to_db)
  * ``cqt``       192-bin constant-Q transform magnitude (24 bins/octave), in dB
  * ``gammatone`` 64-band ERB-spaced gammatone cochleagram energy, in dB

Each feature is emitted in the layout ``bcs/score.py`` consumes, i.e. an
``.npz`` keyed by stimulus id with a single-"layer" array of shape ``[1, T, H]``
(``layers_sample = [0]``). BCS is representation-agnostic, so the identical
estimator scores these baselines and the learned models, and the DSP scores set,
per cue, the cue score attainable from fixed spectro-temporal structure with
no learning.

The paradigm list and audio sample rate are read from the config.
The feature geometry is fixed to match the scored NPZ layout and is
documented as module constants below (hop 640 -> T = 226 frames on 48 kHz, 3.0 s
mono WAV; log-mel 128 / CQT 192 (24 bins/oct) / gammatone 64 ERB bands; 80 dB
floor (log-mel, CQT)). They are set in this file, not in the config.

CLI:
  python3 extract_dsp_frontends.py --stim_root <stimulus root> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict

import multiprocessing as mp

import numpy as np
import soundfile as sf
import librosa
import scipy.signal
import scipy.fft

N_WORKERS = max(1, min(14, (os.cpu_count() or 2) - 2))

from extract_common import load_config, get_stim_paradigms

# Fixed DSP front-end geometry (48 kHz, hop 640 -> 226 frames per 3 s clip;
# H = 128 / 192 / 64). SR is read from the config.
HOP = 640
N_FFT = 2048
N_MELS = 128
CQT_BINS = 192
CQT_BPO = 24
CQT_FMIN = 32.70319566   # C1
GT_BANDS = 64
GT_FMIN = 50.0
TOP_DB = 80.0   # 80 dB floor (log-mel, CQT; librosa default)


def _frame_energy(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    """Framewise RMS energy with centre padding (vectorised, librosa framing)."""
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")
    n_frames = 1 + len(x) // hop
    frames = librosa.util.frame(xp, frame_length=win, hop_length=hop)  # [win, F]
    e = np.sqrt(np.mean(frames * frames, axis=0) + 1e-12)
    if e.shape[0] >= n_frames:
        return e[:n_frames]
    return np.pad(e, (0, n_frames - e.shape[0]), mode="edge")


def _gammatone_fb(sr: int, n_bands: int, fmin: float) -> dict:
    """ERB-spaced 4th-order gammatone filterbank (centre freqs + 20 ms IRs).

    Returns a dict; IR FFTs are cached per FFT length in gammatone() so the
    per-band convolution is one batched FFT-domain multiply (not 64 fftconvolves).
    """
    fmax = sr / 2.0
    # ERB-rate spaced centre frequencies (Glasberg & Moore 1990).
    erb = lambda f: 21.4 * np.log10(4.37e-3 * f + 1.0)
    inv = lambda e: (10 ** (e / 21.4) - 1.0) / 4.37e-3
    cfs = inv(np.linspace(erb(fmin), erb(fmax), n_bands))
    t = np.arange(int(0.020 * sr)) / sr  # 20 ms IR
    irs = np.empty((n_bands, t.size), dtype=np.float64)
    for i, fc in enumerate(cfs):
        b = 1.019 * 24.7 * (4.37e-3 * fc + 1.0)  # ERB bandwidth
        ir = (t ** 3) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
        irs[i] = ir / (np.sqrt(np.sum(ir * ir)) + 1e-12)
    return {"cfs": cfs, "irs": irs, "ir_len": t.size, "_cache": {}}


def logmel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)
    return librosa.power_to_db(S, ref=np.max, top_db=TOP_DB).T.astype(np.float32)  # [T, 128]


def cqt(y: np.ndarray, sr: int) -> np.ndarray:
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=HOP, fmin=CQT_FMIN,
                           n_bins=CQT_BINS, bins_per_octave=CQT_BPO))
    return librosa.amplitude_to_db(C, ref=np.max, top_db=TOP_DB).T.astype(np.float32)  # [T, 192]


def gammatone(y: np.ndarray, sr: int, fb: dict) -> np.ndarray:
    irs = fb["irs"]
    n, m = y.shape[0], fb["ir_len"]
    nfft = scipy.fft.next_fast_len(n + m - 1)
    cache = fb["_cache"]
    if nfft not in cache:
        cache[nfft] = np.fft.rfft(irs, nfft, axis=1)  # [64, nfft//2+1]
    IRf = cache[nfft]
    full = np.fft.irfft(np.fft.rfft(y, nfft)[None, :] * IRf, nfft, axis=1)  # [64, nfft]
    start = (m - 1) // 2
    conv = full[:, start:start + n]  # 'same'-mode central band signals
    bands = [20.0 * np.log10(_frame_energy(conv[i], HOP, N_FFT) + 1e-8)
             for i in range(conv.shape[0])]
    G = np.stack(bands, axis=-1).astype(np.float32)  # [T, 64]
    return G - G.max()


_W: Dict[str, object] = {}


def _init_worker(sr: int, fb: dict) -> None:
    _W["sr"] = sr
    _W["fb"] = fb


def _extract_one(wav_path: str):
    sr = _W["sr"]
    fb = _W["fb"]
    key = os.path.splitext(os.path.basename(wav_path))[0]
    y, fsr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if fsr != sr:
        y = librosa.resample(y, orig_sr=fsr, target_sr=sr)
    return (key,
            logmel(y, sr)[None, :, :],       # [1, T, 128]
            cqt(y, sr)[None, :, :],          # [1, T, 192]
            gammatone(y, sr, fb)[None, :, :])  # [1, T, 64]


def extract_paradigm(stim_dir: Path, sr: int, fb) -> Dict[str, Dict[str, np.ndarray]]:
    wavs = sorted(glob.glob(str(stim_dir / "*.wav")))
    feats = {"logmel": {}, "cqt": {}, "gammatone": {}}
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(sr, fb)) as pool:
        for key, lm, cq, gt in pool.imap_unordered(_extract_one, wavs, chunksize=8):
            feats["logmel"][key] = lm
            feats["cqt"][key] = cq
            feats["gammatone"][key] = gt
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    sr = int(cfg.get("audio", {}).get("sample_rate", 48000))
    paradigms = get_stim_paradigms(cfg)
    fb = _gammatone_fb(sr, GT_BANDS, GT_FMIN)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stim_root = Path(args.stim_root)
    meta = {"sr": sr, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS,
            "cqt_bins": CQT_BINS, "cqt_bpo": CQT_BPO, "gt_bands": GT_BANDS}
    written = {}
    for p in paradigms:
        sd = stim_root / p
        if not sd.is_dir():
            print(f"  [skip] no stimuli dir: {sd}")
            continue
        feats = extract_paradigm(sd, sr, fb)
        for feat_name, d in feats.items():
            d = dict(d)
            d["_meta"] = np.array(str(meta))
            np.savez_compressed(out / f"{feat_name}_{p}.npz", **d)
            written[f"{feat_name}_{p}.npz"] = len(d) - 1
            print(f"  wrote {feat_name}_{p}.npz  n={len(d) - 1}", flush=True)

    # The DSP geometry constants live only in this module; record them, with
    # the library versions that realise them, next to the outputs.
    import json
    import platform
    import scipy
    import soundfile as _sf
    index = {
        "stage": "dsp_frontends",
        "script": Path(__file__).name,
        "constants": {"HOP": HOP, "N_FFT": N_FFT, "N_MELS": N_MELS,
                      "CQT_BINS": CQT_BINS, "CQT_BPO": CQT_BPO,
                      "CQT_FMIN": CQT_FMIN, "GT_BANDS": GT_BANDS,
                      "GT_FMIN": GT_FMIN, "TOP_DB": TOP_DB, "sample_rate": sr},
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "librosa": librosa.__version__,
                     "soundfile": _sf.__version__,
                     "libsndfile": getattr(_sf, "__libsndfile_version__", "n/a")},
        "config": os.environ.get("EXPERIMENT_CONFIG",
                                 "config/stimuli.yaml + config/checkpoints.yaml"),
        "n_workers": N_WORKERS,
        "outputs": written,
        "note": "TOP_DB=80: 80 dB floor (log-mel, CQT).",
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"  wrote index.json ({len(written)} npz)")


if __name__ == "__main__":
    main()
