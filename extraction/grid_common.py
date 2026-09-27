#!/usr/bin/env python3
"""Shared utilities for the PupuJEPA, Magenta-RT2 and Stable Audio 3 extractors.

Conventions shared by these extractors:
  * activations are computed in float32 and stored as float16;
  * capture ids = unique(round(linspace(0, L-1, 9))) over the L hookable blocks;
  * 2-D token grids are averaged over the non-time axis before storage;
  * per-(model, cue) frame-rate diagnostics are written next to the features.

Nothing in this module computes a BCS value.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

RUN = Path(os.environ.get("BCS_WORK_DIR", "work"))
SUBSET_ROOT = RUN / "subset"
# Stimulus root: BCS_STIM_ROOT points at the full 8,530-clip stimulus set;
# without it a fixed 48-clip subset under SUBSET_ROOT is read.
STIM = Path(os.environ.get("BCS_STIM_ROOT", str(SUBSET_ROOT / "stimuli")))
OUT = RUN / "04_extract" / "_features"

# Nominal clip length per paradigm (s).
T_SEQ = {"freq_proximity": 3.0, "temp_proximity": 3.0, "harmonicity": 0.5,
         "onset_sync": 1.5, "timbre_ablation": 3.0}
DEFAULT_PARADIGMS = ["freq_proximity", "temp_proximity"]
# All five paradigms, in the config's order.
ALL_PARADIGMS = ["freq_proximity", "temp_proximity", "harmonicity",
                 "onset_sync", "timbre_ablation"]


# ---------------------------------------------------------------------------
# Capture-id sampling rule
# ---------------------------------------------------------------------------

def layer_ids(n_blocks: int) -> list[int]:
    """ids = unique(round(linspace(0, L-1, 9))); numpy half-to-even rounding."""
    raw = np.linspace(0, n_blocks - 1, 9)
    return sorted(set(int(v) for v in np.round(raw).astype(int)))


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_wav_mono(path, target_sr: int):
    """48 kHz stimulus -> mono (channel mean) -> torchaudio sinc resample.

    No normalisation is applied (the stimuli are already RMS-normalised).
    Returns a torch.Tensor [1, T] float32.
    """
    import soundfile as sf
    import torch
    import torchaudio

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def load_wav_mono_np(path, target_sr: int) -> np.ndarray:
    """Same contract as load_wav_mono but returns np.ndarray [T].

    torch/torchaudio are imported ONLY when a resample is actually required, so
    that torch-free environments (the JAX env) can still read 48 kHz stimuli at
    their native rate. When a resample is needed the torchaudio sinc kernel is
    used, exactly as in load_wav_mono.
    """
    import soundfile as sf
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr == target_sr:
        return audio
    import torch
    import torchaudio
    w = torchaudio.functional.resample(
        torch.from_numpy(audio).unsqueeze(0), sr, target_sr)
    return w.squeeze(0).numpy()


def iter_paradigm(paradigm: str):
    """Sorted (stem, path) pairs for a paradigm under the active STIM root.

    With BCS_STIM_ROOT set this is the full stimulus set of the paradigm,
    otherwise the fixed subset. Files are visited in filename order.
    """
    d = STIM / paradigm
    return [(p.stem, p) for p in sorted(d.glob("*.wav"))]


iter_stimuli = iter_paradigm          # alias


def paradigm_meta(paradigm: str) -> list[dict]:
    return json.loads((STIM / paradigm / "metadata.json").read_text())["stimuli"]


# ---------------------------------------------------------------------------
# 2-D token flattening
# ---------------------------------------------------------------------------

def flatten_time_major(x, n_time: int, n_other: int):
    """[B, n_time*n_other, H] (time-major order) -> [B, n_time, H] by mean."""
    B, N, H = x.shape
    assert N == n_time * n_other, f"{N} != {n_time}*{n_other}"
    return x.reshape(B, n_time, n_other, H).mean(axis=2)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_npz(path: Path, arrays: dict, meta: dict) -> None:
    """Store float16. Atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cast = {}
    for k, v in arrays.items():
        v = np.asarray(v, dtype=np.float32)
        h = v.astype(np.float16)
        if np.isfinite(v).all() and not np.isfinite(h).all():
            cast[k] = v  # fp16 overflow -> keep float32 rather than lose data
        else:
            cast[k] = h
    cast["_meta"] = np.array(json.dumps(meta), dtype=object)
    tmp = path.with_suffix(".npz.tmp")
    np.savez(tmp, **cast)
    t = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    t.rename(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Frame-rate diagnostics (no rho values involved)
# ---------------------------------------------------------------------------

def _onsets_of(entry: dict, paradigm: str) -> list:
    """Onset times (s) used by the same-frame diagnostic.

    Frequency, temporal and timbre stimuli carry an explicit `onsets` list;
    onset_sync does not: its two tones start at 0 and at `async_ms`, and the
    diagnostic asks whether an asynchrony falls into the same frame as its
    reference. The synchronous
    level (async_ms == 0) is excluded -- there the two tones are *meant* to
    coincide, so a same-frame hit is the stimulus, not a resolution failure.
    """
    ons = entry.get("onsets")
    if ons:
        return [float(o["t"]) for o in ons]
    if paradigm == "onset_sync" and entry.get("async_ms") is not None:
        a = float(entry["async_ms"])
        return [0.0, a / 1000.0] if a != 0.0 else []
    return []


def frame_diagnostics(paradigm: str, n_frames_by_stim: dict, reps: dict) -> dict:
    """n_same_frame_onset_pairs / n_frames / degenerate-row share per cue."""
    meta = {e["file"].replace(".wav", ""): e for e in paradigm_meta(paradigm)}
    t_seq = T_SEQ[paradigm]
    same_pairs, n_frames_list, degen = 0, [], []

    for stem, nf in sorted(n_frames_by_stim.items()):
        n_frames_list.append(int(nf))
        e = meta.get(stem)
        if e is not None:
            onsets = _onsets_of(e, paradigm)
            fr = [min(max(int(round(t / t_seq * nf)), 0), nf - 1) for t in onsets]
            same_pairs += sum(1 for a, b in zip(fr, fr[1:]) if a == b)
        arr = reps.get(stem)
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32)          # [L, T, H]
            norms = np.linalg.norm(a, axis=2)              # [L, T]
            degen.append(float((norms <= 1e-6).mean()))

    return {
        "n_same_frame_onset_pairs": int(same_pairs),
        "n_frames_min": int(min(n_frames_list)) if n_frames_list else None,
        "n_frames_max": int(max(n_frames_list)) if n_frames_list else None,
        "degenerate_row_share": float(np.mean(degen)) if degen else 0.0,
        "flag_coarse_frame_rate": bool(
            same_pairs > 0
            or (n_frames_list and min(n_frames_list) < 10)
            or (degen and float(np.mean(degen)) >= 0.01)),
    }


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------

class Progress:
    """Per-clip rate / ETA logger for a full run (I/O only, no values)."""

    def __init__(self, label: str, total: int, every: int = 100):
        import time
        self.label, self.total, self.every = label, total, every
        self.t0 = time.time()
        self.n = 0
        self._time = time
        print(f"[{label}] start n={total}", flush=True)

    def tick(self) -> None:
        self.n += 1
        if self.n == 1 or self.n % self.every == 0 or self.n == self.total:
            el = self._time.time() - self.t0
            rate = el / self.n
            eta = rate * (self.total - self.n)
            print(f"[{self.label}] {self.n}/{self.total} "
                  f"elapsed={el:.1f}s {rate:.3f}s/clip eta={eta / 60:.1f}min",
                  flush=True)

    def done(self) -> float:
        el = self._time.time() - self.t0
        print(f"[{self.label}] complete n={self.n} in {el:.1f}s "
              f"({el / max(self.n, 1):.3f}s/clip)", flush=True)
        return el


class VramPeak:
    """Context manager returning peak allocated + reserved GB (torch)."""

    def __enter__(self):
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self._t = torch
        return self

    def __exit__(self, *a):
        self.alloc_gb = self._t.cuda.max_memory_allocated() / 1e9
        self.reserved_gb = self._t.cuda.max_memory_reserved() / 1e9
        return False


def nvidia_peak_gb() -> float:
    """Process-independent peak read straight from nvidia-smi (for JAX etc.)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
        return float(out) / 1024.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Disk projection
# ---------------------------------------------------------------------------

CUE_SECONDS = {"freq_proximity": 2304 * 3.0, "temp_proximity": 1992 * 3.0,
               "harmonicity": 1648 * 0.5, "onset_sync": 1536 * 1.5,
               "timbre_ablation": 1050 * 3.0}
TOTAL_SECONDS = sum(CUE_SECONDS.values())   # 19,166 s


def project_bytes(n_layers: int, frame_rate: float, hidden: int) -> float:
    """Bytes ~ n_sampled_layers * frame_rate * H * 2 B * 19,166 s."""
    return n_layers * frame_rate * hidden * 2.0 * TOTAL_SECONDS
