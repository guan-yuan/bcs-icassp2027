#!/usr/bin/env python3
"""Shared utilities for the front-end (input transform) extraction.

Front-ends are scored by the same operator as the checkpoints; the
front-ends are the three fixed DSP transforms, the eight learned codec/VAE
outputs, and PupuM2D-L's log-spectrogram (`fe_pupujepa_logmel`).

Output layout (scored by bcs/score.py through its single-capture path):

    {out}/fe_<name>_<cue>.npz  ->  {stim_stem: [1, T, H] float}  + `_meta`

Storage dtype: every front-end (fixed DSP and learned) is stored as float32;
the arrays are small. `BCS_FE_NPZ_DTYPE=float16` switches to the float16
storage used for checkpoint features.

BCS_STIM_ROOT (required) is the root written by stimuli/generate_stimuli.py;
a complete run checks the per-cue clip counts.

Optional disk guard: with BCS_DISK_BUDGET_GB set, the first complete cue is
used to project the footprint of all five cues, and the run stops if the
projected total of .npz files in the output directory would reach that
budget. Unset (the default), the projection is only recorded.

Nothing in this module computes a BCS value.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

# Stimulus root written by stimuli/generate_stimuli.py (required).
STIM = Path(os.environ["BCS_STIM_ROOT"]) if os.environ.get("BCS_STIM_ROOT") else None


def stim_root() -> Path:
    if STIM is None:
        raise SystemExit("BCS_STIM_ROOT is not set: point it at the stimulus root "
                         "written by stimuli/generate_stimuli.py")
    return STIM

ALL_PARADIGMS = ["freq_proximity", "temp_proximity", "harmonicity",
                 "onset_sync", "timbre_ablation"]

# Nominal clip lengths (used only for the reported frame rate)
T_SEQ = {"freq_proximity": 3.0, "temp_proximity": 3.0, "harmonicity": 0.5,
         "onset_sync": 1.5, "timbre_ablation": 3.0}

# Expected per-cue stimulus counts
EXPECT_N = {"freq_proximity": 2304, "temp_proximity": 1992,
            "harmonicity": 1648, "onset_sync": 1536, "timbre_ablation": 1050}

# Stimulus seconds per cue
CUE_SECONDS = {"freq_proximity": 2304 * 3.0, "temp_proximity": 1992 * 3.0,
               "harmonicity": 1648 * 0.5, "onset_sync": 1536 * 1.5,
               "timbre_ablation": 1050 * 3.0}
TOTAL_SECONDS = sum(CUE_SECONDS.values())            # 19,166 s


def disk_budget_bytes() -> float | None:
    """Optional budget from BCS_DISK_BUDGET_GB (unset or empty: no limit)."""
    v = os.environ.get("BCS_DISK_BUDGET_GB", "").strip()
    return float(v) * 1e9 if v else None


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------

def iter_stimuli(paradigm: str):
    """Sorted (stem, path) pairs -- the same iteration order as grid_common."""
    return [(p.stem, p) for p in sorted((stim_root() / paradigm).glob("*.wav"))]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def fe_dtype() -> np.dtype:
    v = os.environ.get("BCS_FE_NPZ_DTYPE", "float32").strip().lower()
    if v in ("float16", "fp16", "half"):
        return np.float16
    if v in ("float32", "fp32", "single", ""):
        return np.float32
    raise ValueError(f"unsupported BCS_FE_NPZ_DTYPE={v!r}")


def save_fe_npz(path: Path, arrays: dict, meta: dict) -> int:
    """Atomic write of {stem: [1, T, H]} + `_meta`; returns bytes on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    want = fe_dtype()
    cast = {}
    for k, v in arrays.items():
        v = np.asarray(v, dtype=np.float32)
        assert v.ndim == 3 and v.shape[0] == 1, (k, v.shape)
        if want is np.float32:
            cast[k] = v
        else:
            h = v.astype(want)
            # never silently lose a finite value to fp16 overflow
            cast[k] = v if (np.isfinite(v).all() and not np.isfinite(h).all()) else h
    cast["_meta"] = np.array(json.dumps(meta), dtype=object)
    tmp = path.with_suffix(".npz.tmp")
    np.savez(tmp, **cast)
    t = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    t.rename(path)
    return path.stat().st_size


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Disk guard (optional)
# ---------------------------------------------------------------------------

def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in Path(d).glob("*.npz") if p.is_file())


def assert_disk_projection(first_cue: str, first_bytes: int, label: str,
                           cache_dir: Path) -> dict:
    """Extrapolate this front-end's five-cue footprint from one measured cue.

    Bytes scale with stimulus seconds because a front-end emits one frame
    grid per clip at a fixed rate. With BCS_DISK_BUDGET_GB set, aborts if the
    projected total of .npz files in `cache_dir` would reach that budget;
    otherwise the projection is only recorded.
    """
    budget = disk_budget_bytes()
    per_second = first_bytes / CUE_SECONDS[first_cue]
    projected = per_second * TOTAL_SECONDS
    current = dir_bytes(cache_dir)
    total = current - first_bytes + projected
    rec = {"projection_from_cue": first_cue,
           "measured_bytes": int(first_bytes),
           "projected_frontend_bytes": int(projected),
           "cache_bytes_now": int(current),
           "projected_cache_total_bytes": int(total),
           "budget_bytes": None if budget is None else int(budget),
           "ok": bool(budget is None or total < budget)}
    print(f"[{label}] disk projection: this front-end "
          f"{projected / 1e9:.2f} GB, output directory total "
          f"{total / 1e9:.1f} GB"
          + ("" if budget is None else f" (budget {budget / 1e9:.0f} GB)"),
          flush=True)
    if not rec["ok"]:
        raise RuntimeError(
            f"disk projection {total / 1e9:.1f} GB >= BCS_DISK_BUDGET_GB "
            f"{budget / 1e9:.0f} GB -- STOP")
    return rec


# ---------------------------------------------------------------------------
# Progress (I/O only; no values)
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self, label: str, total: int, every: int = 250):
        import time
        self.label, self.total, self.every = label, total, every
        self._time = time
        self.t0 = time.time()
        self.n = 0
        print(f"[{label}] start n={total}", flush=True)

    def tick(self) -> None:
        self.n += 1
        if self.n == 1 or self.n % self.every == 0 or self.n == self.total:
            el = self._time.time() - self.t0
            rate = el / self.n
            print(f"[{self.label}] {self.n}/{self.total} elapsed={el:.1f}s "
                  f"{rate:.4f}s/clip eta={rate * (self.total - self.n) / 60:.1f}min",
                  flush=True)

    def done(self) -> float:
        el = self._time.time() - self.t0
        print(f"[{self.label}] complete n={self.n} in {el:.1f}s", flush=True)
        return el


# ---------------------------------------------------------------------------
# Driver shared by extract_fe_samel.py, extract_fe_spectrostream.py and
# extract_fe_pupujepa_logspec.py (extract_learned_frontends.py and
# extract_dsp_frontends.py do not use it)
# ---------------------------------------------------------------------------

def run_frontend(key: str, encode, paradigms, outdir: Path, diag: dict,
                 hidden_dim: int | None = None, limit: int = 0) -> dict:
    """`encode(path) -> np.ndarray [T, H]` for one clip; writes one npz per cue.

    `limit > 0` reads the first N clips of each cue only: no count assertion,
    no disk projection, and the npz name gets the suffix `_partial` so such a
    run is never mistaken for a complete one.

    Returns the completed diag dict (also written to `{outdir}/{key}_diag.json`).
    """
    outdir = Path(outdir)
    diag = dict(diag)
    diag.setdefault("model_name", key)
    diag["storage_dtype"] = np.dtype(fe_dtype()).name
    diag["layers_sample"] = [0]
    diag["stim_root"] = str(stim_root())
    diag["paradigms"] = {}
    if limit:
        diag["clip_limit"] = int(limit)
    first = None

    for pdm in paradigms:
        suffix = "_partial" if limit else ""
        npz = outdir / f"{key}_{pdm}{suffix}.npz"
        if npz.exists():
            print(f"[{key}] SKIP {pdm} (exists)", flush=True)
            continue
        items = iter_stimuli(pdm)
        if limit:
            items = items[:limit]
        else:
            assert len(items) == EXPECT_N[pdm], \
                f"{pdm}: {len(items)} wavs != {EXPECT_N[pdm]}"
        reps, nframes, shape0 = {}, {}, None
        prog = Progress(f"{key}/{pdm}", len(items))
        for stem, path in items:
            h = np.asarray(encode(path), dtype=np.float32)
            assert h.ndim == 2, (stem, h.shape)
            if hidden_dim is not None:
                assert h.shape[1] == hidden_dim, (stem, h.shape, hidden_dim)
            assert np.isfinite(h).all(), f"non-finite front-end output: {stem}"
            reps[stem] = h[None, :, :]
            nframes[stem] = h.shape[0]
            if shape0 is None:
                shape0 = [1, int(h.shape[0]), int(h.shape[1])]
            prog.tick()
        secs = prog.done()
        if not limit:
            assert len(reps) == EXPECT_N[pdm], (pdm, len(reps))
        nbytes = save_fe_npz(npz, reps, diag)
        fr = float(np.mean(list(nframes.values()))) / T_SEQ[pdm]
        diag["paradigms"][pdm] = {
            "n_stimuli": len(reps), "shape_first": shape0,
            "n_frames_min": int(min(nframes.values())),
            "n_frames_max": int(max(nframes.values())),
            "measured_frame_rate_hz": round(fr, 4),
            "npz": str(npz), "npz_bytes": int(nbytes),
            "extract_seconds": round(secs, 1)}
        if first is None and not limit:
            first = (pdm, nbytes)
            diag["disk_projection"] = assert_disk_projection(pdm, nbytes, key, outdir)
        del reps

    name = f"{key}_diag{'_partial' if limit else ''}.json"
    (outdir / name).write_text(json.dumps(diag, indent=2))
    print(f"[{key}] done", flush=True)
    return diag
