#!/usr/bin/env python3
"""Model-free clock reference for temporal proximity (no audio at all).

h(t) = (cos 2 pi f t, sin 2 pi f t) is a function of the frame index only:
frame k of a T-frame grid over the 3 s clip carries t_k = k * 3 / T. Every
stimulus receives the identical [1, T, 2] array; the read-out (bcs/score.py:
build_score_table + cluster_rhos, a single capture) is then applied at each
stimulus's own onset frames. Cosine distance at lag tau is 1 - cos 2 pi f tau,
so the statistic is [1 - cos 2 pi f r tau] / [1 - cos 2 pi f tau], increasing in
the IOI ratio r while f r tau < 1/2.

The paper reports f in {0.05, 0.1, 0.2} Hz on the nine frame grids of the
read-outs (T = 33, 38, 64, 75, 150, 173, 224, 226, 300).
results/clock_reference.json holds every row computed, including a 1 kHz grid
(T = 3000) that no read-out uses, other clock frequencies (0.333 Hz and above),
a linear/affine clock and random Fourier features. This script recomputes the
pos_fourier rows; the printed min/max are over the nine read-out grids.

Usage
  python3 controls/clock_reference.py --stimuli <stimulus root> [--freqs 0.05,0.1,0.2]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs.score as B  # noqa: E402

GRIDS = [33, 38, 64, 75, 150, 173, 224, 226, 300, 3000]
READOUT_GRIDS = GRIDS[:9]        # the frame grids of the read-outs; 3000 is not one
CLIP_S = 3.0


def clock(f_hz: float, T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float64) * (CLIP_S / float(T))
    w = 2.0 * math.pi * f_hz
    return np.stack([np.cos(w * t), np.sin(w * t)], axis=1)[None, :, :].astype(np.float32)


class _SameArray:
    """Stands in for an npz file: every stimulus key maps to the same array."""

    def __init__(self, keys, arr):
        self.files = list(keys)
        self._arr = arr

    def __getitem__(self, k):
        return self._arr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def score_clock(meta, f_hz: float, T: int, stim_cfg: dict) -> float:
    seq, tone = B.seq_tone(stim_cfg, "temp_proximity")
    keys = [B._stim_key(e) for e in meta]
    arr = clock(f_hz, T)
    opener = B._open_npz
    B._open_npz = lambda _path: _SameArray(keys, arr)
    try:
        tab = B.build_score_table("<clock>", meta, "temp_proximity", seq, tone, stim_cfg)
    finally:
        B._open_npz = opener
    R, _ = B.cluster_rhos(tab.x, tab.Y[:, [0]], tab.cluster)
    r = R[:, 0]
    return float(np.mean(r[np.isfinite(r)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--freqs", default="0.05,0.1,0.2")
    ap.add_argument("--stimulus-config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "stimuli.yaml")
    a = ap.parse_args()
    import yaml
    stim_cfg = yaml.safe_load(a.stimulus_config.read_text())
    meta = json.loads((a.stimuli / "temp_proximity" / "metadata.json").read_text())
    meta = meta["stimuli"] if isinstance(meta, dict) else meta
    rows = [{"f_hz": float(f), "T": T, "rho": score_clock(meta, float(f), T, stim_cfg)}
            for f in a.freqs.split(",") for T in GRIDS]
    used = [r["rho"] for r in rows if r["T"] in READOUT_GRIDS]
    print(json.dumps({"rows": rows, "min": min(used), "max": max(used)}, indent=1))


if __name__ == "__main__":
    main()
