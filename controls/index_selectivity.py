#!/usr/bin/env python3
"""Index selectivity for temporal proximity: Delta_I = rho_9(recorded) - rho_9(index-only).

For one read-out, the recorded rows are the per-stimulus temporal statistics of
its own features (bcs.score.build_score_table); the index-only rows read the
across-stimulus mean trajectory M at each stimulus's own onset frames
(controls/fixed_trajectory.py). Both sides use the same stimuli, anchors and
cue levels, so the paired two-stage cluster bootstrap (anchors, then rows
within anchor x level; 10,000 draws, seed 42) uses one shared index draw for
both sides; the interval is the 95% percentile interval. With the same row
structure for every read-out, the draws are also shared across read-outs.

It applies unchanged to the MusicGen features of the native token schedule
(extraction/extract_musicgen_native.py): pass that feature file as --features
(results/index_selectivity_native.json; same row structure and draws).

Usage
  python3 controls/index_selectivity.py --features <read-out>_temp_proximity.npz \\
      --stimuli <stimulus root> --layers 0,3,6,... [--n-bootstrap 10000]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bcs.score as B  # noqa: E402
from fixed_trajectory import mean_trajectory, pairs  # noqa: E402

CUE = "temp_proximity"


def rows(features: Path, stimuli: Path, layer_ids, stim_cfg):
    """Recorded and index-only rows, in sorted stimulus-key order."""
    meta = json.loads((stimuli / CUE / "metadata.json").read_text())
    meta = meta["stimuli"] if isinstance(meta, dict) else meta
    midx = {e["file"][:-4]: e for e in meta}
    seq, tone = B.seq_tone(stim_cfg, CUE)
    tab = B.build_score_table(features, meta, CUE, seq, tone, stim_cfg)
    geom = B.layer_geometry({"layers_sample": list(layer_ids)}, tab.Y.shape[1])
    M, keys, _ = mean_trajectory(features, midx)
    n = M / np.linalg.norm(M, axis=-1, keepdims=True)
    D = 1 - np.einsum("lth,lsh->lts", n, n)
    T = M.shape[1]
    Yt = {}
    for k in keys:
        wa, wb, ba, bb = pairs(midx[k], T)
        with np.errstate(divide="ignore", invalid="ignore"):
            Yt[k] = D[:, ba, bb].mean(1) / D[:, wa, wb].mean(1)
    order = np.argsort(np.array(tab.keys))
    ck = [tab.keys[i] for i in order]
    cols = geom["nonin_cols"]
    return dict(x=tab.x[order], cl=tab.cluster[order], pl=tab.plevel[order],
                Ya=tab.Y[order][:, cols].astype(float),
                Yb=np.stack([Yt[k] for k in ck])[:, cols],
                depths=np.asarray(geom["depths"], float))


def point(d, n_grid=9):
    Ra, _ = B.cluster_rhos(d["x"], d["Ya"], d["cl"])
    Rb, _ = B.cluster_rhos(d["x"], d["Yb"], d["cl"])
    a = B.aul_common_grid_rows(Ra, d["depths"], n_grid)
    b = B.aul_common_grid_rows(Rb, d["depths"], n_grid)
    dc = a - b
    f = lambda v: float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else math.nan
    return f(a), f(b), f(dc)


def interval(d, n_boot, seed, n_grid=9):
    """Paired bootstrap with one shared index draw (as bcs.score.paired_cell)."""
    rng = np.random.default_rng(seed)
    ri = B.ResampleIndex(d["cl"], d["pl"])
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        _, picks = ri.draw(rng)
        da = np.empty(len(picks)); db = np.empty(len(picks))
        for i, r in enumerate(picks):
            da[i] = B.aul_common_grid(B._rho_for_rows(d["x"], d["Ya"], r), d["depths"], n_grid)
            db[i] = B.aul_common_grid(B._rho_for_rows(d["x"], d["Yb"], r), d["depths"], n_grid)
        with np.errstate(invalid="ignore"):
            draws[b] = float(np.nanmean(da - db))
    return B._percentile_ci(draws, 0.95)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--layers", required=True, help="stored capture ids, comma-separated")
    ap.add_argument("--n-bootstrap", type=int, default=B.ANALYSIS["n_bootstrap"])
    a = ap.parse_args()
    import yaml
    stim_cfg = yaml.safe_load((HERE.parent / "config" / "stimuli.yaml").read_text())
    d = rows(a.features, a.stimuli, [int(v) for v in a.layers.split(",")], stim_cfg)
    rec, ref, delta = point(d)
    lo, hi = interval(d, a.n_bootstrap, B.ANALYSIS["bootstrap_seed"])
    print(json.dumps({"recorded": rec, "trajectory": ref, "delta": delta,
                      "ci_low": lo, "ci_high": hi,
                      "resolves_above": bool(lo > 0), "resolves_below": bool(hi < 0)}, indent=1))


if __name__ == "__main__":
    main()
