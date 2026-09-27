#!/usr/bin/env python3
"""Fixed-trajectory (index-only) control for temporal proximity.

For one read-out, replace the features of all stimuli by one array, the
across-stimulus mean trajectory M[layer, t, :], and score it at each
stimulus's own onset frames: phi = Phi(M, O_s). No stimulus-specific feature
variation remains, only the onset indices differ between stimuli.

Same read-out as bcs/score.py (onset frame = round(t / 3 s * T), clamped;
mean between-triplet over mean within-triplet cosine distance; per-anchor
Spearman against the IOI ratio; capture id 0 dropped for multi-capture
read-outs; nine-point interpolation over relative depth; anchor mean),
implemented independently of it. Distances are formed in float64.

Usage
  python3 controls/fixed_trajectory.py --features <read-out>_temp_proximity.npz \
      --metadata <stimuli>/temp_proximity/metadata.json --layers 0,3,6,...
Prints rho of the mean trajectory, its share of feature variance and the
number of scorable anchors (results/fixed_trajectory.json holds all cells).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CLIP_S = 3.0


def frame(t, n):
    return min(max(int(round(t / CLIP_S * n)), 0), n - 1)


def pairs(entry, T):
    on = [frame(float(o['t']), T) for o in entry['onsets']]
    wa, wb, ba, bb = [], [], [], []
    for tr in range(len(on) // 3):
        b = 3 * tr
        wa += [on[b], on[b + 1]]; wb += [on[b + 1], on[b + 2]]
        if tr < len(on) // 3 - 1:
            ba.append(on[b + 2]); bb.append(on[b + 3])
    return wa, wb, ba, bb


def score_fixed(M, layer_ids, keys, midx):
    """M: [L, T, H]; every stimulus uses M, only its onset frames differ."""
    M = M.astype(np.float64)   # float64: a smooth mean trajectory has tiny within-triplet distances
    n = M / np.linalg.norm(M, axis=-1, keepdims=True)
    D = 1 - np.einsum('lth,lsh->lts', n, n)
    T = M.shape[1]
    Y, x, cl = [], [], []
    for k in keys:
        e = midx[k]; wa, wb, ba, bb = pairs(e, T)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y.append(D[:, ba, bb].mean(1) / D[:, wa, wb].mean(1))
        x.append(e['ratio']); cl.append(e['cluster'])
    Y, x, cl = np.array(Y), np.array(x), np.array(cl)
    cols = [i for i, l in enumerate(layer_ids) if l > 0] or [0]
    first, last = layer_ids[cols[0]], layer_ids[cols[-1]]
    depth = [(layer_ids[i] - first) / max(last - first, 1) for i in cols]
    grid = np.linspace(0, 1, 9)
    per_anchor = []
    for c in sorted(set(cl)):
        m = cl == c
        r = [spearmanr(x[m], Y[m, i]).correlation if np.ptp(Y[m, i]) > 0 else np.nan for i in cols]
        per_anchor.append(float(np.mean(np.interp(grid, depth, r))) if len(cols) > 1 else float(r[0]))
    pa = np.array(per_anchor)
    return (float(np.nanmean(pa)) if np.isfinite(pa).any() else None), int(np.isfinite(pa).sum())


def mean_trajectory(npz_path, midx):
    """Across-stimulus mean array M and the share of variance it carries."""
    nz = np.load(npz_path, allow_pickle=True)
    keys = [k for k in nz.files if k != '_meta' and k in midx]
    s = None; ss = 0.0
    for k in keys:
        arr = nz[k].astype(np.float64)
        s = arr.copy() if s is None else s + arr
        ss += float((arr ** 2).sum())
    N = len(keys); M = s / N; L, T, H = M.shape
    g = M.mean(axis=1, keepdims=True)                      # per-layer global mean vector
    total = ss - N * T * float((g ** 2).sum())             # variance around it, all layers
    traj = N * float(((M - g) ** 2).sum())
    return M, keys, (traj / total if total > 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--layers', required=True, help='stored capture ids, comma-separated')
    a = ap.parse_args()
    meta = json.loads(a.metadata.read_text())
    midx = {e['file'][:-4]: e for e in (meta['stimuli'] if isinstance(meta, dict) else meta)}
    M, keys, share = mean_trajectory(a.features, midx)
    rho, n_anchor = score_fixed(M, [int(v) for v in a.layers.split(',')], keys, midx)
    print(json.dumps({'mean_trajectory': rho, 'share': share, 'n_anchors': n_anchor,
                      'n_stimuli': len(keys)}, indent=1))


if __name__ == '__main__':
    main()
