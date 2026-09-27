"""The BCS operator: cue statistics, anchor-mean Spearman, depth summaries,
paired front-end contrasts and the two-stage cluster bootstrap.

Pipeline for one read-out (checkpoint or front-end) and one cue
  1. build_score_table: per stimulus and per captured layer, one scalar cue
     statistic phi from the time-averaged / onset-indexed features
       frequency proximity  cosine distance between stream-A and stream-B frames
       temporal proximity   mean between-triplet / mean within-triplet distance
                            of onset-frame pairs (means taken before the ratio)
       harmonicity, onset   distance to the anchor's zero-manipulation reference
  2. cluster_rhos: Spearman(phi, cue value) within each anchor, average ranks,
     anchors with a constant cue or statistic give NaN and are dropped
  3. anchor mean per capture, then a depth summary over the retained captures
     (capture id 0 excluded): exact_auc (primary) or aul_common_grid (rho_9)
  4. intervals: two-stage cluster bootstrap (anchors, then rows within
     anchor x cue level), one index draw shared by every capture and by both
     sides of a paired contrast; exact-AUC and paired intervals are
     percentile intervals, rho_9 cells use BCa when all six conditions
     hold (BCA_RULES) and fall back to percentile otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# Analysis settings used for every number in the paper.
ANALYSIS = {
    "n_bootstrap": 10000,
    "bootstrap_seed": 42,
    "n_permutation": 10000,
    "permutation_seed": 4242,
    "aul_grid_points": 9,
    "tost_delta": 0.3,
    "tost_ci_level": 0.9,
    "bca": {
        "clip": [0.005, 0.995],
        "min_n_clusters": 20,
        "max_abs_a_hat": 0.25,
        "max_abs_z0": 0.5,
        "width_ratio_range": [0.5, 2.0],
    },
    "diagnostics": {
        "distinct_multiset_min": 0.99,
        "influence_concentration_max": 0.3,
        "degenerate_draw_max_fraction": 0.05,
    },
}
BCA_RULES = ANALYSIS["bca"]
N_BOOT = ANALYSIS["n_bootstrap"]
SEED = ANALYSIS["bootstrap_seed"]


def _stim_key(meta_entry: dict) -> str:
    """Metadata entry -> feature key (the WAV file stem)."""
    return Path(meta_entry["file"]).stem


# -----------------------------------------------------------------------------
# Cue keys
# -----------------------------------------------------------------------------

CORE_CUES = ["freq_proximity", "temp_proximity", "harmonicity", "onset_sync"]

RERENDER_CUES = ("tp_rr_main",)



# -----------------------------------------------------------------------------
# Frame alignment and distances
# -----------------------------------------------------------------------------

def _time_to_frame(t_sec: float, n_frames: int, seq_dur_s: float) -> int:
    if seq_dur_s <= 0 or n_frames <= 0:
        return 0
    idx = int(round(t_sec / seq_dur_s * n_frames))
    return max(0, min(n_frames - 1, idx))


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 1.0
    return float(1.0 - float(np.dot(u, v)) / (nu * nv))


def _cosine_distance_rows(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance for [L, H] matrices -> [L]."""
    nu = np.linalg.norm(U, axis=1)
    nv = np.linalg.norm(V, axis=1)
    denom = nu * nv
    num = np.sum(U * V, axis=1)
    out = np.ones_like(num, dtype=float)
    ok = denom > 1e-12
    out[ok] = 1.0 - num[ok] / denom[ok]
    return out


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pooled Spearman over all rows (companion statistic, not the estimator)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), 1.0
    res = stats.spearmanr(x, y, nan_policy="omit")
    r = float(res.correlation if hasattr(res, "correlation") else res[0])
    p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
    if not math.isfinite(r):
        return float("nan"), 1.0
    return r, p




# -----------------------------------------------------------------------------
# Per-anchor Spearman, anchor mean, nine-point depth summary (rho_9)
# -----------------------------------------------------------------------------

def _rank_avg(a: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks along axis 0 (bootstrap duplicates create real ties)."""
    return stats.rankdata(a, axis=0)


def _corr_cols(rx: np.ndarray, RY: np.ndarray) -> np.ndarray:
    """Pearson r between a fixed vector rx [n] and each column of RY [n, L]."""
    rxc = rx - rx.mean()
    ryc = RY - RY.mean(axis=0, keepdims=True)
    denom = np.sqrt(float(np.dot(rxc, rxc)) * np.sum(ryc * ryc, axis=0))
    num = rxc @ ryc
    out = np.full(RY.shape[1], np.nan, dtype=float)
    ok = denom > 1e-15
    out[ok] = num[ok] / denom[ok]
    return out


def cluster_rhos(x: np.ndarray, Y: np.ndarray,
                 cluster: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Per-cluster Spearman rho for every layer.

    Returns (R [n_clusters, L], cluster_codes). rho_c is NaN when the cluster's
    x or y is constant (degenerate); callers decide how to treat it.
    """
    codes = np.unique(cluster)
    out = np.full((codes.size, Y.shape[1]), np.nan, dtype=float)
    for i, c in enumerate(codes):
        m = cluster == c
        if m.sum() < 3:
            continue
        xi = x[m]
        if np.all(xi == xi[0]):
            continue
        rx = stats.rankdata(xi)
        RY = _rank_avg(Y[m])
        out[i] = _corr_cols(rx, RY)
    return out, list(codes)


def cluster_mean_rho(x: np.ndarray, y: np.ndarray,
                     cluster: np.ndarray) -> Dict[str, float]:
    """Anchor-mean Spearman for a single capture (convenience function)."""
    R, codes = cluster_rhos(x, y.reshape(-1, 1), cluster)
    r = R[:, 0]
    fin = r[np.isfinite(r)]
    return {
        "rho_bar": float(np.mean(fin)) if fin.size else float("nan"),
        "sd_cluster": float(np.std(fin, ddof=1)) if fin.size > 1 else float("nan"),
        "n_clusters": int(fin.size),
        "per_cluster": r.tolist(),
        "cluster_codes": [str(c) for c in codes],
    }


def aul_common_grid(rho_by_layer: Sequence[float], depths: Sequence[float],
                    n_grid: int = 9) -> float:
    """rho_9: interpolate rho(depth) onto t_j = j/(n_grid-1) and average.

    Linear interpolation between adjacent sampled layers, no extrapolation
    (the endpoints are exactly the d=0 / d=1 layer values).
    """
    r = np.asarray(rho_by_layer, dtype=float)
    d = np.asarray(depths, dtype=float)
    ok = np.isfinite(r)
    if ok.sum() == 0:
        return float("nan")
    if ok.sum() == 1:
        return float(r[ok][0])
    t = np.linspace(0.0, 1.0, n_grid)
    return float(np.mean(np.interp(t, d[ok], r[ok])))


def aul_common_grid_rows(R: np.ndarray, depths: np.ndarray,
                         n_grid: int = 9) -> np.ndarray:
    """Vectorised AUL for R [k, L] (k = clusters or bootstrap draws)."""
    t = np.linspace(0.0, 1.0, n_grid)
    out = np.full(R.shape[0], np.nan, dtype=float)
    d = np.asarray(depths, dtype=float)
    for i in range(R.shape[0]):
        row = R[i]
        ok = np.isfinite(row)
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            out[i] = row[ok][0]
            continue
        out[i] = float(np.mean(np.interp(t, d[ok], row[ok])))
    return out




# -----------------------------------------------------------------------------
# Exact depth AUC and the other depth summaries over retained captures
# -----------------------------------------------------------------------------

SUMMARY_NAMES = ("auc", "rho5", "rho9", "rho17", "last", "first", "ret_mean")


def _clean(depths: Sequence[float], values: Sequence[float]):
    """Drop non-finite scores, returning float arrays sorted by depth."""
    d = np.asarray(depths, dtype=float)
    y = np.asarray(values, dtype=float)
    if d.shape != y.shape:
        raise ValueError(f"depth/value shape mismatch: {d.shape} vs {y.shape}")
    ok = np.isfinite(y) & np.isfinite(d)
    d, y = d[ok], y[ok]
    order = np.argsort(d, kind="stable")
    return d[order], y[order]


def grid_mean(depths: Sequence[float], values: Sequence[float], n_grid: int) -> float:
    """Mean of f(t) on the equally spaced grid t_j = j/(n_grid-1), j=0..n_grid-1.

    n_grid = 9 gives the nine-point summary rho_9 used for the controls.
    """
    d, y = _clean(depths, values)
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])
    t = np.linspace(0.0, 1.0, int(n_grid))
    return float(np.mean(np.interp(t, d, y)))


def exact_auc(depths: Sequence[float], values: Sequence[float]) -> float:
    """Normalized exact AUC = integral_0^1 f(t) dt of the same interpolant.

    f is piecewise linear, so the integral is the trapezoidal rule on the knots,
    divided by the span of the finite knots (1 when every capture is finite).
    """
    d, y = _clean(depths, values)
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])
    span = float(d[-1] - d[0])
    if span <= 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, d) / span)


def last_value(depths: Sequence[float], values: Sequence[float]) -> float:
    d, y = _clean(depths, values)
    return float(y[-1]) if y.size else float("nan")


def first_value(depths: Sequence[float], values: Sequence[float]) -> float:
    d, y = _clean(depths, values)
    return float(y[0]) if y.size else float("nan")


def retained_mean(depths: Sequence[float], values: Sequence[float]) -> float:
    """Unweighted mean over retained captures (no depth interpolation)."""
    d, y = _clean(depths, values)
    return float(np.mean(y)) if y.size else float("nan")


def all_summaries(depths: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    return {
        "auc": exact_auc(depths, values),
        "rho5": grid_mean(depths, values, 5),
        "rho9": grid_mean(depths, values, 9),
        "rho17": grid_mean(depths, values, 17),
        "last": last_value(depths, values),
        "first": first_value(depths, values),
        "ret_mean": retained_mean(depths, values),
    }


def summaries_rows(depths: Sequence[float], V: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised ``all_summaries`` over rows of ``V`` [n, K] (bootstrap draws)."""
    V = np.asarray(V, dtype=float)
    out = {k: np.full(V.shape[0], np.nan) for k in SUMMARY_NAMES}
    for i in range(V.shape[0]):
        s = all_summaries(depths, V[i])
        for k in SUMMARY_NAMES:
            out[k][i] = s[k]
    return out


def percentile_ci(draws: Sequence[float], level: float = 0.95):
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 100:
        return float("nan"), float("nan")
    a = 1.0 - level
    lo, hi = np.percentile(d, [100.0 * a / 2.0, 100.0 * (1.0 - a / 2.0)])
    return float(lo), float(hi)




# -----------------------------------------------------------------------------
# Two-stage cluster bootstrap (anchors, then rows within anchor x cue level)
# -----------------------------------------------------------------------------

class ResampleIndex:
    """Pre-compiled row index for the two-stage resample of one score table."""

    def __init__(self, cluster: np.ndarray, plevel: np.ndarray):
        self.cluster_codes = np.unique(cluster)
        self.n_clusters = self.cluster_codes.size
        # rows of each cluster, grouped by predictor level
        self.groups: List[List[np.ndarray]] = []
        self.rows: List[np.ndarray] = []
        for c in self.cluster_codes:
            m = np.where(cluster == c)[0]
            self.rows.append(m)
            gl = [m[plevel[m] == p] for p in np.unique(plevel[m])]
            self.groups.append([g for g in gl if g.size > 0])

    def draw(self, rng: np.random.Generator) -> Tuple[List[int], List[np.ndarray]]:
        """Stage 1: clusters with replacement. Stage 2: within (cluster, plevel).

        Returns (chosen_cluster_indices, per-chosen-cluster row indices).
        """
        chosen = rng.integers(0, self.n_clusters, size=self.n_clusters)
        picks = []
        for c in chosen:
            parts = [g[rng.integers(0, g.size, size=g.size)] for g in self.groups[c]]
            picks.append(np.concatenate(parts))
        return list(chosen), picks


def _rho_for_rows(x: np.ndarray, Y: np.ndarray,
                  rows: np.ndarray) -> np.ndarray:
    """Per-layer Spearman rho for one (resampled) cluster's rows."""
    xi = x[rows]
    if xi.size < 3 or np.all(xi == xi[0]):
        return np.full(Y.shape[1], np.nan)
    rx = stats.rankdata(xi)
    RY = _rank_avg(Y[rows])
    return _corr_cols(rx, RY)


def bootstrap_aul(x: np.ndarray, Y: np.ndarray, cluster: np.ndarray,
                  plevel: np.ndarray, depths: np.ndarray,
                  n_bootstrap: int, seed: int, n_grid: int = 9,
                  max_redraw: int = 10) -> Dict[str, object]:
    """One shared index draw per replicate, applied to every layer.

    Returns the AUL draws, the per-layer rho_bar draws, the distinct
    first-stage multiset ratio and the degenerate-draw count.
    """
    rng = np.random.default_rng(seed)
    ri = ResampleIndex(cluster, plevel)
    L = Y.shape[1]
    layer_draws = np.full((n_bootstrap, L), np.nan, dtype=float)
    multisets = set()
    n_degenerate = 0
    for b in range(n_bootstrap):
        chosen, picks = ri.draw(rng)
        multisets.add(tuple(sorted(chosen)))
        R = np.empty((len(picks), L), dtype=float)
        for i, rows in enumerate(picks):
            r = _rho_for_rows(x, Y, rows)
            attempts = 0
            while not np.all(np.isfinite(r)) and attempts < max_redraw:
                # redraw this cluster up to `max_redraw` times
                c = chosen[i]
                parts = [g[rng.integers(0, g.size, size=g.size)]
                         for g in ri.groups[c]]
                r = _rho_for_rows(x, Y, np.concatenate(parts))
                attempts += 1
            if not np.all(np.isfinite(r)):
                n_degenerate += 1
            R[i] = r
        with np.errstate(invalid="ignore"):
            layer_draws[b] = np.nanmean(R, axis=0)
    aul_draws = aul_common_grid_rows(layer_draws, depths, n_grid)
    return {
        "aul_draws": aul_draws,
        "layer_draws": layer_draws,
        "distinct_multiset_ratio": len(multisets) / float(n_bootstrap),
        "n_degenerate_draws": int(n_degenerate),
    }




# -----------------------------------------------------------------------------
# Intervals: BCa when six conditions hold, percentile fallback
# -----------------------------------------------------------------------------

def _percentile_ci(draws: np.ndarray, ci_level: float) -> Tuple[float, float]:
    d = draws[np.isfinite(draws)]
    if d.size < 100:
        return float("nan"), float("nan")
    a = 1.0 - ci_level
    lo, hi = np.percentile(d, [100.0 * a / 2.0, 100.0 * (1.0 - a / 2.0)])
    return float(lo), float(hi)


def bca_ci(theta_hat: float, draws: np.ndarray, jack: np.ndarray,
           ci_level: float, clip: Tuple[float, float],
           n_clusters: int, rules: dict) -> Dict[str, object]:
    """BCa when six conditions hold; falls back to percentile.

    The six conditions: n_clusters >= min_n_clusters; a finite delete-one-
    cluster jackknife; |a_hat| <= max_abs_a_hat; |z0| <= max_abs_z0; BCa
    quantiles not clipped; BCa/percentile width ratio in width_ratio_range.
    """
    out: Dict[str, object] = {"ci_method": "bca", "ci_fallback_reason": None}
    d = draws[np.isfinite(draws)]
    jk = jack[np.isfinite(jack)]
    p_lo, p_hi = _percentile_ci(draws, ci_level)
    out["ci_percentile"] = [p_lo, p_hi]

    reasons: List[str] = []
    if n_clusters < int(rules.get("min_n_clusters", 20)):
        reasons.append(f"n_clusters<{rules.get('min_n_clusters', 20)}")
    if jk.size < max(0, n_clusters - 2):
        reasons.append("jackknife_finite<n_clusters-2")

    z0 = a_hat = float("nan")
    a_lo_raw = a_hi_raw = float("nan")
    if d.size >= 100 and jk.size >= 3:
        frac_less = float((d < theta_hat).mean())
        frac_less = min(max(frac_less, 1e-6), 1 - 1e-6)
        z0 = float(stats.norm.ppf(frac_less))
        jbar = float(jk.mean())
        num = float(np.sum((jbar - jk) ** 3))
        den = 6.0 * (float(np.sum((jbar - jk) ** 2)) ** 1.5 + 1e-12)
        a_hat = num / den
        alpha = 1.0 - ci_level
        z_lo = float(stats.norm.ppf(alpha / 2.0))
        z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))
        a_lo_raw = float(stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo))))
        a_hi_raw = float(stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi))))
        if abs(a_hat) > float(rules.get("max_abs_a_hat", 0.25)):
            reasons.append(f"|a_hat|={abs(a_hat):.3f}>{rules.get('max_abs_a_hat', 0.25)}")
        if abs(z0) > float(rules.get("max_abs_z0", 0.5)):
            reasons.append(f"|z0|={abs(z0):.3f}>{rules.get('max_abs_z0', 0.5)}")
        lo_c, hi_c = clip
        if a_lo_raw <= lo_c or a_hi_raw >= hi_c:
            reasons.append(f"bca_quantile_clipped_at_{lo_c}/{hi_c}")
    else:
        reasons.append("insufficient_draws_or_jackknife")

    out.update({"z0": z0, "a_hat": a_hat, "a_lo": a_lo_raw, "a_hi": a_hi_raw})

    if not reasons and math.isfinite(a_lo_raw) and math.isfinite(a_hi_raw):
        lo = float(np.quantile(d, np.clip(a_lo_raw, clip[0], clip[1])))
        hi = float(np.quantile(d, np.clip(a_hi_raw, clip[0], clip[1])))
        w_bca = hi - lo
        w_pct = p_hi - p_lo
        rng_lo, rng_hi = rules.get("width_ratio_range", [0.5, 2.0])
        if math.isfinite(w_pct) and w_pct > 0 and not (rng_lo * w_pct <= w_bca <= rng_hi * w_pct):
            reasons.append(f"width_ratio={w_bca / w_pct:.2f}_outside_{rng_lo}-{rng_hi}")
        else:
            out["ci_low"], out["ci_high"] = lo, hi
            return out

    out["ci_method"] = "percentile"
    out["ci_fallback_reason"] = ";".join(reasons)
    out["ci_low"], out["ci_high"] = p_lo, p_hi
    return out




# -----------------------------------------------------------------------------
# Permutation p-values
# -----------------------------------------------------------------------------

def cluster_permutation_p(x: np.ndarray, Y: np.ndarray, cluster: np.ndarray,
                          depths: np.ndarray, theta_hat: float,
                          n_perm: int, seed: int, n_grid: int = 9) -> Dict[str, float]:
    """Within-cluster shuffle of the predictor; statistic = rho_bar_AUL."""
    rng = np.random.default_rng(seed)
    codes = np.unique(cluster)
    idx = [np.where(cluster == c)[0] for c in codes]
    L = Y.shape[1]
    stat = np.empty(n_perm, dtype=float)
    ranksY = [_rank_avg(Y[i]) for i in idx]
    xs = [x[i] for i in idx]
    for b in range(n_perm):
        R = np.empty((len(idx), L), dtype=float)
        for i in range(len(idx)):
            xi = rng.permutation(xs[i])
            R[i] = _corr_cols(stats.rankdata(xi), ranksY[i])
        with np.errstate(invalid="ignore"):
            stat[b] = aul_common_grid(np.nanmean(R, axis=0), depths, n_grid)
    fin = stat[np.isfinite(stat)]
    n_ge = int(np.sum(np.abs(fin) >= abs(theta_hat)))
    p = (1.0 + n_ge) / (1.0 + fin.size)
    return {"p_permutation": float(p), "n_perm_valid": int(fin.size),
            "perm_floor": 1.0 / (1.0 + fin.size),
            "perm_null_sd": float(np.std(fin)) if fin.size else float("nan")}


def signflip_permutation_p(deltas: np.ndarray, n_perm: int,
                           seed: int) -> Dict[str, float]:
    """Sign-flip permutation on the per-cluster paired differences."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {"p_permutation": float("nan"), "n_perm_valid": 0,
                "perm_floor": float("nan")}
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(d)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    stat = np.abs((signs * d[None, :]).mean(axis=1))
    p = (1.0 + int(np.sum(stat >= obs))) / (1.0 + n_perm)
    return {"p_permutation": float(p), "n_perm_valid": int(n_perm),
            "perm_floor": 1.0 / (1.0 + n_perm)}




# -----------------------------------------------------------------------------
# Per-stimulus score tables: the four cue statistics
# -----------------------------------------------------------------------------

class ScoreTable:
    __slots__ = ("x", "Y", "cluster", "plevel", "cluster_names", "keys",
                 "n_dropped", "dropped_reasons")

    def __init__(self, x, Y, cluster, plevel, cluster_names, keys,
                 n_dropped, dropped_reasons):
        self.x = x
        self.Y = Y
        self.cluster = cluster
        self.plevel = plevel
        self.cluster_names = cluster_names
        self.keys = keys
        self.n_dropped = n_dropped
        self.dropped_reasons = dropped_reasons

    @property
    def n_rows(self) -> int:
        return int(self.x.size)


def _finish(rows, drops) -> ScoreTable:
    if not rows:
        return ScoreTable(np.zeros(0), np.zeros((0, 0)), np.zeros(0, dtype=int),
                          np.zeros(0, dtype=int), [], [], sum(drops.values()), drops)
    names = sorted({r[2] for r in rows})
    code = {n: i for i, n in enumerate(names)}
    x = np.array([r[0] for r in rows], dtype=float)
    Y = np.stack([r[1] for r in rows], axis=0)
    cl = np.array([code[r[2]] for r in rows], dtype=int)
    pl = np.array([r[3] for r in rows], dtype=int)
    keys = [r[4] for r in rows]
    return ScoreTable(x, Y, cl, pl, names, keys, sum(drops.values()), drops)


def _as_f32(a: np.ndarray) -> np.ndarray:
    """Promote a stored representation to float32 before any arithmetic.

    float16 is a storage cast only; float16 arithmetic overflows: |h| > 256
    squares past the 65504 fp16 ceiling, so norms and dot products silently
    become inf and every cosine distance in that stimulus turns into NaN. Every read of an
    npz entry therefore goes through this promoter.
    """
    a = np.asarray(a)
    return a.astype(np.float32) if a.dtype not in (np.float32, np.float64) else a


def _open_npz(path):
    return np.load(path, allow_pickle=True)


def _meta_index(meta: List[dict]) -> Dict[str, dict]:
    return {_stim_key(e): e for e in meta}


def _cluster_of(e: dict, paradigm: str) -> str:
    c = e.get("cluster")
    if c is not None:
        return str(c)
    # metadata without an explicit cluster field
    if paradigm == "freq_proximity":
        return f"anchor_{e['anchor_hz']:g}"
    if paradigm == "temp_proximity":
        return f"ioi_{float(e['base_ioi_s'])*1000:g}ms"
    if paradigm == "harmonicity":
        return f"f0_{e['f0']:g}_rk{e['mistuned_rank']}"
    if paradigm == "onset_sync":
        return str(e["config"])
    return f"anchor_{e.get('base_freq_hz', 0):g}"


def build_score_table(npz_path: Path, meta: List[dict], paradigm: str,
                      seq_dur_s: float, tone_dur: float,
                      cfg: dict, include_harm_12pct: bool = False):
    """Stream the npz once (twice for baseline cues) -> per-stimulus scalars."""
    midx = _meta_index(meta)
    drops: Dict[str, int] = {}

    def drop(reason):
        drops[reason] = drops.get(reason, 0) + 1

    with _open_npz(npz_path) as nz:
        keys = [k for k in nz.files if k != "_meta"]

        # ---- frequency proximity: A-vs-B cosine distance, one pass --------
        if paradigm in ("freq_proximity", "timbre_ablation"):
            per_timbre: Dict[str, list] = {}
            rows = []
            for k in keys:
                e = midx.get(k)
                if e is None:
                    drop("no_metadata")
                    continue
                arr = _as_f32(nz[k])
                T = arr.shape[1]
                a_frames, b_frames = [], []
                for o in e.get("onsets") or []:
                    t_on = float(o["t"])
                    lo = _time_to_frame(t_on, T, seq_dur_s)
                    hi = _time_to_frame(t_on + tone_dur, T, seq_dur_s)
                    rngf = list(range(lo, max(lo + 1, hi)))
                    (a_frames if o.get("stream", "A") == "A" else b_frames).extend(rngf)
                if not a_frames or not b_frames:
                    drop("no_ab_frames")
                    continue
                A = arr[:, a_frames, :].mean(axis=1)     # [L, H]
                B = arr[:, b_frames, :].mean(axis=1)
                d = _cosine_distance_rows(A, B)          # [L]
                row = (float(e["delta_f"]), d, _cluster_of(e, paradigm),
                       int(e.get("plevel", 0)), k)
                if paradigm == "timbre_ablation":
                    per_timbre.setdefault(str(e["timbre"]), []).append(row)
                else:
                    rows.append(row)
            if paradigm == "timbre_ablation":
                return {t: _finish(r, dict(drops)) for t, r in per_timbre.items()}
            return _finish(rows, drops)

        # ---- temporal proximity (and its re-rendered set): between/within
        #      distance ratio, one pass
        if paradigm == "temp_proximity" or paradigm in RERENDER_CUES:
            TRIAL = 3
            rows = []
            for k in keys:
                e = midx.get(k)
                if e is None:
                    drop("no_metadata")
                    continue
                arr = _as_f32(nz[k])
                T = arr.shape[1]
                onsets = e.get("onsets") or []
                n_tr = len(onsets) // TRIAL
                if n_tr < 2:
                    drop("fewer_than_2_trials")
                    continue
                within_a, within_b, betw_a, betw_b = [], [], [], []
                for tr in range(n_tr):
                    base = tr * TRIAL
                    for j in range(TRIAL - 1):
                        within_a.append(_time_to_frame(float(onsets[base + j]["t"]), T, seq_dur_s))
                        within_b.append(_time_to_frame(float(onsets[base + j + 1]["t"]), T, seq_dur_s))
                    if tr < n_tr - 1:
                        betw_a.append(_time_to_frame(float(onsets[base + TRIAL - 1]["t"]), T, seq_dur_s))
                        betw_b.append(_time_to_frame(float(onsets[(tr + 1) * TRIAL]["t"]), T, seq_dur_s))
                if not within_a or not betw_a:
                    drop("no_pairs")
                    continue
                L = arr.shape[0]
                dw = np.stack([_cosine_distance_rows(arr[:, i, :], arr[:, j, :])
                               for i, j in zip(within_a, within_b)], axis=0).mean(axis=0)
                db = np.stack([_cosine_distance_rows(arr[:, i, :], arr[:, j, :])
                               for i, j in zip(betw_a, betw_b)], axis=0).mean(axis=0)
                if np.any(dw < 1e-9):
                    drop("degenerate_within_distance")
                    continue
                rows.append((float(e["ratio"]), db / dw, _cluster_of(e, paradigm),
                             int(e.get("plevel", 0)), k))
            return _finish(rows, drops)

        # ---- harmonicity / onset synchrony: distance from the per-anchor
        #      zero-manipulation reference, two passes
        base_key = "mistuning_pct" if paradigm == "harmonicity" else "async_ms"
        x_key = base_key
        acc: Dict[str, List[np.ndarray]] = {}
        for k in keys:
            e = midx.get(k)
            if e is None:
                continue
            if float(e[base_key]) != 0.0:
                continue
            acc.setdefault(_cluster_of(e, paradigm), []).append(
                _as_f32(nz[k]).mean(axis=1))                       # [L, H]
        baselines = {c: np.mean(np.stack(v, axis=0), axis=0) for c, v in acc.items()}
        del acc
        rows = []
        main_mist = None
        if paradigm == "harmonicity" and not include_harm_12pct:
            main_mist = set(float(m) for m in cfg["stimuli"]["harmonicity"].get(
                "main_estimate_mistuning_pcts", [1, 1.5, 2, 3, 4, 6, 8]))
        for k in keys:
            e = midx.get(k)
            if e is None:
                drop("no_metadata")
                continue
            xv = float(e[x_key])
            if xv == 0.0:
                continue                                   # baseline row
            if main_mist is not None and xv not in main_mist:
                drop("harm_unbalanced_12pct_excluded")
                continue
            c = _cluster_of(e, paradigm)
            if c not in baselines:
                drop("no_baseline_for_cluster")
                continue
            v = _as_f32(nz[k]).mean(axis=1)                          # [L, H]
            d = _cosine_distance_rows(v, baselines[c])
            rows.append((xv, d, c, int(e.get("plevel", 0)), k))
        return _finish(rows, drops)


def layer_geometry(model_cfg: dict, n_layers_npz: int) -> Dict[str, object]:
    """Map npz layer columns -> sampled layer ids, non-input mask, depths."""
    ids = list(model_cfg.get("layers_sample") or list(range(n_layers_npz)))
    if len(ids) != n_layers_npz:
        # extraction may have clamped/deduplicated; fall back to positional ids
        ids = list(range(n_layers_npz))
    is_dsp = (len(ids) == 1)
    nonin = [i for i, lid in enumerate(ids) if lid > 0]
    if not nonin:                       # single-layer DSP front-end
        return {"layer_ids": ids, "nonin_cols": [0], "depths": np.array([0.0]),
                "is_dsp": True}
    first, last = ids[nonin[0]], ids[nonin[-1]]
    span = float(last - first) if last > first else 1.0
    depths = np.array([(ids[i] - first) / span for i in nonin], dtype=float)
    return {"layer_ids": ids, "nonin_cols": nonin, "depths": depths,
            "is_dsp": is_dsp}




# -----------------------------------------------------------------------------
# rho_9 cell and paired rho_9 contrast on common stimuli
# -----------------------------------------------------------------------------

def analyse_cell(tab: ScoreTable, geom: dict, an: dict, label: str) -> dict:
    """rho_9 point estimate, BCa/percentile interval and permutation p for one cell."""
    t0 = time.time()
    n_grid = int(an.get("aul_grid_points", 9))
    n_boot = int(an.get("n_bootstrap", 10000))
    n_perm = int(an.get("n_permutation", 10000))
    boot_seed = int(an.get("bootstrap_seed", 42))
    perm_seed = int(an.get("permutation_seed", 4242))
    diag = an.get("diagnostics", {})
    bca_rules = an.get("bca", {})
    clip = tuple(bca_rules.get("clip", [0.005, 0.995]))

    cols = geom["nonin_cols"]
    depths = geom["depths"]
    Yn = tab.Y[:, cols]

    out: Dict[str, object] = {
        "label": label,
        "n_rows": tab.n_rows,
        "n_dropped_rows": int(tab.n_dropped),
        "dropped_reasons": tab.dropped_reasons,
        "layer_ids": geom["layer_ids"],
        "nonin_layer_ids": [geom["layer_ids"][i] for i in cols],
        "relative_depths": [float(d) for d in depths],
        "is_dsp_single_layer": bool(geom["is_dsp"]),
    }
    if tab.n_rows < 10:
        out["status"] = "insufficient_rows"
        return out

    # ---- point estimates -------------------------------------------------
    R, codes = cluster_rhos(tab.x, Yn, tab.cluster)      # [C, Ln]
    per_cluster_aul = aul_common_grid_rows(R, depths, n_grid)
    fin = per_cluster_aul[np.isfinite(per_cluster_aul)]
    theta = float(np.mean(fin)) if fin.size else float("nan")
    n_clusters = int(fin.size)
    sd_cluster = float(np.std(fin, ddof=1)) if fin.size > 1 else float("nan")

    with np.errstate(invalid="ignore"):
        rho_by_layer = np.nanmean(R, axis=0)
    out["per_layer"] = [
        {"col": int(c), "layer_id": int(geom["layer_ids"][c]),
         "depth": float(depths[i]),
         "rho_bar": (float(rho_by_layer[i]) if math.isfinite(rho_by_layer[i]) else None),
         "sd_cluster": (float(np.nanstd(R[:, i], ddof=1)) if np.isfinite(R[:, i]).sum() > 1 else None),
         "n_clusters": int(np.isfinite(R[:, i]).sum())}
        for i, c in enumerate(cols)]

    if geom["is_dsp"]:
        # a single-capture front-end has one layer; its cluster-mean rho is the
        # cue sensitivity. The non-input averaging path is not taken.
        out["dsp_rho"] = theta
    else:
        out["dsp_rho"] = None
        # companion: native-grid unweighted mean over non-input layers
        vals = rho_by_layer[np.isfinite(rho_by_layer)]
        out["mean_nonin_native"] = float(np.mean(vals)) if vals.size else None

    out.update({
        "rho_aul": theta,
        "n_clusters": n_clusters,
        "n_eff": n_clusters,
        "sd_cluster": sd_cluster,
        "cluster_names": tab.cluster_names,
        "per_cluster_aul": [None if not math.isfinite(v) else float(v)
                            for v in per_cluster_aul],
    })

    # descriptive peak layer (never selection-adjusted, no p-value)
    if np.isfinite(rho_by_layer).any():
        pk = int(np.nanargmax(np.abs(rho_by_layer)))
        out["peak_descriptive"] = {
            "layer_id": int(geom["layer_ids"][cols[pk]]),
            "rho_bar": float(rho_by_layer[pk]),
            "note": "conditional on the selected layer, not selection-adjusted"}

    # pooled rho companion
    pooled = [_spearman_rho(tab.x, Yn[:, i])[0] for i in range(Yn.shape[1])]
    out["rho_pooled_aul"] = aul_common_grid(pooled, depths, n_grid)

    # ---- bootstrap CI ----------------------------------------------------
    bs = bootstrap_aul(tab.x, Yn, tab.cluster, tab.plevel, depths,
                       n_boot, boot_seed, n_grid)
    draws = bs["aul_draws"]

    # delete-one-cluster jackknife on the rho_9 estimate
    jack = np.array([
        float(np.mean(np.delete(fin, i))) if fin.size > 1 else np.nan
        for i in range(fin.size)], dtype=float)
    ci = bca_ci(theta, draws, jack, 0.95, clip, n_clusters, bca_rules)
    out["ci_95"] = {k: ci[k] for k in ("ci_low", "ci_high", "ci_method",
                                       "ci_fallback_reason", "z0", "a_hat",
                                       "a_lo", "a_hi", "ci_percentile")}
    out["ci_width"] = (float(ci["ci_high"] - ci["ci_low"])
                       if math.isfinite(ci["ci_low"]) and math.isfinite(ci["ci_high"])
                       else None)
    out["ci_excludes_point_estimate"] = bool(
        math.isfinite(ci["ci_low"]) and math.isfinite(ci["ci_high"])
        and math.isfinite(theta) and not (ci["ci_low"] <= theta <= ci["ci_high"]))
    t_lo, t_hi = _percentile_ci(draws, float(an.get("tost_ci_level", 0.90)))
    tost_delta = float(an.get("tost_delta", 0.3))
    out["tost"] = {"tost_delta": tost_delta,
                   "tost_ci_level": float(an.get("tost_ci_level", 0.90)),
                   "tost_ci_low": t_lo, "tost_ci_high": t_hi,
                   "equivalent": bool(math.isfinite(t_lo) and math.isfinite(t_hi)
                                      and t_lo > -tost_delta and t_hi < tost_delta),
                   "n_eff": n_clusters}

    # ---- diagnostics ----------------------------------------------------
    da = float(bs["distinct_multiset_ratio"])
    if fin.size > 1:
        dev = np.abs(theta - jack)
        db = float(np.max(dev) / (np.sum(dev) + 1e-300))
    else:
        db = float("nan")
    n_deg = int(bs["n_degenerate_draws"])
    deg_frac = n_deg / float(max(1, n_boot * max(1, n_clusters)))
    out["diagnostics"] = {
        "distinct_multiset_ratio": da,
        "ci_discretization_flag": bool(da < float(diag.get("distinct_multiset_min", 0.99))),
        "influence_concentration": db,
        "influence_concentration_flag": bool(
            math.isfinite(db) and db > float(diag.get("influence_concentration_max", 0.30))),
        "n_degenerate_draws": n_deg,
        "degenerate_fraction": deg_frac,
        "ci_coarse": bool(deg_frac > float(diag.get("degenerate_draw_max_fraction", 0.05))),
        "max_influence_cluster": (tab.cluster_names[int(np.argmax(np.abs(theta - jack)))]
                                  if fin.size > 1 and tab.cluster_names else None),
    }
    out["no_interval_diagnostic_flag"] = not (out["diagnostics"]["ci_discretization_flag"]
                                              or out["diagnostics"]["influence_concentration_flag"])

    # ---- permutation p --------------------------------------------------
    perm = cluster_permutation_p(tab.x, Yn, tab.cluster, depths, theta,
                                 n_perm, perm_seed, n_grid)
    out["permutation"] = perm
    out["p_value"] = perm["p_permutation"]
    out["elapsed_s"] = round(time.time() - t0, 2)
    out["status"] = "ok"
    return out


def paired_cell(tab_a: ScoreTable, geom_a: dict, tab_b: ScoreTable,
                geom_b: dict, an: dict, label: str) -> dict:
    """Paired rho_9 delta on common stimuli, per-cluster differences, shared draws."""
    n_grid = int(an.get("aul_grid_points", 9))
    perm_seed = int(an.get("permutation_seed", 4242))
    boot_seed = int(an.get("bootstrap_seed", 42))
    n_boot = int(an.get("n_bootstrap", 10000))
    n_perm = int(an.get("n_permutation", 10000))

    common = sorted(set(tab_a.keys) & set(tab_b.keys))
    if len(common) < 20:
        return {"label": label, "status": "insufficient_common_stimuli",
                "n_common": len(common)}
    ia = {k: i for i, k in enumerate(tab_a.keys)}
    ib = {k: i for i, k in enumerate(tab_b.keys)}
    sel_a = np.array([ia[k] for k in common])
    sel_b = np.array([ib[k] for k in common])

    x = tab_a.x[sel_a]
    cl = tab_a.cluster[sel_a]
    pl = tab_a.plevel[sel_a]
    Ya = tab_a.Y[sel_a][:, geom_a["nonin_cols"]]
    Yb = tab_b.Y[sel_b][:, geom_b["nonin_cols"]]

    Ra, codes = cluster_rhos(x, Ya, cl)
    Rb, _ = cluster_rhos(x, Yb, cl)
    aul_a = aul_common_grid_rows(Ra, geom_a["depths"], n_grid)
    aul_b = aul_common_grid_rows(Rb, geom_b["depths"], n_grid)
    delta_c = aul_a - aul_b
    fin = delta_c[np.isfinite(delta_c)]
    delta = float(np.mean(fin)) if fin.size else float("nan")

    # cluster bootstrap CI on the mean paired difference (shared index)
    rng = np.random.default_rng(boot_seed)
    ri = ResampleIndex(cl, pl)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        _, picks = ri.draw(rng)
        da = np.empty(len(picks)); db_ = np.empty(len(picks))
        for i, rows in enumerate(picks):
            ra = _rho_for_rows(x, Ya, rows)
            rb = _rho_for_rows(x, Yb, rows)
            da[i] = aul_common_grid(ra, geom_a["depths"], n_grid)
            db_[i] = aul_common_grid(rb, geom_b["depths"], n_grid)
        with np.errstate(invalid="ignore"):
            draws[b] = float(np.nanmean(da - db_))
    lo, hi = _percentile_ci(draws, 0.95)
    tost_lo, tost_hi = _percentile_ci(draws, float(an.get("tost_ci_level", 0.90)))
    perm = signflip_permutation_p(delta_c, n_perm, perm_seed)
    tost_delta = float(an.get("tost_delta", 0.3))
    return {
        "label": label, "status": "ok",
        "delta": delta,
        "rho_aul_a": float(np.nanmean(aul_a)), "rho_aul_b": float(np.nanmean(aul_b)),
        "ci_low": lo, "ci_high": hi,
        "p_permutation": perm["p_permutation"],
        "perm_floor": perm["perm_floor"],
        "n_clusters": int(fin.size), "n_common_stimuli": len(common),
        "per_cluster_delta": [None if not math.isfinite(v) else float(v)
                              for v in delta_c],
        "cluster_names": [tab_a.cluster_names[c] for c in np.unique(cl)],
        "tost": {"tost_delta": tost_delta, "tost_ci_low": tost_lo,
                 "tost_ci_high": tost_hi,
                 "equivalent": bool(math.isfinite(tost_lo) and math.isfinite(tost_hi)
                                    and tost_lo > -tost_delta and tost_hi < tost_delta)},
    }




# -----------------------------------------------------------------------------
# Exact-AUC cell and paired exact-AUC contrast (shared draws)
# -----------------------------------------------------------------------------

def summarise_draws(depths, layer_draws):
    """Percentile intervals for every depth summary from shared per-capture draws."""
    s = summaries_rows(depths, layer_draws)
    out = {}
    for k, v in s.items():
        lo, hi = percentile_ci(v, 0.95)
        out[k] = {"ci_low": lo, "ci_high": hi,
                  "boot_mean": float(np.nanmean(v)),
                  "boot_sd": float(np.nanstd(v)),
                  "ci_width": (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else None}
    return out


def exact_auc_cell(t, key, cue):
    """Exact-AUC point, per-capture and depth-summary intervals for one cell.

    `t` is table_dict(build_score_table(...), layer_geometry(...)).
    """
    t0 = time.time()
    keep = np.ones(t["x"].size, dtype=bool)
    x, cl, pl = t["x"][keep], t["cluster"][keep], t["plevel"][keep]
    Yn = t["Y"][keep][:, t["nonin_cols"]]
    depths = t["depths"]
    R, _ = cluster_rhos(x, Yn, cl)
    with np.errstate(invalid="ignore"):
        rho_layer = np.nanmean(R, axis=0)
    point = all_summaries(depths, rho_layer)
    # pooled companion (one Spearman over all rows, no per-anchor conditioning),
    # reported next to the anchor-conditioned estimate as a robustness check
    pooled_layer = np.array([_spearman_rho(x, Yn[:, i])[0]
                             for i in range(Yn.shape[1])], dtype=float)
    pooled = all_summaries(depths, pooled_layer)
    bs = bootstrap_aul(x, Yn, cl, pl, depths, N_BOOT, SEED, 9)
    LD = bs["layer_draws"]
    ci = summarise_draws(depths, LD)
    per_cap = []
    for i in range(LD.shape[1]):
        lo, hi = percentile_ci(LD[:, i], 0.95)
        per_cap.append({"capture_index": i, "layer_id": int(t["layer_ids"][t["nonin_cols"][i]]),
                        "depth": float(depths[i]), "rho": float(rho_layer[i]),
                        "ci_low": lo, "ci_high": hi})
    return {"key": key, "cue": cue, "status": "ok", "point": point, "ci": ci,
            "pooled_point": pooled,
            "anchor_minus_pooled": {k: point[k] - pooled[k] for k in point},
            "per_capture": per_cap, "n_rows": int(keep.sum()),
            "n_anchors": int(np.unique(cl).size),
            "elapsed_s": round(time.time() - t0, 1)}


def exact_auc_paired(ta, tb, model, cue, fe):
    """Delta_E = checkpoint minus front-end under every depth summary.

    One resample is shared by the checkpoint and its front-end; both sides use
    the stimuli scorable on both. `ta`, `tb` as in exact_auc_cell.
    """
    t0 = time.time()
    ka, kb = np.ones(ta["x"].size, dtype=bool), np.ones(tb["x"].size, dtype=bool)
    ia = {k: i for i, k in enumerate(ta["keys"]) if ka[i]}
    ib = {k: i for i, k in enumerate(tb["keys"]) if kb[i]}
    common = sorted(set(ia) & set(ib))
    if len(common) < 20:
        return {"model": model, "cue": cue, "status": "insufficient_common"}
    sa = np.array([ia[k] for k in common])
    sb = np.array([ib[k] for k in common])
    x, cl, pl = ta["x"][sa], ta["cluster"][sa], ta["plevel"][sa]
    Ya = ta["Y"][sa][:, ta["nonin_cols"]]
    Yb = tb["Y"][sb][:, tb["nonin_cols"]]
    if Yb.shape[1] != 1:
        return {"model": model, "cue": cue, "status": f"front_end_not_single_capture:{Yb.shape[1]}"}
    depths = ta["depths"]
    Yab = np.hstack([Ya, Yb])
    K = Ya.shape[1]

    Rab, _ = cluster_rhos(x, Yab, cl)
    d_per_cluster_cap = Rab[:, :K] - Rab[:, K:K + 1]          # [C, K]
    with np.errstate(invalid="ignore"):
        delta_cap = np.nanmean(d_per_cluster_cap, axis=0)
    summ_c = np.array([[all_summaries(depths, Rab[c, :K])[s]
                        for s in SUMMARY_NAMES] for c in range(Rab.shape[0])])
    with np.errstate(invalid="ignore"):
        delta_summ = np.nanmean(summ_c - Rab[:, K:K + 1], axis=0)

    rng = np.random.default_rng(SEED)
    ri = ResampleIndex(cl, pl)
    Dsumm = np.full((N_BOOT, len(SUMMARY_NAMES)), np.nan)
    Dcap = np.full((N_BOOT, K), np.nan)
    for b in range(N_BOOT):
        chosen, picks = ri.draw(rng)
        Rb_ = np.empty((len(picks), K + 1))
        for i, rows in enumerate(picks):
            r = _rho_for_rows(x, Yab, rows)
            att = 0
            while not np.all(np.isfinite(r)) and att < 10:
                c = chosen[i]
                parts = [g[rng.integers(0, g.size, size=g.size)] for g in ri.groups[c]]
                r = _rho_for_rows(x, Yab, np.concatenate(parts))
                att += 1
            Rb_[i] = r
        with np.errstate(invalid="ignore"):
            Dcap[b] = np.nanmean(Rb_[:, :K] - Rb_[:, K:K + 1], axis=0)
            S = np.array([[all_summaries(depths, Rb_[i, :K])[s]
                           for s in SUMMARY_NAMES] for i in range(Rb_.shape[0])])
            Dsumm[b] = np.nanmean(S - Rb_[:, K:K + 1], axis=0)

    out_summ = {}
    for j, s in enumerate(SUMMARY_NAMES):
        lo, hi = percentile_ci(Dsumm[:, j], 0.95)
        out_summ[s] = {"delta": float(delta_summ[j]), "ci_low": lo, "ci_high": hi,
                       "resolves_above": bool(np.isfinite(lo) and lo > 0),
                       "resolves_below": bool(np.isfinite(hi) and hi < 0),
                       "ci_width": (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else None}
    out_cap = []
    for i in range(K):
        lo, hi = percentile_ci(Dcap[:, i], 0.95)
        out_cap.append({"capture_index": i,
                        "layer_id": int(ta["layer_ids"][ta["nonin_cols"][i]]),
                        "depth": float(depths[i]), "delta": float(delta_cap[i]),
                        "ci_low": lo, "ci_high": hi,
                        "resolves_above": bool(np.isfinite(lo) and lo > 0),
                        "resolves_below": bool(np.isfinite(hi) and hi < 0)})
    # sign-flip permutation on the per-anchor differences, rho_9 endpoint
    j9 = SUMMARY_NAMES.index("rho9")
    perc = summ_c[:, j9] - Rab[:, K]
    perm = signflip_permutation_p(perc, 10000, 4242)
    return {"model": model, "cue": cue, "front_end": fe, "status": "ok",
            "n_common_stimuli": len(common), "n_anchors": int(Rab.shape[0]),
            "by_summary": out_summ, "per_capture": out_cap,
            "p_permutation_rho9": perm["p_permutation"],
            "elapsed_s": round(time.time() - t0, 1)}




# ---------------------------------------------------------------------------
# Glue and a small command-line interface
# ---------------------------------------------------------------------------

def table_dict(tab: ScoreTable, geom: dict) -> dict:
    """ScoreTable + layer_geometry -> the dict the exact-AUC functions read."""
    return {"x": tab.x, "Y": tab.Y, "cluster": tab.cluster, "plevel": tab.plevel,
            "keys": list(tab.keys), "cluster_names": list(tab.cluster_names),
            "layer_ids": np.asarray(geom["layer_ids"]),
            "nonin_cols": np.asarray(geom["nonin_cols"]),
            "depths": np.asarray(geom["depths"], dtype=float)}


def seq_tone(stim_cfg: dict, cue: str) -> Tuple[float, float]:
    """Clip length and tone length used by the frame mapping of a cue."""
    pc = stim_cfg["stimuli"]["temp_proximity" if cue in RERENDER_CUES else cue]
    seq = float(pc.get("sequence_duration_s") or pc.get("total_duration_s")
                or pc.get("tone_duration_s", 1.0))
    return seq, float(pc.get("tone_duration_s", 0.050))


def score_readout(features: Path, stimuli: Path, cue: str, layer_ids: List[int],
                  stim_cfg: dict) -> Tuple[ScoreTable, dict]:
    """Score one feature file `<features>` against `<stimuli>/<cue>/metadata.json`."""
    with open(stimuli / cue / "metadata.json") as fh:
        meta = json.load(fh)
    meta = meta["stimuli"] if isinstance(meta, dict) else meta
    seq, tone = seq_tone(stim_cfg, cue)
    tab = build_score_table(features, meta, cue, seq, tone, stim_cfg)
    geom = layer_geometry({"layers_sample": layer_ids}, tab.Y.shape[1])
    return tab, geom


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score feature .npz files with the BCS operator.")
    ap.add_argument("--features", type=Path, required=True,
                    help="npz with one [n_layers, T, H] array per stimulus key")
    ap.add_argument("--stimuli", type=Path, required=True,
                    help="stimulus root holding <cue>/metadata.json")
    ap.add_argument("--cue", required=True,
                    choices=CORE_CUES + list(RERENDER_CUES))
    ap.add_argument("--layers", required=True,
                    help="comma-separated capture ids stored in the npz, e.g. 0,3,6")
    ap.add_argument("--frontend", type=Path, help="front-end npz for a paired Delta_E")
    ap.add_argument("--frontend-layers", default="0")
    ap.add_argument("--stimulus-config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "stimuli.yaml")
    ap.add_argument("--n-bootstrap", type=int, default=ANALYSIS["n_bootstrap"],
                    help="bootstrap draws; also sets the permutation count")
    ap.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    a = ap.parse_args()

    import yaml
    with open(a.stimulus_config) as fh:
        stim_cfg = yaml.safe_load(fh)
    global N_BOOT
    N_BOOT = a.n_bootstrap
    an = dict(ANALYSIS, n_bootstrap=a.n_bootstrap, n_permutation=a.n_bootstrap)

    ids = [int(v) for v in a.layers.split(",")]
    tab, geom = score_readout(a.features, a.stimuli, a.cue, ids, stim_cfg)
    out = {"rho9_cell": analyse_cell(tab, geom, an, str(a.features.name)),
           "exact_auc_cell": exact_auc_cell(table_dict(tab, geom), a.features.stem, a.cue)}
    if a.frontend is not None:
        fids = [int(v) for v in a.frontend_layers.split(",")]
        ftab, fgeom = score_readout(a.frontend, a.stimuli, a.cue, fids, stim_cfg)
        out["rho9_paired"] = paired_cell(tab, geom, ftab, fgeom, an, "paired")
        out["exact_auc_paired"] = exact_auc_paired(
            table_dict(tab, geom), table_dict(ftab, fgeom),
            a.features.stem, a.cue, a.frontend.stem)
    text = json.dumps(out, indent=1, default=_json_default)
    if a.out:
        a.out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
