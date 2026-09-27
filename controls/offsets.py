#!/usr/bin/env python3
"""Read-out offset sensitivity: shift every onset-to-frame index by o frames.

The frame of physical time t is round(t * T / T_clip), clamped to [0, T-1]
(bcs.score._time_to_frame). This control adds o in {-2, -1, 0, +1, +2} native
frames before clamping and rebuilds the score table; o = 0 reproduces the
primary map. Only frequency and temporal proximity map times to frames;
harmonicity and onset synchrony average the whole clip and are
offset-invariant. Exact-AUC cells and Delta_E against the front-end reference
at the same offset use the shared-draw bootstrap of bcs.score
(results/offsets.json, summarised in results/offset_counts.json).

Usage
  python3 controls/offsets.py --features <ckpt>_temp_proximity.npz --layers 0,3,6,... \
      --frontend <fe>_temp_proximity.npz --stimuli <stimulus root> --cue temp_proximity
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs.score as B  # noqa: E402

OFFSETS = (-2, -1, 0, 1, 2)
_ORIG_TF = B._time_to_frame


def shifted(o: int):
    def tf(t_sec, n_frames, seq_dur_s):
        return int(min(max(_ORIG_TF(t_sec, n_frames, seq_dur_s) + o, 0), n_frames - 1))
    return tf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--frontend", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--cue", default="temp_proximity", choices=["temp_proximity", "freq_proximity"])
    ap.add_argument("--n-bootstrap", type=int, default=B.ANALYSIS["n_bootstrap"])
    a = ap.parse_args()
    import yaml
    stim_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "stimuli.yaml").read_text())
    B.N_BOOT = a.n_bootstrap
    ids = [int(v) for v in a.layers.split(",")]
    out = []
    for o in OFFSETS:
        B._time_to_frame = shifted(o)
        try:
            tab, geom = B.score_readout(a.features, a.stimuli, a.cue, ids, stim_cfg)
            ftab, fgeom = B.score_readout(a.frontend, a.stimuli, a.cue, [0], stim_cfg)
        finally:
            B._time_to_frame = _ORIG_TF
        cell = B.exact_auc_cell(B.table_dict(tab, geom), a.features.stem, a.cue)
        pair = B.exact_auc_paired(B.table_dict(tab, geom), B.table_dict(ftab, fgeom),
                                  a.features.stem, a.cue, a.frontend.stem)
        out.append({"offset": o, "auc": cell["point"]["auc"],
                    "delta_E_auc": pair["by_summary"]["auc"]})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
