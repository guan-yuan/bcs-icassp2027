#!/usr/bin/env python3
"""Score MusicGen features extracted under the native token schedule (temporal cue).

Input: the three feature files written by extraction/extract_musicgen_native.py
(musicgen_<size>_temp_proximity.npz) and the EnCodec RVQ front-end features of the
same stimuli. For each size and read-out offset (-1, 0, +1 frames; see
controls/offsets.py) this prints the exact AUC and rho_9 with their intervals and
Delta_E against the front-end read at the same offset (shared-draw bootstrap,
10,000 draws, seed 42), in the layout of results/musicgen_native_schedule.json.

Usage
  python3 controls/native_schedule.py --features-dir <dir> \\
      --frontend <fe_encodec32_rvq_temp_proximity.npz> --stimuli <stimulus root> \\
      [--sizes small,medium,large] [--offsets -1,0,1] [--n-bootstrap 10000]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bcs.score as B  # noqa: E402
from offsets import shifted, _ORIG_TF  # noqa: E402

CUE = "temp_proximity"
LAYERS = {"small": [0, 3, 6, 9, 12, 15, 18, 21, 23],
          "medium": [0, 6, 12, 18, 24, 30, 36, 42, 47],
          "large": [0, 6, 12, 18, 24, 30, 36, 42, 47]}


def flat(cell, pair):
    out = {}
    for e in ("auc", "rho9"):
        out[e] = cell["point"][e]
        out[e + "_lo"] = cell["ci"][e]["ci_low"]
        out[e + "_hi"] = cell["ci"][e]["ci_high"]
    for e in ("auc", "rho9"):
        d = pair["by_summary"][e]
        out["dE_" + e] = d["delta"]
        out["dE_" + e + "_lo"] = d["ci_low"]
        out["dE_" + e + "_hi"] = d["ci_high"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--frontend", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--offsets", default="-1,0,1")
    ap.add_argument("--n-bootstrap", type=int, default=B.ANALYSIS["n_bootstrap"])
    a = ap.parse_args()
    import yaml
    stim_cfg = yaml.safe_load((HERE.parent / "config" / "stimuli.yaml").read_text())
    B.N_BOOT = a.n_bootstrap
    offsets = [int(v) for v in a.offsets.split(",")]
    native = {}
    for size in a.sizes.split(","):
        key = "musicgen_" + size
        native[key] = {}
        for o in offsets:
            B._time_to_frame = shifted(o)
            try:
                tab, geom = B.score_readout(a.features_dir / ("%s_%s.npz" % (key, CUE)),
                                            a.stimuli, CUE, LAYERS[size], stim_cfg)
                ftab, fgeom = B.score_readout(a.frontend, a.stimuli, CUE, [0], stim_cfg)
            finally:
                B._time_to_frame = _ORIG_TF
            t, f = B.table_dict(tab, geom), B.table_dict(ftab, fgeom)
            cell = B.exact_auc_cell(t, key, CUE)
            pair = B.exact_auc_paired(t, f, key, CUE, a.frontend.stem)
            native[key][str(o)] = dict(flat(cell, pair), n_rows=cell["n_rows"],
                                       per_capture_rho=[c["rho"] for c in cell["per_capture"]])
            print(key, o, "auc %+.3f  dE_auc %+.3f" % (native[key][str(o)]["auc"],
                                                       native[key][str(o)]["dE_auc"]), flush=True)
    print(json.dumps({"native": native}, indent=1))


if __name__ == "__main__":
    main()
