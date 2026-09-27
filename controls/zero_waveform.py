#!/usr/bin/env python3
"""All-zero input control for temporal proximity.

Step 1 (make): write one WAV whose samples are exactly zero, with the sample
rate, channel layout, subtype and length of a reference stimulus.
Step 2 (extract): put zero.wav in <zero root>/temp_proximity/ next to a
metadata.json {"stimuli": [{"file": "zero.wav"}]} and run the read-out's
extractor on <zero root> with --paradigms temp_proximity (root via --stim_root,
--stim-root or BCS_STIM_ROOT). Extractors built on extraction/frontend_common.py
also need --limit 1 and write <key>_temp_proximity_partial.npz;
extract_dsp_frontends.py takes no --paradigms. Z is the entry `zero` of the
feature file.
Step 3 (score): read Z at each temporal stimulus's own onset frames with the
same function as the fixed-trajectory control (only the array differs).

Usage
  python3 controls/zero_waveform.py make  --reference <any stimulus>.wav \
      --out <zero root>/temp_proximity/zero.wav
  python3 extraction/extract_dsp_frontends.py --stim_root <zero root> --out <dir>
  BCS_STIM_ROOT=<zero root> python3 extraction/extract_fe_pupujepa_logspec.py \
      --out <dir> --paradigms temp_proximity --limit 1
  python3 controls/zero_waveform.py score --array <dir>/logmel_temp_proximity.npz --key zero \
      --metadata <stimuli>/temp_proximity/metadata.json --layers 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixed_trajectory import score_fixed  # noqa: E402


def make(reference: Path, out: Path) -> dict:
    info = sf.info(str(reference))
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((info.frames, info.channels), dtype=np.float64)
    sf.write(str(out), data if info.channels > 1 else data[:, 0], info.samplerate,
             subtype=info.subtype, format=info.format)
    chk = sf.info(str(out))
    same = (chk.samplerate, chk.channels, chk.frames, chk.subtype, chk.format) == \
           (info.samplerate, info.channels, info.frames, info.subtype, info.format)
    y, _ = sf.read(str(out))
    if not same or np.any(y != 0):
        raise SystemExit("zero clip does not match the reference format or is not all-zero")
    return {"samplerate": chk.samplerate, "channels": chk.channels, "frames": chk.frames}


def score(array: Path, key: str, metadata: Path, layers) -> dict:
    Z = np.load(array, allow_pickle=True)[key].astype(np.float64)
    meta = json.loads(metadata.read_text())
    midx = {e['file'][:-4]: e for e in (meta['stimuli'] if isinstance(meta, dict) else meta)}
    keys = sorted(midx)
    rho, n_anchor = score_fixed(Z, list(layers), keys, midx)
    return {"rho_zero": rho, "n_anchors_scored": n_anchor, "shape": list(Z.shape),
            "frames_identical": bool(np.all(Z == Z[:, :1, :]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--array", type=Path, required=True)
    s.add_argument("--key", default="zero"); s.add_argument("--metadata", type=Path, required=True)
    s.add_argument("--layers", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        print(json.dumps(make(a.reference, a.out)))
    else:
        print(json.dumps(score(a.array, a.key, a.metadata, [int(v) for v in a.layers.split(",")]), indent=1))


if __name__ == "__main__":
    main()
