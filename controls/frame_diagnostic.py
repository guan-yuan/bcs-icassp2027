#!/usr/bin/env python3
"""Frame-grid diagnostic per (read-out, cue): recomputes and checks the `cells` of results/frame_diagnostic.json.

For every read-out and cue this calls extraction/grid_common.frame_diagnostics
on that read-out's stored features:

  * n_frames per stimulus = T of its [L, T, H] array (read from the .npy header);
  * n_same_frame_onset_pairs = consecutive annotated onsets that map to the same
    frame, frame = clamp(round(t * n_frames / T_clip)) (the synchronous onset
    level is excluded);
  * degenerate_row_share = share of (capture, frame) rows with norm <= 1e-6;
  * flag_coarse_frame_rate = pairs > 0 or n_frames_min < 10 or share >= 0.01.

The flag is the dagger ("coarse frame grid") of Fig. 2 and Table 3.  It is a
mark only: every pairing is counted.

Inputs: the per-read-out feature files (one .npz per read-out and cue, named
<readout>_<cue>.npz, one [L, T, H] array per stimulus stem) written by
extraction/, and the stimulus set (stimuli/, BCS_STIM_ROOT layout with
<cue>/metadata.json).  Producing them needs the model weights and a GPU; this
step itself runs on CPU.

Usage
  python3 controls/frame_diagnostic.py --features <dir of .npz> --stimuli <stimulus root> \
      --out frame_diagnostic_new.json [--readouts a,b,...] [--cues c,...] \
      [--check results/frame_diagnostic.json] [--workers 8]

--check compares every computed cell field by field with the `cells` of the
given file; a cell or field missing there is reported as MISSING, a different
value as DIFF, and either makes the exit status 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "extraction"))
import grid_common  # noqa: E402

FIELDS = ["n_same_frame_onset_pairs", "n_frames_min", "n_frames_max",
          "degenerate_row_share", "flag_coarse_frame_rate", "n_stimuli"]
FLAG_RULE = ("flag_coarse_frame_rate = n_same_frame_onset_pairs > 0 or "
             "n_frames_min < 10 or degenerate_row_share >= 0.01")


def frame_counts(path: Path) -> dict:
    """{stem: T} from the npy headers inside the npz (arrays are not loaded)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            stem = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            if stem == "_meta":
                continue
            with z.open(info) as f:
                ver = np.lib.format.read_magic(f)
                hdr = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                       else np.lib.format.read_array_header_2_0(f))
            if len(hdr[0]) != 3:
                raise SystemExit("%s: %s is not [L, T, H]: %r" % (path, stem, hdr[0]))
            out[stem] = int(hdr[0][1])
    return out


class LazyArrays:
    """reps.get(stem) -> [L, T, H] array, loaded one at a time."""

    def __init__(self, npz, stems):
        self.npz, self.stems = npz, stems

    def get(self, stem):
        return self.npz[stem] if stem in self.stems else None


def one(job):
    features, stimuli, readout, cue = job
    grid_common.STIM = Path(stimuli)
    path = Path(features) / ("%s_%s.npz" % (readout, cue))
    nframes = frame_counts(path)
    npz = np.load(path, allow_pickle=False)
    res = grid_common.frame_diagnostics(cue, nframes, LazyArrays(npz, set(nframes)))
    meta = {e["file"].replace(".wav", "") for e in grid_common.paradigm_meta(cue)}
    if set(nframes) != meta:
        raise SystemExit("%s|%s: %d arrays without metadata, %d metadata entries without array"
                         % (readout, cue, len(set(nframes) - meta), len(meta - set(nframes))))
    res["n_stimuli"] = len(nframes)
    return "%s|%s" % (readout, cue), {f: res[f] for f in FIELDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--stimuli", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--readouts", default=None,
                    help="comma list; default: the read-outs of results/frame_diagnostic.json "
                         "that an extractor writes")
    ap.add_argument("--cues", default="freq_proximity,temp_proximity,harmonicity,onset_sync",
                    help="comma list; default: the four cues of the paper")
    ap.add_argument("--check", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.readouts:
        readouts = a.readouts.split(",")
    else:
        shipped = json.load(open(HERE / "results" / "frame_diagnostic.json"))
        readouts = [r for r in shipped["readouts"]["checkpoints"] + shipped["readouts"]["front_ends"]
                    if r != "fe_spectrostream"]      # same cells as fe_spectrostream_rvq
    jobs = [(str(a.features), str(a.stimuli), r, c)
            for r in readouts for c in a.cues.split(",")]
    cells = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for key, cell in ex.map(one, jobs):
            cells[key] = cell
            print("%-40s pairs=%-6d nmin=%-4d degen=%.6f flag=%s"
                  % (key, cell["n_same_frame_onset_pairs"], cell["n_frames_min"],
                     cell["degenerate_row_share"], cell["flag_coarse_frame_rate"]),
                  flush=True)
    with open(HERE / "extraction" / "grid_common.py", "rb") as fh:
        # LF-normalised, as in make_paper_numbers.py (CRLF checkouts hash the same)
        gsha = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    payload = {"function": "grid_common.frame_diagnostics", "flag_rule": FLAG_RULE,
               "grid_common_sha256": gsha, "cells": dict(sorted(cells.items()))}
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    if a.check:
        ref = json.load(open(a.check)).get("cells", {})
        bad = []
        for k in sorted(cells):
            if k not in ref:
                bad.append(("MISSING", k, "(cell not in reference)"))
                continue
            for f in FIELDS:
                if f not in ref[k]:
                    bad.append(("MISSING", k, f, cells[k][f], "(field not in reference)"))
                elif cells[k][f] != ref[k][f]:
                    bad.append(("DIFF", k, f, cells[k][f], ref[k][f]))
        print("check against %s: %d cells, %d differences or missing entries"
              % (a.check, len(cells), len(bad)))
        for row in bad:
            print(*row)
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
