#!/usr/bin/env python3
"""Re-render and random-decoder controls for MusicGen-S/M/L (temporal proximity).

Features: extraction/extract_musicgen.py on the re-rendered stimuli (cue
tp_rr_main: per anchor the triplet/onset counts, duty cycle and level are
fixed and the span-ratio relation is reversed), once with the pretrained
decoder and once per seed 4242, 4243, 4244 with --init-seed (a randomly
initialised decoder of the same configuration; the pretrained EnCodec
tokenizer is kept).

Scoring (this script): rho_9 cells with bcs.score.analyse_cell and paired
rho_9 contrasts on the common stimuli with bcs.score.paired_cell (one shared
bootstrap draw per pair, 10,000 draws, seed 42):
  trained minus random, per size and seed (9 contrasts);
  large minus small, trained and per random seed.
Prints the contrasts stored in results/rerender_random.json,
results/rerender_random_intervals.json (the nine trained-minus-random
intervals) and trained_LS of results/rerender_size_contrast.json.

Usage
  python3 controls/rerender_random.py --features-dir <dir with <key>_tp_rr_main.npz> \
      --stimuli <re-rendered stimulus root (stimuli/generate_rerender.py)> [--n-bootstrap 10000]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs.score as B  # noqa: E402

CUE = "tp_rr_main"
SIZES = {"small": [0, 3, 6, 9, 12, 15, 18, 21, 23],
         "medium": [0, 6, 12, 18, 24, 30, 36, 42, 47],
         "large": [0, 6, 12, 18, 24, 30, 36, 42, 47]}
SEEDS = (4242, 4243, 4244)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=B.ANALYSIS["n_bootstrap"])
    a = ap.parse_args()
    import yaml
    stim_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "stimuli.yaml").read_text())
    an = dict(B.ANALYSIS, n_bootstrap=a.n_bootstrap, n_permutation=a.n_bootstrap)
    tabs, cells, pairs = {}, {}, {}
    for size, ids in SIZES.items():
        for seed in (None,) + SEEDS:
            key = "musicgen_%s" % size + ("" if seed is None else "_randinit_s%d" % seed)
            npz = a.features_dir / ("%s_%s.npz" % (key, CUE))
            tabs[key] = B.score_readout(npz, a.stimuli, CUE, ids, stim_cfg)
            cells[key] = B.analyse_cell(*tabs[key], an, key)["rho_aul"]
    for size in SIZES:
        for seed in SEEDS:
            t, r = "musicgen_%s" % size, "musicgen_%s_randinit_s%d" % (size, seed)
            p = B.paired_cell(*tabs[t], *tabs[r], an, "%s-minus-%s" % (t, r))
            pairs["trained_minus_randinit_%s_s%d" % (size, seed)] = {
                k: p[k] for k in ("delta", "ci_low", "ci_high")}
    for tag, (l, s) in {"trained": ("musicgen_large", "musicgen_small"),
                        **{"randinit_s%d" % d: ("musicgen_large_randinit_s%d" % d,
                                                "musicgen_small_randinit_s%d" % d) for d in SEEDS}}.items():
        p = B.paired_cell(*tabs[l], *tabs[s], an, "large-minus-small")
        pairs["large_minus_small_" + tag] = {k: p[k] for k in ("delta", "ci_low", "ci_high")}
    print(json.dumps({"rho9": cells, "pairs": pairs}, indent=1))


if __name__ == "__main__":
    main()
