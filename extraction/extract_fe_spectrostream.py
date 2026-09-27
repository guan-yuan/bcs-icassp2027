#!/usr/bin/env python3
"""Front-end: SpectroStream 25 Hz de-quantised RVQ latent (`fe_spectrostream_rvq`).

The learned front-end shared by both Magenta-RT2 checkpoints, used as the
front-end reference of Magenta-RT2-S and Magenta-RT2-B.

What is extracted: the de-quantised RVQ embedding at the truncation level the
extractor feeds the model (`RVQ = 12`), i.e.

    waveform -> SpectroStream encoder -> RVQ -> codes[:, :, :12]
             -> quantizer.codes_to_embeddings(...)        [1, T, D]

This is the analogue of `fe_encodec32_rvq`, the token stream the MusicGen
decoder is teacher-forced on, de-quantised
back to a continuous vector so the identical cosine estimator applies.  The
codec's own decoder reaches the waveform through this same tensor
(`codes_to_features` -> decoder), so it is the codec's representation of the
stimulus, not a model-specific embedding: MRT2's own `dec.embedder` is a
per-checkpoint parameter (H = 1024 for small, 3072 for base) and therefore
cannot serve as a front-end shared by both checkpoints.

Output key: files are named `fe_spectrostream_rvq_<cue>.npz`, the key used in
config/checkpoints.yaml and in results/.

Determinism: the XLA autotune / deterministic-ops setting of
`extract_magenta_rt2.py` is inherited by importing that module before `jax`.
Audio loading, stereo duplication and sample rate are those of the MRT2
extractor.

Usage: BCS_STIM_ROOT=<stimulus root> extract_fe_spectrostream.py --out <dir> [--paradigms cue1,cue2,...]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# extract_magenta_rt2 sets XLA_FLAGS before importing jax, so import it first.
from extract_magenta_rt2 import (HF_ID, HF_REV, RVQ, SR,                    # noqa: E402
                          build_spectrostream, snapshot)
import jax.numpy as jnp                                              # noqa: E402
import sequence_layers.jax as sl                                     # noqa: E402
from magenta_rt.jax import spectrostream                             # noqa: E402

from frontend_common import (ALL_PARADIGMS, run_frontend, sha256_file)     # noqa: E402
from grid_common import load_wav_mono_np                               # noqa: E402

KEY = "fe_spectrostream_rvq"


def make_probe(use_gather: bool):
    """waveform -> [1, T, D] de-quantised RVQ embedding at `RVQ` levels."""
    def probe(mod, waveform):
        codes = mod.waveform_to_codes(waveform, training=False)
        codes = codes[:, :, :RVQ]
        return mod.quantizer.codes_to_embeddings(codes, use_gather=use_gather)
    return probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS))
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    args = ap.parse_args()

    snap = snapshot()
    ss, ssp, _cfg = build_spectrostream(snap)
    # Stored value = the exact de-quantisation, i.e. the sum of the 12 selected
    # codebook vectors (`use_gather=True`, the API default, and the same
    # definition `fe_encodec32_rvq` uses via `quantizer.decode`). The
    # codec's own decoder reaches the waveform through the algebraically
    # identical one-hot einsum branch (`codes_to_features`, use_gather=False),
    # which on GPU runs in TF32; the largest difference between the two on the
    # first clip is written to the diag file.
    probe_gather = make_probe(use_gather=True)
    probe_einsum = make_probe(use_gather=False)

    def encode(path, _state={}):
        wav = load_wav_mono_np(path, SR)                 # mono @ 48 kHz
        stereo = np.stack([wav, wav], -1)[None]          # codec is stereo
        seq = sl.Sequence.from_values(jnp.asarray(stereo))
        emb = ss.apply(ssp, seq, method=probe_gather)
        arr = np.asarray(emb.values[0], dtype=np.float32)     # [T, D]
        if "gather_delta" not in _state:                 # once, on the 1st clip
            e = ss.apply(ssp, seq, method=probe_einsum)
            _state["gather_delta"] = float(np.max(np.abs(
                np.asarray(e.values[0], dtype=np.float32) - arr)))
            _state["abs_max"] = float(np.abs(arr).max())
        encode.state = _state
        return arr

    outdir = Path(args.out)
    b = snap / "resources" / "spectrostream"
    diag = {
        "model_name": KEY,
        "full_name": "SpectroStream 25 Hz de-quantised RVQ latent "
                     "(12 levels, 48 kHz stereo)",
        "shared_by": ["magenta_rt2_small", "magenta_rt2_base"],
        "hf_id": HF_ID, "hf_revision": HF_REV,
        "sample_rate": SR, "rvq_levels_used": RVQ,
        "captured": "quantizer.codes_to_embeddings(waveform_to_codes(x)[:, :, :12], "
                    "use_gather=True)  # exact sum of the 12 selected codebook vectors",
        "encoder_sha256": sha256_file(b / "encoder.safetensors"),
        "quantizer_sha256": sha256_file(b / "quantizer.safetensors"),
        "determinism": "XLA_FLAGS autotune off + deterministic ops "
                       "(inherited from extract_magenta_rt2.py)",
        "role": "learned front-end shared by magenta_rt2_small and magenta_rt2_base",
    }

    d = run_frontend(KEY, encode, args.paradigms.split(","), outdir, diag,
                     limit=args.limit)
    st = getattr(encode, "state", {})
    delta = st.get("gather_delta")
    if delta is not None:
        import json
        d["rvq_gather_vs_einsum_max_abs_delta"] = delta
        d["rvq_first_clip_abs_max"] = st.get("abs_max")
        name = f"{KEY}_diag{'_partial' if args.limit else ''}.json"
        (outdir / name).write_text(json.dumps(d, indent=2))
        print(f"[{KEY}] gather-vs-einsum max|delta| = {delta:g} "
              f"(clip abs-max {st.get('abs_max'):g})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
