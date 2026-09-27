#!/usr/bin/env python3
"""Extractor: Magenta-RealTime-2 small (230M) / base (2.4B).

Forward:
  SpectroStream 25 Hz RVQ encodes the stimulus -> fed to the raw checkpoint as
  the teacher-forced target (no sampling, single forward) -> hook the temporal
  (frame-axis) decoder blocks.  The Depthformer's depth/codebook axis is not
  taken; conditioning (style / notes / drums) is null and CFG is off.

Hook note: the checkpoint is a Flax/JAX safetensors file, so
there are no torch forward hooks.  Flax `capture_intermediates` does not reach
these blocks either -- `sl.Serial` invokes children through a path Flax does not
intercept.  The blocks are therefore walked explicitly inside an `apply`
`method=`, and on every clip the walk's last output is compared with the
library's own `temporal_body.layer(...)`; the largest absolute difference is
recorded as `block_walk_max_abs_delta_vs_library` in the diag file.

Usage: BCS_STIM_ROOT=<stimulus root> extract_magenta_rt2.py --size mrt2_small|mrt2_base --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# Autotuning is off so reruns are bit-identical.
# Must be set before jax is imported.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_gpu_autotune_level=0"
                           " --xla_gpu_deterministic_ops=true").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("MAGENTA_RT_SRC", ""))  # clone of magenta-realtime
sys.path.insert(0, os.path.join(os.environ.get("MAGENTA_RT_SRC", ""), "magenta_rt/_vendor/sequence-layers"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import sequence_layers.jax as sl                              # noqa: E402
import safetensors.flax as stf                                # noqa: E402
import flax.traverse_util as ftu                              # noqa: E402

from magenta_rt.jax import model as model_configs             # noqa: E402
from magenta_rt.jax import spectrostream                      # noqa: E402
from grid_common import (ALL_PARADIGMS, DEFAULT_PARADIGMS, Progress, T_SEQ,  # noqa: E402
                       frame_diagnostics, iter_paradigm, layer_ids,
                       load_wav_mono_np, save_npz, sha256_file)

HF_ID = "google/magenta-realtime-2"
HF_REV = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
SR = 48000                       # SpectroStream operates on 48 kHz stereo
RVQ = 12                         # SPECTROSTREAM.rvq_truncation_level
NUM_RESERVED_TOKENS = 6

# Model keys; npz filenames carry the model key.
MODEL_KEY = {"mrt2_small": "magenta_rt2_small", "mrt2_base": "magenta_rt2_base"}


def snapshot() -> Path:
    hits = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub/models--google--magenta-realtime-2/")
                     + 
                     f"snapshots/{HF_REV}")
    assert hits, "MRT2 snapshot not found"
    return Path(hits[0])


def _load(p) -> dict:
    return ftu.unflatten_dict({tuple(k.split("/")): v
                               for k, v in stf.load_file(str(p)).items()})


def build_spectrostream(snap: Path):
    b = snap / "resources" / "spectrostream"
    params = {"params": {
        "encoder": _load(b / "encoder.safetensors")["params"]["encoder"],
        "decoder": _load(b / "decoder.safetensors")["params"]["decoder"],
        "quantizer": _load(b / "quantizer.safetensors")["params"]["soundstream"]["quantizer"],
    }}
    cfg = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=RVQ, encoded_truncation_level=RVQ,
        use_unique_codes=False)
    return cfg.make(), params, cfg


def null_source(mcls, n_frames: int) -> np.ndarray:
    """Null style / zero notes / no CFG. Built from the model's
    own input_configs (no hard-coded channel indices)."""
    from magenta_rt.jax.system import discretize_cfg
    chans, col = [], 0
    for c in mcls.input_configs:
        n = c.rvq_truncation_level
        if c.cfg_scale_keys:                    # CFG scale 1.0 == no guidance
            vals = [discretize_cfg(1.0, c.step, c.codebook_size - 1)
                    for _ in c.cfg_scale_keys][:n]
        else:
            vals = [-1] * n                     # -1 == masked / unconditioned
        chans.extend(vals)
        col += n
    assert col == mcls.input_num_channels, (col, mcls.input_num_channels)
    src = np.array(chans, np.int32) + (NUM_RESERVED_TOKENS + 1)
    return np.tile(src[None, None, :], (1, n_frames, 1))


def make_probe(verify: bool):
    def probe(mod, source, target):
        enc = mod.encoder.body.layer(source, training=False)
        constants = {"source": enc}
        dec = mod.decoder
        x = target.pad_time(pad_left=1, pad_right=0, valid=True,
                            pad_value=dec.config.sos_id)
        embedded = dec.embedder(x, training=False)
        h = embedded.apply_values(jnp.mean, axis=-2)[:, :-1]
        tb = dec.temporal_body
        assert len(tb.layers) == 1, [type(l).__name__ for l in tb.layers]
        outs, cur = [], h
        for blk in tb.layers[0].layers:         # x_layers_0 .. x_layers_{L-1}
            cur = blk.layer(cur, training=False, constants=constants)
            outs.append(cur.values)
        if verify:
            ref = tb.layer(h, training=False, constants=constants).values
            delta = jnp.max(jnp.abs(outs[-1].astype(jnp.float32)
                                    - ref.astype(jnp.float32)))
            return outs, delta
        return outs, jnp.float32(0.0)
    return probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["mrt2_small", "mrt2_base"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS[:4]),
                    help="comma-separated cues (default: the four cues of the paper)")
    args = ap.parse_args()
    if not os.environ.get("BCS_STIM_ROOT"):
        ap.error("set BCS_STIM_ROOT to the stimulus root written by stimuli/generate_stimuli.py")

    snap = snapshot()
    ss, ssp, sscfg = build_spectrostream(snap)
    mcls = model_configs.get_model_class(args.size)()
    ed = mcls.depthformer_config().make()
    ckpt = snap / "checkpoints" / f"{args.size}.safetensors"
    edp = {"params": _load(ckpt)["params"]["depthformer"]}

    mkey = MODEL_KEY[args.size]
    n_blocks = mcls.decoder_temporal_size.num_layers
    hidden = mcls.decoder_temporal_size.model_dims
    ids = layer_ids(n_blocks)
    probe = make_probe(verify=True)
    print(f"[{mkey}] L={n_blocks} H={hidden} layer_ids={ids}", flush=True)

    outdir = Path(args.out)
    diag = {"model_key": mkey, "hf_id": HF_ID, "hf_revision": HF_REV,
            "architecture": "ar", "checkpoint_file": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "layers_total": n_blocks, "hidden_dim": hidden, "layer_ids": ids,
            "dtype": f"param={mcls.param_dtype.__name__}/"
                     f"compute={mcls.compute_dtype.__name__}",
            "captured_module": "decoder.temporal_body.layers[0].layers[i] "
                               "(frame-axis blocks; depth/codebook axis not taken)",
            "conditioning": "null style (masked), zero notes/drums, CFG=1.0 (off)",
            "sample_rate": SR, "front_end": "SpectroStream 25 Hz RVQ (12 levels)",
            "teacher_forcing": True, "sampling": False,
            "paradigms": {}}

    max_verify = 0.0
    for pdm in args.paradigms.split(","):
        reps, nframes, shapes = {}, {}, None
        items = iter_paradigm(pdm)
        prog = Progress(f"{mkey}/{pdm}", len(items))
        for stem, path in items:
            wav = load_wav_mono_np(path, SR)                  # mono @ 48 kHz
            stereo = np.stack([wav, wav], -1)[None]           # codec is stereo
            codes = ss.apply(ssp, sl.Sequence.from_values(jnp.asarray(stereo)),
                             training=False,
                             method=spectrostream.SpectroStream.waveform_to_codes)
            tgt = codes.values[:, :, :RVQ]
            src = null_source(mcls, tgt.shape[1])
            outs, delta = ed.apply(
                edp, sl.Sequence.from_values(jnp.asarray(src)),
                sl.Sequence.from_values(tgt), method=probe,
                rngs={"random": jax.random.PRNGKey(0)})
            max_verify = max(max_verify, float(delta))
            arr = np.stack([np.asarray(outs[i][0], dtype=np.float32) for i in ids])
            reps[stem] = arr
            nframes[stem] = arr.shape[1]
            prog.tick()
            if shapes is None:
                shapes = {"codec_frames": int(tgt.shape[1]),
                          "rvq_levels_used": RVQ,
                          "shape_before_flatten": list(arr.shape),
                          "shape_after_flatten": list(arr.shape)}
        secs = prog.done()
        npz = outdir / f"{mkey}_{pdm}.npz"
        save_npz(npz, reps, diag)
        fr = float(np.mean(list(nframes.values()))) / T_SEQ[pdm]
        diag["paradigms"][pdm] = {
            "n_stimuli": len(reps), "shapes": shapes,
            "measured_frame_rate_hz": fr, "npz": str(npz),
            "npz_bytes": npz.stat().st_size,
            "extract_seconds": round(secs, 1),
            "seconds_per_clip": round(secs / max(len(reps), 1), 4),
            **frame_diagnostics(pdm, nframes, reps)}
        del reps

    try:
        st = jax.local_devices()[0].memory_stats() or {}
        peak = float(st.get("peak_bytes_in_use", 0)) / 1e9
    except Exception:
        peak = float("nan")
    diag["vram_peak_reserved_gb"] = round(peak, 3)
    diag["block_walk_max_abs_delta_vs_library"] = max_verify
    (outdir / f"{mkey}_diag.json").write_text(json.dumps(diag, indent=2))
    print(f"[{mkey}] done; VRAM peak {peak:.2f} GB; "
          f"block-walk verify delta {max_verify:g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
