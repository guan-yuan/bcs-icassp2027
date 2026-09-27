#!/usr/bin/env python3
"""Front-end: SAME-L autoencoder latent (`fe_samel`).

The learned front-end of `stableaudio3_medium`, used as its front-end
reference (Stable Audio 3 minus SAME-L).

What is extracted: the tensor `extract_stable_audio3.py` calls `z0`:

    mono -> stereo duplicate -> ae.encode_audio(x, chunked=False)   [1, 256, T]

i.e. the deterministic mean latent, with the encoder's training-time
`mask_noise = 1e-3` regulariser switched off and the same non-autotuned
attention kernel pinned that the model extractor uses.  The AE is built by the
same `load_autoencoder` call on the same checkpoint; with --model-diag (the
`stableaudio3_medium_diag.json` written by `extract_stable_audio3.py`) the
checkpoint hash, latent geometry, attention pin and the number of blocks whose
`mask_noise` is disabled are asserted against the model run.  The latent is
transposed to [T, 256] so the stored array is [1, T, H], the layout
`bcs/score.py` scores as a single capture.

The DiT is not built here: the front-end is the autoencoder alone.

Usage: BCS_STIM_ROOT=<stimulus root> extract_fe_samel.py --out <dir> [--paradigms cue1,cue2,...]
                           [--model-diag <dir>/stableaudio3_medium_diag.json]
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_stable_audio3 import (HF_ID, HF_REV, SEED, SR,                    # noqa: E402
                         pin_deterministic_attention, resolve_snapshot)
from frontend_common import ALL_PARADIGMS, run_frontend, sha256_file       # noqa: E402
from grid_common import load_wav_mono                                  # noqa: E402

KEY = "fe_samel"


def build_ae(device, dtype):
    """The AE half of `extract_stable_audio3.build_model`, verbatim."""
    from stable_audio_3.loading_utils import load_autoencoder

    attn_pin = pin_deterministic_attention()
    root = resolve_snapshot()
    cfg_path = root / "model_config.json"
    ckpt_path = root / "model.safetensors"
    config = json.loads(cfg_path.read_text())

    ae = load_autoencoder(str(cfg_path), str(ckpt_path), device="cpu")
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    n_off = 0
    for m in ae.encoder.modules():
        if getattr(m, "mask_noise", 0):
            m.mask_noise = 0.0
            n_off += 1

    meta = {"checkpoint_file": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "latent_dim": int(config["model"]["io_channels"]),
            "downsampling_ratio": int(
                config["model"]["pretransform"]["config"]["downsampling_ratio"]),
            "encoder_mask_noise_disabled_blocks": n_off,
            "attention_pin": attn_pin}
    return ae, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS))
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    ap.add_argument("--model-diag", type=Path, default=None,
                    help="stableaudio3_medium_diag.json of the model run; "
                         "if given, the encoder settings must match it")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    dtype = torch.bfloat16                       # same pin as the model run

    ae, mm = build_ae(device, dtype)

    # Cross-check against the model run so the front-end contrast
    # cannot be computed off a differently-configured encoder.
    if args.model_diag is None:
        print(f"[{KEY}] no --model-diag: encoder settings not cross-checked "
              "against the model run", flush=True)
    else:
        md = json.loads(args.model_diag.read_text())
        for k in ("checkpoint_sha256", "latent_dim", "downsampling_ratio",
                  "encoder_mask_noise_disabled_blocks", "attention_pin"):
            assert md.get(k) == mm.get(k), (k, md.get(k), mm.get(k))
        mm["matches_model_run_diag"] = True

    @torch.no_grad()
    def encode(path):
        wav = load_wav_mono(path, SR)                        # [1, T] fp32 cpu
        x = wav.repeat(2, 1).unsqueeze(0).to(device=device, dtype=dtype)
        z = ae.encode_audio(x, chunked=False)                # [1, 256, T_lat]
        return z[0].transpose(0, 1).to(torch.float32).cpu().numpy()   # [T, 256]

    diag = {
        "model_name": KEY,
        "full_name": "Stable Audio 3 SAME-L autoencoder latent "
                     "(deterministic mean, mask_noise off)",
        "shared_by": ["stableaudio3_medium"],
        "hf_id": HF_ID, "hf_revision": HF_REV,
        "sample_rate": SR, "dtype_compute": "bfloat16",
        "captured": "ae.encode_audio(stereo(mono), chunked=False) -> [T, 256]",
        "role": "learned front-end of stableaudio3_medium",
        **mm,
    }
    run_frontend(KEY, encode, args.paradigms.split(","), Path(args.out), diag,
                 hidden_dim=mm["latent_dim"], limit=args.limit)
    print(f"[{KEY}] VRAM peak {torch.cuda.max_memory_reserved() / 1e9:.2f} GB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
