#!/usr/bin/env python3
"""Front-end: PupuJEPA repo log-mel spectrogram (`fe_pupujepa_logmel`).

PupuM2D-L's own log-spectrogram front-end; it is the front-end reference of
PupuM2D-L.

What is extracted: `extract_pupujepa.extract_mel`, the front-end of the
PupuJEPA code (`exp_config_pupujepa_large.json` /
`train_pupujepa.py::extract_mel_features`): 24 kHz, n_fft 1024, hop 240, win
1024, 128 mels, fmin 0 / fmax 12000, reflect pad (n_fft-hop)/2, center=False,
log + fixed (mean, std) normalisation, `flip_ft` -> [T_mel, 128].

No patch padding and no patch-time pooling are applied: those belong to the
model's learned `patch_embed`, not to the front-end.  The stored grid is
therefore the raw mel grid at 100 Hz (24000 / 240), while the model is read at
25 Hz (mel frames / PATCH_T = 4).

Device: pure DSP, torch STFT + a librosa mel filterbank, no learned weights.
It runs on CPU by default; `--device cuda`
reproduces the tensor the model extractor computes on-GPU, and the two are
compared on 48 clips (the first 24 of the frequency and of the temporal cue
under BCS_STIM_ROOT) by `--device_check`.

Usage: BCS_STIM_ROOT=<stimulus root> extract_fe_pupujepa_logspec.py --out <dir> [--paradigms ...] [--device cpu]
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
sys.path.insert(0, os.environ.get("PUPUJEPA_SRC", ""))  # clone of the PupuJEPA code

from extract_pupujepa import (FMAX, FMIN, HF_ID, HF_REV, HOP, N_FFT,  # noqa: E402
                              N_MELS, NORM_MEAN, NORM_STD, PATCH_F,
                              PATCH_T, SR, SUBDIR, WIN, extract_mel)
from frontend_common import ALL_PARADIGMS, iter_stimuli, run_frontend            # noqa: E402
from grid_common import load_wav_mono                                   # noqa: E402

KEY = "fe_pupujepa_logmel"


def make_encode(device):
    def encode(path):
        wav = load_wav_mono(path, SR).to(device)          # [1, T] fp32
        mel = extract_mel(wav)                            # [1, 1, T_mel, 128]
        return mel[0, 0].to(torch.float32).cpu().numpy()  # [T_mel, 128]
    return encode


def device_check() -> dict:
    """CPU vs CUDA agreement of the front-end on 48 clips (24 per cue)."""
    if not torch.cuda.is_available():
        return {"ran": False, "reason": "cuda unavailable"}
    e_cpu, e_gpu = make_encode(torch.device("cpu")), make_encode(torch.device("cuda"))
    worst, n = 0.0, 0
    for pdm in ("freq_proximity", "temp_proximity"):
        for _stem, p in iter_stimuli(pdm)[:24]:
            a, b = e_cpu(p), e_gpu(p)
            assert a.shape == b.shape, (p.name, a.shape, b.shape)
            worst = max(worst, float(np.max(np.abs(a - b))))
            n += 1
    return {"ran": True, "n_clips": n, "max_abs_delta_cpu_vs_cuda": worst,
            "note": "normalised log-mel units (unit std by construction)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    ap.add_argument("--device_check", action="store_true",
                    help="only run the CPU-vs-CUDA agreement diagnostic")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    outdir = Path(args.out)

    if args.device_check:
        rec = device_check()
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"{KEY}_device_check.json"
        p.write_text(json.dumps(rec, indent=2))
        print(f"[{KEY}] device check: {rec}", flush=True)
        return 0

    device = torch.device(args.device)
    diag = {
        "model_name": KEY,
        "full_name": "PupuJEPA repo log-spectrogram front-end",
        "shared_by": ["pupujepa_large"],
        "hf_id": HF_ID, "hf_revision": HF_REV, "subdir": SUBDIR,
        "sample_rate": SR, "n_fft": N_FFT, "hop": HOP, "win": WIN,
        "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "reflect_pad_samples": int((N_FFT - HOP) / 2), "center": False,
        "captured": "extract_pupujepa.extract_mel (repo-exact) -> [T_mel, 128]",
        "patch_pooling_applied": False,
        "model_patch_grid": {"patch_t": PATCH_T, "patch_f": PATCH_F},
        "frame_rate_note": (
            "raw mel grid = 100 Hz (24000/240); the model is read at 25 Hz "
            "(mel frames / patch_t=4). The front-end is the repo "
            "log-spectrogram itself, so no pooling is applied."),
        "device": args.device,
        "role": "log-spectrogram front-end of pupujepa_large",
    }
    run_frontend(KEY, make_encode(device), args.paradigms.split(","), outdir,
                 diag, hidden_dim=N_MELS, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
