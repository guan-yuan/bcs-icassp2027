#!/usr/bin/env python3
"""Extractor: PupuM2D-L (spellbrush/PupuJEPA).

Forward:
  log-mel front-end (repo-exact) -> patch_embed -> student PupuJEPAEncoder,
  random masking fully disabled (deterministic wrapper) -> hook encoder blocks.
  2-D freq x time patch grid -> mean over the frequency-patch axis.

Usage: BCS_STIM_ROOT=<stimulus root> extract_pupujepa.py --out <dir> [--paradigms cue1,cue2]
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

from grid_common import (OUT, DEFAULT_PARADIGMS, Progress, VramPeak,  # noqa
                       ALL_PARADIGMS,
                       flatten_time_major, frame_diagnostics, iter_paradigm,
                       layer_ids, load_wav_mono, save_npz, sha256_file)

MODEL_KEY = "pupujepa_large"
HF_ID = "spellbrush/PupuJEPA"
HF_REV = "2ba230e41440c5b450a8dc8ad5d4a3cc9930f01d"
SUBDIR = "pupujepaV2_25hz_large"

# --- preprocessing constants, as in the PupuJEPA code
#     (exp_config_pupujepa_large.json + train_pupujepa.py::extract_mel_features).
SR = 24000
N_FFT = 1024
HOP = 240
WIN = 1024
N_MELS = 128
FMIN, FMAX = 0, 12000
NORM_MEAN = -4.089994845986366
NORM_STD = 2.0242277159094813
PATCH_T, PATCH_F = 4, 16

_mel_basis = {}
_hann = {}


def extract_mel(y: torch.Tensor) -> torch.Tensor:
    """Repo-exact log-mel. y: [B, T] -> [B, 1, T_mel, 128] (flip_ft=True)."""
    from librosa.filters import mel as librosa_mel_fn
    dev = str(y.device)
    if dev not in _mel_basis:
        m = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        _mel_basis[dev] = torch.from_numpy(m).float().to(y.device)
        _hann[dev] = torch.hann_window(WIN).to(y.device)

    pad = int((N_FFT - HOP) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, N_FFT, hop_length=HOP, win_length=WIN, window=_hann[dev],
                      center=False, pad_mode="reflect", normalized=False,
                      onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[dev], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    spec = (spec - NORM_MEAN) / (NORM_STD + 1e-8)
    spec = spec.transpose(-2, -1)            # flip_ft -> [B, T_mel, 128]
    return spec.unsqueeze(1)                 # [B, 1, T_mel, 128]


def build_model(device):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from util import load_config
    from model.pupujepa import PupuJEPA

    root = Path(snapshot_download(repo_id=HF_ID, revision=HF_REV,
                                  allow_patterns=[f"{SUBDIR}/**"],
                                  local_files_only=True))
    cfg = load_config(str(Path(os.environ.get("PUPUJEPA_SRC", ""))
                          / "exp_config_pupujepa_large.json"))
    cfg.train.grad_checkpointing_step = None
    model = PupuJEPA(cfg)

    ckpts = sorted((root / SUBDIR / "checkpoint").glob("*/model.safetensors"))
    assert len(ckpts) == 1, f"expected 1 checkpoint, found {ckpts}"
    sd = load_file(str(ckpts[0]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("teacher.")]
    assert not missing, f"missing keys: {missing[:10]}"
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, str(ckpts[0]), root


@torch.no_grad()
def run_clip(model, wav: torch.Tensor, device, ids: list[int]):
    """Deterministic forward: no masking, all patches to the student encoder."""
    mel = extract_mel(wav.to(device))                      # [1, 1, T_mel, 128]
    t_mel = mel.shape[2]
    pad_t = (-t_mel) % PATCH_T
    if pad_t:                                              # keep patch grid exact
        mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_t), mode="replicate")
    grid = (mel.shape[2] // PATCH_T, mel.shape[3] // PATCH_F)

    x = model.patch_embed(mel)                             # [1, Tp*Fp, D]
    # training path: rope_full[masks] -> [B, N, D_rope] then .unsqueeze(1).
    # With no masking the full grid is used, so add the batch axis explicitly.
    rope = model.rope_encoder.get_embed(grid).unsqueeze(0).unsqueeze(1)

    caught: dict[int, torch.Tensor] = {}
    handles = []
    for i in ids:
        handles.append(model.student.blocks[i].register_forward_hook(
            lambda m, inp, out, i=i: caught.__setitem__(i, out.detach())))
    try:
        model.student(x, rope=rope)
    finally:
        for h in handles:
            h.remove()

    stack = []
    for i in ids:
        h = caught[i].to(torch.float32)                    # [1, Tp*Fp, D]
        h = flatten_time_major(h, grid[0], grid[1])        # -> [1,Tp,D]
        stack.append(h[0])
    return (torch.stack(stack).cpu().numpy(), grid, t_mel, pad_t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS[:4]),
                    help="comma-separated cues (default: the four cues of the paper)")
    args = ap.parse_args()
    if not os.environ.get("BCS_STIM_ROOT"):
        ap.error("set BCS_STIM_ROOT to the stimulus root written by stimuli/generate_stimuli.py")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    model, cfg, ckpt, root = build_model(device)
    n_blocks = len(model.student.blocks)
    hidden = int(cfg.model.embed_dim)
    ids = layer_ids(n_blocks)
    print(f"[{MODEL_KEY}] L={n_blocks} H={hidden} layer_ids={ids}", flush=True)

    outdir = Path(args.out)
    diag = {"model_key": MODEL_KEY, "hf_id": HF_ID, "hf_revision": HF_REV,
            "checkpoint_file": ckpt, "checkpoint_sha256": sha256_file(ckpt),
            "layers_total": n_blocks, "hidden_dim": hidden, "layer_ids": ids,
            "dtype": "float32", "captured_module": "student.blocks[i]",
            "conditioning": "none (mask_ratio=0, all patches, deterministic)",
            "sample_rate": SR, "front_end": "repo log-mel, log-spectrogram front-end",
            "paradigms": {}}

    with VramPeak() as vp:
        for pdm in args.paradigms.split(","):
            reps, nframes, shapes = {}, {}, None
            items = iter_paradigm(pdm)
            prog = Progress(f"{MODEL_KEY}/{pdm}", len(items))
            for stem, path in items:
                wav = load_wav_mono(path, SR)
                arr, grid, t_mel, pad_t = run_clip(model, wav, device, ids)
                reps[stem] = arr
                nframes[stem] = arr.shape[1]
                prog.tick()
                if shapes is None:
                    shapes = {"mel_frames": int(t_mel), "mel_pad": int(pad_t),
                              "patch_grid_time": int(grid[0]),
                              "patch_grid_freq": int(grid[1]),
                              "shape_before_flatten": [1, grid[0] * grid[1], hidden],
                              "shape_after_flatten": list(arr.shape)}
            secs = prog.done()
            npz = outdir / f"{MODEL_KEY}_{pdm}.npz"
            save_npz(npz, reps, diag)
            fr = float(np.mean([n for n in nframes.values()]))
            from grid_common import T_SEQ
            diag["paradigms"][pdm] = {
                "n_stimuli": len(reps), "shapes": shapes,
                "measured_frame_rate_hz": fr / T_SEQ[pdm],
                "npz": str(npz), "npz_bytes": npz.stat().st_size,
                "extract_seconds": round(secs, 1),
                "seconds_per_clip": round(secs / max(len(reps), 1), 4),
                **frame_diagnostics(pdm, nframes, reps)}
            del reps
    diag["vram_peak_alloc_gb"] = round(vp.alloc_gb, 3)
    diag["vram_peak_reserved_gb"] = round(vp.reserved_gb, 3)

    (outdir / f"{MODEL_KEY}_diag.json").write_text(json.dumps(diag, indent=2))
    print(f"[{MODEL_KEY}] done; VRAM peak {vp.reserved_gb:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
