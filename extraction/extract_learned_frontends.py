#!/usr/bin/env python3
"""Learned front-end references (tokeniser / VAE outputs).

A checkpoint does not read the waveform directly but the output of its own
tokeniser or VAE; a randomly initialised decoder on a pretrained tokeniser
still inherits whatever cue geometry that tokeniser output carries. The fixed
DSP front-ends (log-mel / CQT / gammatone) are not what the models read, so
this script extracts the learned front-end of every model family, scored on
the same stimuli with the same operator.

  fe_encodec32_pre    EnCodec-32 kHz encoder output before the RVQ
                      (continuous latent, 128-d, 50 Hz)
  fe_encodec32_rvq    EnCodec-32 kHz de-quantised RVQ embedding at 4 codebooks
                      / 2.2 kbps (the MusicGen and MAGNeT tokenizer), computed
                      in float32
  fe_lac_pre          encoder output before the RVQ of the DAC-44.1 kHz codec
                      VampNet ships (lac package) (1024-d)
  fe_lac_rvq          coarse-4 code latents of that codec (32-d), the tensor
                      VampNet's coarse transformer receives
  fe_diffrhythm_vae   DiffRhythm v1.2 VAE posterior mean (64-d)
  fe_acestep_vae      ACE-Step 1.5 Oobleck VAE posterior mean (64-d)

Every front-end is emitted in the layout `bcs/score.py` consumes for a
single-"layer" front-end, i.e. `[1, T, H]` per stimulus with `layers_sample =
[0]`, so the single-capture path scores them exactly as it scores the fixed DSP
baselines. No projection, no whitening, no pooling: the raw front-end output is
scored.

Output: {out}/{model_key}_{paradigm}.npz  ({stim_key: [1, T, H]} + `_meta`)

CLI
  python3 extract_learned_frontends.py --frontend fe_encodec32_rvq \
      --paradigms freq_proximity temp_proximity harmonicity onset_sync \
      --stim_root DIR --out DIR
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from extract_common import (
    atomic_savez,
    build_metadata,
    load_config,
    load_metadata,
    load_wav_mono,
    require_gpu,
    set_deterministic_seeds,
    snapshot_download_pinned,
    stim_key,
)

FRONTENDS = ("fe_encodec32_pre", "fe_encodec32_rvq", "fe_lac_pre", "fe_lac_rvq",
             "fe_diffrhythm_vae", "fe_acestep_vae")

CORE_CUES = ["freq_proximity", "temp_proximity", "harmonicity", "onset_sync"]


# ─────────────────────────────────────────────────────────────────────────────
# Front-end builders.  Each returns `encode(wav_1T_on_device) -> [T, H] fp32`.
# ─────────────────────────────────────────────────────────────────────────────

def _build_encodec32(model_cfg: dict, device, which: str):
    from transformers import EncodecModel

    m = EncodecModel.from_pretrained(model_cfg["hf_id"],
                                     revision=model_cfg["hf_revision"])
    m = m.to(device).eval()
    # Top bandwidth (4 codebooks, 2.2 kbps), target_bandwidths[-1], as in
    # extract_musicgen.py.
    bandwidth = m.config.target_bandwidths[-1]
    print(f"  EnCodec-32k bandwidth={bandwidth} kbps  stage={which}")

    def encode(wav: torch.Tensor) -> np.ndarray:      # wav: [1, T] on device
        x = wav.unsqueeze(0)                          # [1, 1, T]
        with torch.no_grad():
            if which == "pre":
                h = m.encoder(x)                      # [1, 128, T_f]
            else:
                enc = m.encode(x, bandwidth=bandwidth)
                codes = enc.audio_codes
                if codes.dim() == 4:                  # [n_chunks, B, K, T_f]
                    codes = codes.squeeze(0)          # [B, K, T_f]
                # quantizer.decode expects [K, B, T_f]
                h = m.quantizer.decode(codes.transpose(0, 1))   # [1, 128, T_f]
        return h.squeeze(0).transpose(0, 1).to(torch.float32).cpu().numpy()

    return encode


def _build_lac(model_cfg: dict, device, which: str):
    from huggingface_hub import hf_hub_download
    from vampnet.interface import Interface
    from audiotools import AudioSignal

    rev = model_cfg["hf_revision"]
    iface = Interface(
        coarse_ckpt=hf_hub_download("hugggof/vampnet", "coarse.pth", revision=rev),
        coarse2fine_ckpt=hf_hub_download("hugggof/vampnet", "c2f.pth", revision=rev),
        codec_ckpt=hf_hub_download("hugggof/vampnet", "codec.pth", revision=rev),
        wavebeat_ckpt=None, device=str(device), compile=False,
    )
    iface.coarse.eval()
    iface.codec.eval()
    n_coarse = 4
    sr = int(model_cfg["sample_rate"])

    def encode(wav: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            sig = AudioSignal(wav.unsqueeze(0), sample_rate=sr)
            if which == "rvq":
                codes = iface.encode(sig)                       # [1, K, T_f]
                h = iface.coarse.embedding.from_codes(
                    codes[:, :n_coarse, :].clone(), iface.codec)  # [1, 32, T_f]
            else:
                pp = iface.codec.preprocess(sig.samples.to(device), sr)
                x = pp[0] if isinstance(pp, (tuple, list)) else pp
                h = iface.codec.encoder(x)                      # [1, 1024, T_f]
        return h.squeeze(0).transpose(0, 1).to(torch.float32).cpu().numpy()

    return encode


def _build_diffrhythm_vae(model_cfg: dict, device):
    vae_dir = snapshot_download_pinned(model_cfg["hf_id"], model_cfg["hf_revision"])
    vae = torch.jit.load(str(vae_dir / "vae_model.pt"), map_location="cpu")
    vae = vae.to(device).eval()

    def encode(wav: torch.Tensor) -> np.ndarray:
        # DiffRhythm's VAE is stereo; `extract_diffrhythm.py` duplicates mono.
        a = wav.squeeze(0).unsqueeze(0).repeat(2, 1).unsqueeze(0)   # [1, 2, T]
        with torch.no_grad():
            latent = vae.encode_export(a.float())                   # [1, 128, T_lat]
        mean, _scale = latent.chunk(2, dim=1)                       # [1, 64, T_lat]
        return mean.squeeze(0).transpose(0, 1).to(torch.float32).cpu().numpy()

    return encode


def _build_acestep_vae(model_cfg: dict, device):
    from diffusers.models import AutoencoderOobleck

    root = snapshot_download_pinned(model_cfg["hf_id"], model_cfg["hf_revision"])
    vae = AutoencoderOobleck.from_pretrained(
        str(root / model_cfg.get("vae_subfolder", "vae")))
    vae = vae.to(device).eval()

    def encode(wav: torch.Tensor) -> np.ndarray:
        a = wav.squeeze(0).unsqueeze(0).repeat(2, 1).unsqueeze(0)   # [1, 2, T]
        with torch.no_grad():
            latent = vae.encode(a.float()).latent_dist.mean         # [1, 64, T_lat]
        return latent.squeeze(0).transpose(0, 1).to(torch.float32).cpu().numpy()

    return encode


def build_frontend(key: str, model_cfg: dict, device):
    if key == "fe_encodec32_pre":
        return _build_encodec32(model_cfg, device, "pre")
    if key == "fe_encodec32_rvq":
        return _build_encodec32(model_cfg, device, "rvq")
    if key == "fe_lac_pre":
        return _build_lac(model_cfg, device, "pre")
    if key == "fe_lac_rvq":
        return _build_lac(model_cfg, device, "rvq")
    if key == "fe_diffrhythm_vae":
        return _build_diffrhythm_vae(model_cfg, device)
    if key == "fe_acestep_vae":
        return _build_acestep_vae(model_cfg, device)
    raise ValueError(key)


# ─────────────────────────────────────────────────────────────────────────────

def extract_paradigm(encode, device, paradigm_dir: Path, target_sr: int,
                     hidden_dim: int, stim_root: Path,
                     limit: int = 0) -> dict[str, np.ndarray]:
    paradigm = paradigm_dir.name
    metadata = load_metadata(stim_root, paradigm)
    if limit:
        metadata = metadata[:limit]
    if not metadata:
        print(f"  [{paradigm}] empty metadata; skipping")
        return {}

    out: dict[str, np.ndarray] = {}
    for i, item in enumerate(metadata):
        wav = load_wav_mono(paradigm_dir / item["file"], target_sr,
                            as_tensor=True).to(device)          # [1, T]
        h = encode(wav)                                         # [T_f, H]
        assert h.ndim == 2, h.shape
        assert h.shape[-1] == hidden_dim, (
            f"hidden_dim mismatch: got {h.shape[-1]}, config says {hidden_dim}")
        out[stim_key(item)] = h[None, :, :].astype(np.float32)   # [1, T_f, H]
        if (i + 1) % 250 == 0 or i + 1 == len(metadata):
            print(f"  [{paradigm}] {i + 1}/{len(metadata)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frontend", required=True, choices=FRONTENDS)
    ap.add_argument("--paradigms", nargs="*", default=CORE_CUES)
    ap.add_argument("--stim_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="First N stimuli per cue only.")
    args = ap.parse_args()

    require_gpu()
    config = load_config()
    key = args.frontend
    model_cfg = config["models"][key]
    set_deterministic_seeds(int(config["random_seeds"]["stimuli_base"]))

    target_sr = int(model_cfg["sample_rate"])
    hidden_dim = int(model_cfg["hidden_dim"])
    device = torch.device("cuda")

    print(f"[{key}] {model_cfg.get('full_name', key)}")
    print(f"[{key}] hf_id={model_cfg['hf_id']} rev={model_cfg['hf_revision'][:12]} "
          f"sr={target_sr} H={hidden_dim}")

    encode = build_frontend(key, model_cfg, device)

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, key)
    run_meta["role"] = "learned_frontend_reference"
    run_meta["note"] = (
        "the learned tokeniser/VAE stage that sits between the waveform and "
        "the checkpoint, scored with the same single-capture operator as the "
        "fixed DSP front-ends")

    for paradigm in args.paradigms:
        pdir = args.stim_root / paradigm
        if not pdir.is_dir():
            warnings.warn(f"paradigm dir not found: {pdir}", RuntimeWarning)
            continue
        out_path = args.out / f"{key}_{paradigm}.npz"
        if out_path.exists():
            print(f"  SKIP {paradigm} (exists)")
            continue
        stim_dict = extract_paradigm(encode, device, pdir, target_sr,
                                     hidden_dim, args.stim_root, args.limit)
        if not stim_dict:
            continue
        n_bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in stim_dict.values())
        tot = sum(int(v.size) for v in stim_dict.values())
        print(f"  [{paradigm}] non-finite entries: {n_bad}/{tot}")
        if n_bad > 0:
            print(f"  [{paradigm}] ABORT: non-finite front-end output", file=sys.stderr)
            return 3
        shp = next(iter(stim_dict.values())).shape
        atomic_savez(out_path, _meta=run_meta, **stim_dict)
        print(f"  wrote {out_path.name} ({len(stim_dict)} stims, shape {shp})")

    torch.cuda.empty_cache()
    print(f"[{key}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
