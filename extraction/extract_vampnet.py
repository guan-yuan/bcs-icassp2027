#!/usr/bin/env python3
"""Extract VampNet-coarse hidden states for the stimulus cues.

VampNet (Flores-Garcia et al., ISMIR 2023) is a non-autoregressive masked
parallel decoder on tokens of the DAC-44.1 kHz codec VampNet ships (lac
package). Only the coarse transformer is read (20 layers, embed_dim=1280,
n_heads=20); the coarse-to-fine refinement net is not.

Pipeline per stimulus:
  1. Load wav (mono) -> resample to 44.1 kHz
  2. Interface.encode(AudioSignal) -> codes [B, n_codebooks, T_codes]
  3. Keep top-N_COARSE codebooks -> coarse.embedding.from_codes(codes, codec)
     -> latents [B, n_coarse*latent_dim, T]
  4. coarse(latents, return_activations=True) -> list of per-layer
     activations [B, T, d_model] (vampnet/modules/transformer.py)
  5. Stack sampled layers -> [n_layers, T, 1280] npz entry

The beat tracker is not loaded (wavebeat_ckpt=None); it plays no part in this
forward pass. Code: vampnet 0.0.1 and lac 0.0.1 (requirements.txt); weights:
hugggof/vampnet at the revision in config/checkpoints.yaml. Runs on a GPU with
the seeds of the config; layers_sample is clamped to the transformer depth.

Output: {out}/vampnet_{paradigm}.npz
  {stim_name: array[n_layers, T, 1280]}  plus  _meta: run_meta
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
    clamp_layers,
    get_stim_paradigms,
    iter_wavs,
    load_config,
    load_wav_mono,
    require_gpu,
    set_deterministic_seeds,
)

MODEL_KEY = "vampnet"
OUT_PREFIX = "vampnet"

def build_iface(device: torch.device, revision: str):
    """Load VampNet through its Interface, without the beat tracker.

    The three checkpoints are fetched with hf_hub_download at the pinned
    revision and passed to the `Interface(...)` constructor.
    """
    from huggingface_hub import hf_hub_download
    from vampnet.interface import Interface

    # vampnet.download_codec()/download_default() resolve revision "main" into a
    # local_dir, which cannot be served from the HF cache under HF_HUB_OFFLINE=1.
    # Fetch the pinned revision instead (served from the local cache when present).
    print(f"  resolving VampNet checkpoints at revision {revision}", flush=True)
    codec_path = hf_hub_download("hugggof/vampnet", "codec.pth", revision=revision)
    coarse_path = hf_hub_download("hugggof/vampnet", "coarse.pth", revision=revision)
    c2f_path = hf_hub_download("hugggof/vampnet", "c2f.pth", revision=revision)
    print(f"    codec:  {codec_path}")
    print(f"    coarse: {coarse_path}")

    iface = Interface(
        coarse_ckpt=coarse_path,
        coarse2fine_ckpt=c2f_path,
        codec_ckpt=codec_path,
        wavebeat_ckpt=None,      # the beat tracker is not used
        device=str(device),
        compile=False,
    )
    iface.coarse.eval()
    iface.codec.eval()
    return iface

def _n_coarse_layers(iface) -> int:
    layers = getattr(iface.coarse.transformer, "layers", None)
    if layers is None:
        raise RuntimeError(
            "Expected iface.coarse.transformer.layers; got "
            f"{type(iface.coarse.transformer).__name__}"
        )
    return len(layers)

def _encode_to_latents(iface, audio: torch.Tensor, target_sr: int, n_coarse: int):
    """Encode audio -> codec int codes -> float latents.

      Interface.encode(AudioSignal) -> [B, K, T] int codes
      VampNet.embedding.from_codes(codes, codec) -> [B, K*latent_dim, T] floats
    """
    from audiotools import AudioSignal

    # audio: [1, T] mono -> AudioSignal expects [B, 1, T]
    sig = AudioSignal(audio.unsqueeze(0), sample_rate=target_sr)
    codes = iface.encode(sig)                       # [B, K_total, T_codes]
    coarse_codes = codes[:, :n_coarse, :]
    latents = iface.coarse.embedding.from_codes(coarse_codes, iface.codec)
    return latents

def _forward_coarse_with_activations(
    iface, latents: torch.Tensor, layers_sample: list[int]
) -> tuple[dict[int, torch.Tensor], int]:
    """Per-layer activations from the coarse transformer (return_activations=True).

    Returns ({layer_idx: [T, D] cpu fp32}, n_layers).
    """
    with torch.no_grad():
        _, activations = iface.coarse(latents, return_activations=True)
    n_layers = len(activations)
    captured: dict[int, torch.Tensor] = {}
    for li in layers_sample:
        if li < n_layers:
            act = activations[li]
            # [B=1, T, D] -> [T, D]
            captured[li] = act.detach().squeeze(0).to(torch.float32).cpu()
    return captured, n_layers

def extract_paradigm(
    iface,
    device: torch.device,
    paradigm_dir: Path,
    target_sr: int,
    layers_sample: list[int],
    n_coarse: int,
) -> dict[str, np.ndarray]:
    wav_paths = iter_wavs(paradigm_dir)
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files; skip")
        return {}

    clamped: list[int] | None = None
    out: dict[str, np.ndarray] = {}
    for i, wav_path in enumerate(wav_paths):
        audio = load_wav_mono(wav_path, target_sr, as_tensor=True).to(device)
        with torch.no_grad():
            latents = _encode_to_latents(iface, audio, target_sr, n_coarse)
            captured, n_layers = _forward_coarse_with_activations(
                iface, latents, layers_sample
            )
        if clamped is None:
            clamped = clamp_layers(layers_sample, n_layers, MODEL_KEY)
        stacked = np.stack(
            [captured[li].numpy() for li in clamped if li in captured], axis=0
        )  # [n_layers, T, D]
        out[wav_path.stem] = stacked.astype(np.float32)

        if (i + 1) % 10 == 0 or (i + 1) == len(wav_paths):
            print(f"  [{paradigm_dir.name}] processed {i + 1}/{len(wav_paths)}")
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paradigms", nargs="*", default=None,
                    help="Subset; default = every paradigm in the config")
    ap.add_argument("--stim_root", type=Path, required=True,
                    help="stimulus root written by stimuli/generate_stimuli.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--output_tag", type=str, default="",
                    help="Optional filename tag inserted between MODEL_KEY and "
                         "paradigm: {MODEL}_{tag}_{paradigm}.npz.")
    args = ap.parse_args()

    require_gpu()
    config = load_config()
    model_cfg = config["models"][MODEL_KEY]

    seed = int(config["random_seeds"]["stimuli_base"])
    set_deterministic_seeds(seed)

    paradigms = args.paradigms or get_stim_paradigms(config)
    target_sr = int(model_cfg["sample_rate"])
    hf_id = model_cfg["hf_id"]
    revision = model_cfg["hf_revision"]
    n_coarse = int(model_cfg.get("n_coarse_codebooks", 4))
    layers_sample = list(model_cfg["layers_sample"])

    print(f"[extract_vampnet] device=cuda hf_id={hf_id} revision={revision}")
    print(f"[extract_vampnet] target_sr={target_sr} n_coarse={n_coarse}")
    print(f"[extract_vampnet] layers_sample={layers_sample} paradigms={paradigms}")

    device = torch.device("cuda")
    iface = build_iface(device, revision)
    n_layers = _n_coarse_layers(iface)
    print(f"[extract_vampnet] coarse transformer has {n_layers} layers")

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY, seed=seed)

    try:
        for paradigm in paradigms:
            paradigm_dir = args.stim_root / paradigm
            if not paradigm_dir.is_dir():
                warnings.warn(
                    f"paradigm dir not found: {paradigm_dir}", RuntimeWarning
                )
                continue
            print(f"[extract_vampnet] paradigm={paradigm}")
            stim_dict = extract_paradigm(
                iface=iface,
                device=device,
                paradigm_dir=paradigm_dir,
                target_sr=target_sr,
                layers_sample=layers_sample,
                n_coarse=n_coarse,
            )
            if not stim_dict:
                continue
            name_tag = f"_{args.output_tag}" if args.output_tag else ""
            out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
            if out_path.exists():
                print(f"[extract_vampnet] SKIP {paradigm} (exists: {out_path.name})")
                continue
            atomic_savez(out_path, _meta=run_meta, **stim_dict)
            print(f"[extract_vampnet] wrote {out_path} ({len(stim_dict)} stims)")
    finally:
        del iface
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[extract_vampnet] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
