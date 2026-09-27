#!/usr/bin/env python3
"""Extract ACE-Step 1.5 XL (Turbo, 4B) DiT hidden states for the stimulus cues.

Same probing paradigm as extract_acestep15.py (VAE.encode.latent_dist.mean →
DiT forward at small t with null text cond → hook decoder.layers[i]).

XL differs from extract_acestep15.py:
  - hf_id = "ACE-Step/acestep-v15-xl-turbo" (DiT checkpoint at repo root, no dit_subfolder)
  - VAE still from "ACE-Step/Ace-Step1.5" subfolder="vae"
  - hidden_size=2560, num_hidden_layers=32
  - `trust_remote_code: true` and `encoder_hidden_size` may differ from `hidden_size`

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
    get_stim_paradigms,
    load_config,
    load_wav_mono,
    require_gpu,
    set_deterministic_seeds,
    snapshot_download_pinned,
)

MODEL_KEY = "acestep15_xl"
OUT_PREFIX = "acestep15_xl"

def _load_vae(vae_dir: Path, device: torch.device, dtype):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
    vae = vae.to(device=device, dtype=dtype).eval()
    return vae

def _load_dit_model(dit_dir: Path, device: torch.device, dtype, trust_remote: bool):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        str(dit_dir),
        trust_remote_code=trust_remote,
        low_cpu_mem_usage=False,  # disable meta-tensor init
    )
    model = model.to(device=device, dtype=dtype).eval()
    if not hasattr(model, "decoder"):
        raise RuntimeError(
            f"[{MODEL_KEY}] Loaded model has no `.decoder`; upstream API changed."
        )
    if not hasattr(model, "null_condition_emb"):
        raise RuntimeError(
            f"[{MODEL_KEY}] Loaded model has no `.null_condition_emb`; upstream API changed."
        )
    return model

def _encode_audio_to_latent(audio_np, vae, device, dtype, vae_mode):
    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Only vae_mode='mean' is supported; got '{vae_mode}'."
    )
    a = torch.from_numpy(audio_np).to(device=device, dtype=dtype)
    a = a.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # [1, 2, T] stereo
    with torch.no_grad():
        latent = vae.encode(a).latent_dist.mean  # posterior mean, not a sample
    return latent.transpose(1, 2).contiguous()  # [1, T_lat, C_lat]

def _register_layer_hooks(decoder, layers_sample):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    n_layers = len(decoder.layers)
    valid = []
    for idx in layers_sample:
        if 0 <= idx < n_layers:
            valid.append(idx)
        else:
            warnings.warn(
                f"[{MODEL_KEY}] layers_sample idx {idx} OOR (n={n_layers}); dropping.",
                RuntimeWarning,
            )

    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach()
        return hook

    for idx in valid:
        handles.append(decoder.layers[idx].register_forward_hook(mk(idx)))
    return captured, handles, valid

def _run_dit_forward(model, latent, device, dtype, t_value):
    B, T, C = latent.shape
    noise = torch.randn_like(latent)
    t_s = torch.full((B,), t_value, device=device, dtype=dtype)
    t_e = t_s.view(-1, 1, 1)
    xt = t_e * noise + (1.0 - t_e) * latent

    src = torch.zeros_like(latent)
    mask = torch.zeros_like(latent)
    ctx = torch.cat([src, mask], dim=-1)
    attn = torch.ones(B, T, device=device, dtype=torch.long)
    null_cond = model.null_condition_emb.to(device=device, dtype=dtype)
    enc_h = null_cond.expand(B, 1, null_cond.shape[-1]).contiguous()
    enc_attn = torch.ones(B, 1, device=device, dtype=torch.long)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        _ = model.decoder(
            hidden_states=xt,
            timestep=t_s,
            timestep_r=t_s,
            attention_mask=attn,
            encoder_hidden_states=enc_h,
            encoder_attention_mask=enc_attn,
            context_latents=ctx,
            use_cache=False,
            return_hidden_states=None,
        )

def extract_paradigm(model, vae, device, dtype, paradigm_dir, target_sr,
                    layers_sample, vae_mode, t_value):
    wav_paths = sorted(p for p in paradigm_dir.glob("*.wav"))
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files; skipping")
        return {}
    out: dict[str, np.ndarray] = {}
    done = 0
    for wp in wav_paths:
        audio = load_wav_mono(wp, target_sr)
        latent = _encode_audio_to_latent(audio, vae, device, dtype, vae_mode)
        captured, handles, valid = _register_layer_hooks(model.decoder, layers_sample)
        try:
            _run_dit_forward(model, latent, device, dtype, t_value)
        finally:
            for h in handles:
                h.remove()
        arrs = [captured[i].to(torch.float32).cpu().numpy().squeeze(0) for i in valid]
        out[wp.stem] = np.stack(arrs, axis=0).astype(np.float32)
        done += 1
        if done % 10 == 0 or done == len(wav_paths):
            print(f"  [{paradigm_dir.name}] processed {done}/{len(wav_paths)}")
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paradigms", nargs="*", default=None)
    ap.add_argument("--stim_root", type=Path, required=True,
                    help="stimulus root written by stimuli/generate_stimuli.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t_value", type=float, default=0.01)
    ap.add_argument("--output_tag", type=str, default="",
                    help="Optional filename tag inserted between MODEL_KEY and "
                         "paradigm: {MODEL}_{tag}_{paradigm}.npz.")
    args = ap.parse_args()

    require_gpu()
    config = load_config()
    mc = config["models"][MODEL_KEY]

    seed = int(config["random_seeds"]["stimuli_base"])
    set_deterministic_seeds(seed)

    paradigms = args.paradigms or get_stim_paradigms(config)
    target_sr = int(mc["sample_rate"])
    hf_id = mc["hf_id"]
    revision = mc["hf_revision"]
    # XL has no dit_subfolder (checkpoint lives at repo root).
    dit_subfolder = mc.get("dit_subfolder", None)
    vae_hf_id = mc["vae_hf_id"]
    vae_hf_revision = mc["vae_hf_revision"]
    vae_subfolder = mc.get("vae_subfolder", "vae")
    vae_mode = mc.get("vae_mode", "mean")
    layers_sample = list(mc["layers_sample"])
    trust_remote = bool(mc.get("trust_remote_code", True))

    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Config vae_mode must be 'mean'; got '{vae_mode}'."
    )

    print(f"[extract_{MODEL_KEY}] device=cuda hf_id={hf_id} revision={revision[:12]}")
    print(f"[extract_{MODEL_KEY}] target_sr={target_sr} layers_sample={layers_sample}")
    print(f"[extract_{MODEL_KEY}] vae_mode={vae_mode} t_value={args.t_value}")

    device = torch.device("cuda")
    dtype_str = config["models"][MODEL_KEY].get("dtype_override", "float16")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]
    print(f"[extract_{MODEL_KEY}] dtype={dtype_str}")

    dit_repo = snapshot_download_pinned(hf_id, revision)
    dit_dir = dit_repo / dit_subfolder if dit_subfolder else dit_repo
    if not dit_dir.is_dir():
        raise FileNotFoundError(f"[{MODEL_KEY}] DiT dir not found: {dit_dir}")

    vae_repo = snapshot_download_pinned(vae_hf_id, vae_hf_revision)
    vae_dir = vae_repo / vae_subfolder
    if not vae_dir.is_dir():
        raise FileNotFoundError(f"[{MODEL_KEY}] VAE dir not found: {vae_dir}")

    print(f"[extract_{MODEL_KEY}] loading VAE from {vae_dir}")
    vae = _load_vae(vae_dir, device, dtype)
    print(f"[extract_{MODEL_KEY}] loading DiT from {dit_dir}")
    model = _load_dit_model(dit_dir, device, dtype, trust_remote)
    print(f"[extract_{MODEL_KEY}] decoder.layers = {len(model.decoder.layers)}")

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY, seed=seed)

    for paradigm in paradigms:
        pdir = args.stim_root / paradigm
        if not pdir.is_dir():
            warnings.warn(f"paradigm dir not found: {pdir}", RuntimeWarning)
            continue
        name_tag = f"_{args.output_tag}" if args.output_tag else ""
        out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
        # SKIP-before-extract idempotency (do not waste GPU cycles regenerating
        # an existing NPZ). To force regeneration, delete the file beforehand.
        if out_path.exists():
            print(f"[extract_{MODEL_KEY}] SKIP {paradigm} (exists: {out_path.name})")
            continue
        print(f"[extract_{MODEL_KEY}] paradigm={paradigm}")
        sd = extract_paradigm(model, vae, device, dtype, pdir, target_sr,
                              layers_sample, vae_mode, args.t_value)
        if not sd:
            continue
        atomic_savez(out_path, _meta=run_meta, **sd)
        print(f"[extract_{MODEL_KEY}] wrote {out_path} ({len(sd)} stims)")

    print(f"[extract_{MODEL_KEY}] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
