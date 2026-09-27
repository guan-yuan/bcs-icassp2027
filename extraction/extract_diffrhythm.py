#!/usr/bin/env python3
"""Extract DiffRhythm v1.2 DiT hidden states for the stimulus cues.

DiT flow-matching probe:
  1. VAE.encode_export(audio) → [B, 128, T]  (TorchScript VAE; mean+scale concat)
  2. Split to mean, scale; take the mean (deterministic, `vae_mode: mean`).
  3. Build xt = t*noise + (1-t)*mean at small t=0.01.
  4. Call DiT.transformer.forward(x=xt, cond=0, text=0, time=t, drop_audio_cond=True,
     drop_text=True, style_prompt=<infer/example/vocal.npy, the negative style prompt of the DiffRhythm code>,
     start_time=0)
  5. Hooks on transformer_blocks[layers_sample] to capture LlamaDecoderLayer outputs.

Output: {out}/diffrhythm12_{paradigm}.npz with {stim_name: array[n_layers, T_lat, H=2048]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from extract_common import (
    assert_state_dict_safe,
    atomic_savez,
    build_metadata,
    get_stim_paradigms,
    load_config,
    load_wav_mono,
    require_gpu,
    set_deterministic_seeds,
    snapshot_download_pinned,
)

MODEL_KEY = "diffrhythm12"
OUT_PREFIX = "diffrhythm12"

# DiffRhythm's CFM checkpoint wraps DiT in `self.transformer`, so all DiT weight
# keys are prefixed `transformer.`. CFM also has optional `mel_spec` / vocoder
# heads that some v1.x checkpoints ship and others don't, these do not affect
# the probe. Whitelist `mel_spec.` for both missing and unexpected; any other
# key drift is a hard fail (surfaces upstream API changes).

def _build_dit_model(repo_root: Path, device: torch.device, dtype):
    """Build CFM+DiT from official repo; load cfm_model.pt state_dict.

    repo_root must be a clone of the DiffRhythm code (DIFFRHYTHM_DIR); its
    config/ and model/ trees are imported from there (main() puts it on
    sys.path).
    """
    from model import DiT, CFM  # type: ignore

    config_path = repo_root / "config" / "diffrhythm-1b.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"[{MODEL_KEY}] missing DiT config: {config_path}. "
            f"Set DIFFRHYTHM_DIR to repo clone or ensure clone is present."
        )
    with open(config_path) as f:
        mc = json.load(f)
    max_frames = 2048  # v1.2 (the -1_2 variant; -1_2-full uses 6144)
    dit = DiT(**mc["model"], max_frames=max_frames)
    cfm = CFM(
        transformer=dit,
        num_channels=mc["model"]["mel_dim"],
        max_frames=max_frames,
    )
    cfm = cfm.to(device=device, dtype=dtype).eval()
    return cfm

def _load_dit_weights(cfm, dit_repo_dir: Path):
    ckpt_path = dit_repo_dir / "cfm_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"[{MODEL_KEY}] cfm_model.pt not found in {dit_repo_dir}"
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    # strict=False: upstream includes mel_spec / vocos heads in some checkpoints,
    # which aren't part of our DiT probe.
    load_result = cfm.load_state_dict(state, strict=False)
    assert_state_dict_safe(
        load_result,
        expected_missing_prefixes=("mel_spec.",),  # CFM may lack these if not pretrained
        expected_unexpected_prefixes=("mel_spec.",),  # ckpts sometimes carry them
        label="diffrhythm-cfm",
    )
    return cfm

def _load_vae_jit(vae_dir: Path, device: torch.device):
    """DiffRhythm VAE is distributed as TorchScript (vae_model.pt). JIT-load it."""
    vae_pt = vae_dir / "vae_model.pt"
    if not vae_pt.is_file():
        raise FileNotFoundError(f"[{MODEL_KEY}] vae_model.pt not found in {vae_dir}")
    vae = torch.jit.load(str(vae_pt), map_location="cpu").to(device)
    vae.eval()
    return vae

def _encode_audio_to_latent_mean(
    audio_np, vae, device, dtype, vae_mode, target_sr=44100,
) -> torch.Tensor:
    """Encode to VAE latent and take .mean (deterministic).

    DiffRhythm's VAE expects stereo [B, 2, T]. encode_export returns [B, 128, T]
    which is mean+scale concat along channel dim; we split and use mean only.
    Returns [B, T_lat, 64]  (transposed for DiT.forward).
    """
    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Only vae_mode='mean' is supported; got '{vae_mode}'."
    )
    a = torch.from_numpy(audio_np).to(device=device)  # [T] fp32
    a = a.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # [1, 2, T] stereo
    # VAE is a jit-compiled fp32 module typically; keep inputs fp32 then cast latents.
    with torch.no_grad():
        latent = vae.encode_export(a.float())  # [B, 128, T_lat]
    mean, _scale = latent.chunk(2, dim=1)       # each [B, 64, T_lat]
    mean = mean.transpose(1, 2).contiguous()    # [B, T_lat, 64]
    return mean.to(dtype=dtype)

def _register_layer_hooks(transformer, layers_sample):
    """Hook transformer.transformer_blocks[i] → capture their Llama layer output."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    n_layers = len(transformer.transformer_blocks)
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
        handles.append(
            transformer.transformer_blocks[idx].register_forward_hook(mk(idx))
        )
    return captured, handles, valid

def _load_style_prompt(repo_root: Path, device, dtype):
    """Load infer/example/vocal.npy, which DiffRhythm's inference code uses as its negative style prompt."""
    p = repo_root / "infer" / "example" / "vocal.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"[{MODEL_KEY}] missing style prompt {p}; "
            f"DIFFRHYTHM_DIR must be a full git clone of the DiffRhythm code "
            f"(pip install git+URL drops sibling files)."
        )
    arr = np.load(str(p))
    style = torch.from_numpy(arr).to(device=device, dtype=dtype)  # [1, 512]
    return style

def _run_dit_forward(cfm, mean_latent, device, dtype, t_value, style_prompt_vec):
    """Run DiT.transformer.forward at small t: text and reference audio dropped,
    style prompt = infer/example/vocal.npy of the DiffRhythm code."""
    B, T, C = mean_latent.shape
    noise = torch.randn_like(mean_latent)
    t_s = torch.full((B,), t_value, device=device, dtype=dtype)
    xt = t_s.view(-1, 1, 1) * noise + (1.0 - t_s.view(-1, 1, 1)) * mean_latent

    cond = torch.zeros_like(mean_latent)  # masked cond audio (drop_audio_cond=True)
    text = torch.zeros(B, T, device=device, dtype=torch.long)  # placeholder; drop_text=True
    style = style_prompt_vec.expand(B, -1).contiguous()  # [B, 512]
    start_time = torch.zeros(B, device=device, dtype=dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        _ = cfm.transformer(
            x=xt,
            cond=cond,
            text=text,
            time=t_s,
            drop_audio_cond=True,
            drop_text=True,
            style_prompt=style,
            start_time=start_time,
        )

def extract_paradigm(cfm, vae, device, dtype, paradigm_dir, target_sr,
                    layers_sample, vae_mode, t_value, style_prompt_vec):
    wav_paths = sorted(p for p in paradigm_dir.glob("*.wav"))
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files; skipping")
        return {}
    out: dict[str, np.ndarray] = {}
    done = 0
    for wp in wav_paths:
        audio = load_wav_mono(wp, target_sr)
        mean_latent = _encode_audio_to_latent_mean(
            audio, vae, device, dtype, vae_mode, target_sr
        )
        captured, handles, valid = _register_layer_hooks(
            cfm.transformer, layers_sample
        )
        try:
            _run_dit_forward(cfm, mean_latent, device, dtype, t_value, style_prompt_vec)
        finally:
            for h in handles:
                h.remove()
        arrs = [captured[i].to(torch.float32).cpu().numpy().squeeze(0) for i in valid]
        out[wp.stem] = np.stack(arrs, axis=0).astype(np.float32)
        done += 1
        if done % 10 == 0 or done == len(wav_paths):
            print(f"  [{paradigm_dir.name}] processed {done}/{len(wav_paths)}")
    return out

def _resolve_repo_root() -> Path:
    """Locate the DiffRhythm clone whose model/ package we import from.

    Read from the DIFFRHYTHM_DIR environment variable (required).
    """
    p = os.environ.get("DIFFRHYTHM_DIR")
    if p and Path(p).is_dir():
        return Path(p)
    raise FileNotFoundError(
        f"[{MODEL_KEY}] Cannot find the DiffRhythm code. Set DIFFRHYTHM_DIR "
        f"to a clone of it."
    )

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
    vae_hf_id = mc["vae_hf_id"]
    vae_hf_revision = mc["vae_hf_revision"]
    vae_mode = mc.get("vae_mode", "mean")
    layers_sample = list(mc["layers_sample"])

    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Config vae_mode must be 'mean'; got '{vae_mode}'."
    )

    print(f"[extract_{MODEL_KEY}] device=cuda dit_hf={hf_id} rev={revision[:12]}")
    print(f"[extract_{MODEL_KEY}] vae_hf={vae_hf_id} rev={vae_hf_revision[:12]}")
    print(f"[extract_{MODEL_KEY}] target_sr={target_sr} layers_sample={layers_sample}")
    print(f"[extract_{MODEL_KEY}] vae_mode={vae_mode} t_value={args.t_value}")

    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print(f"[extract_{MODEL_KEY}] repo_root={repo_root}")

    device = torch.device("cuda")
    dtype = torch.float16

    dit_repo = snapshot_download_pinned(hf_id, revision)
    vae_repo = snapshot_download_pinned(vae_hf_id, vae_hf_revision)
    print(f"[extract_{MODEL_KEY}] DiT ckpt dir: {dit_repo}")
    print(f"[extract_{MODEL_KEY}] VAE ckpt dir: {vae_repo}")

    print(f"[extract_{MODEL_KEY}] loading VAE (TorchScript) from {vae_repo}")
    vae = _load_vae_jit(vae_repo, device)
    print(f"[extract_{MODEL_KEY}] building CFM+DiT from repo config")
    cfm = _build_dit_model(repo_root, device, dtype)
    print(f"[extract_{MODEL_KEY}] loading DiT weights into CFM")
    cfm = _load_dit_weights(cfm, dit_repo)

    n_lay = len(cfm.transformer.transformer_blocks)
    print(f"[extract_{MODEL_KEY}] transformer_blocks count = {n_lay}")

    style_prompt_vec = _load_style_prompt(repo_root, device, dtype)
    print(f"[extract_{MODEL_KEY}] style_prompt_vec shape = {tuple(style_prompt_vec.shape)}")

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY, seed=seed)

    for paradigm in paradigms:
        pdir = args.stim_root / paradigm
        if not pdir.is_dir():
            warnings.warn(f"paradigm dir not found: {pdir}", RuntimeWarning)
            continue
        print(f"[extract_{MODEL_KEY}] paradigm={paradigm}")
        sd = extract_paradigm(
            cfm, vae, device, dtype, pdir, target_sr,
            layers_sample, vae_mode, args.t_value, style_prompt_vec,
        )
        if not sd:
            continue
        name_tag = f"_{args.output_tag}" if args.output_tag else ""
        out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
        if out_path.exists():
            print(f"[extract_{MODEL_KEY}] SKIP {paradigm} (exists: {out_path.name})")
            continue
        atomic_savez(out_path, _meta=run_meta, **sd)
        print(f"[extract_{MODEL_KEY}] wrote {out_path} ({len(sd)} stims)")

    print(f"[extract_{MODEL_KEY}] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
