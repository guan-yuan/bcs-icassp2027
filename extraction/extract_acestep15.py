#!/usr/bin/env python3
"""Extract ACE-Step 1.5 DiT hidden states for the stimulus cues.

Weights: the DiT in subfolder `acestep-v15-turbo` (config key `dit_subfolder`)
and the VAE in subfolder `vae` of ACE-Step/Ace-Step1.5 at the revision in
config/checkpoints.yaml.

DiT flow-matching probe:
  1. VAE.encode(audio).latent_dist.mean  (deterministic, `vae_mode: mean`)
  2. Build zero context_latents (src=zeros || chunk_masks=zeros) matching latent T.
  3. DiT forward at small t=0.01 with model.null_condition_emb (null text cond).
  4. Hooks on model.decoder.layers[i] for layers_sample → captured activations.

Runs on a GPU under float16 autocast with the seeds of the config. Both
snapshots are pinned to the configured revision; the DiT is loaded with
AutoModel.from_pretrained(trust_remote_code=True); the VAE latent is the
posterior mean (vae_mode: mean, the only mode accepted).

Output: {out}/acestep15_{paradigm}.npz with {stim_name: array[n_layers, T_latent, H=2048]}
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

MODEL_KEY = "acestep15"
OUT_PREFIX = "acestep15"

# The loaded `AceStepConditionGenerationModel` exposes `.decoder` (DiT),
# `.null_condition_emb`, `.encoder`, `.tokenizer` and `.detokenizer`; the probe
# calls `.decoder` and reads `.null_condition_emb` only.

def _load_vae(vae_dir: Path, device: torch.device, dtype):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
    vae = vae.to(device=device, dtype=dtype).eval()
    return vae

def _load_dit_model(dit_dir: Path, device: torch.device, dtype):
    """Load the full AceStep generation model via trust_remote_code from dit_dir.

    Returns (model, decoder, null_condition_emb).
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        str(dit_dir),
        trust_remote_code=True,
        low_cpu_mem_usage=False,  # disable meta-tensor init
    )
    model = model.to(device=device, dtype=dtype).eval()

    # Sanity-check: the DiT is `model.decoder`, null cond is `model.null_condition_emb`.
    if not hasattr(model, "decoder"):
        raise RuntimeError(
            f"[{MODEL_KEY}] Loaded model has no `.decoder` (DiT). "
            f"Upstream ACE-Step API changed; aborting."
        )
    if not hasattr(model, "null_condition_emb"):
        raise RuntimeError(
            f"[{MODEL_KEY}] Loaded model has no `.null_condition_emb`. "
            f"Upstream ACE-Step API changed; aborting."
        )
    return model

def _encode_audio_to_latent(
    audio_np: np.ndarray,
    vae,
    device: torch.device,
    dtype,
    vae_mode: str,
) -> torch.Tensor:
    """Encode one audio clip to VAE latent using .latent_dist.mean.

    Returns latent tensor of shape [1, T_latent, C_latent].
    """
    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Only vae_mode='mean' is supported "
        f"(probe determinism); got '{vae_mode}'."
    )
    # The Oobleck VAE takes stereo input [B, 2, samples]: the mono clip is
    # duplicated to two channels.
    a = torch.from_numpy(audio_np).to(device=device, dtype=dtype)  # [T]
    a = a.unsqueeze(0).repeat(2, 1)  # [2, T]
    a = a.unsqueeze(0)  # [1, 2, T]
    with torch.no_grad():
        enc = vae.encode(a)
        latent = enc.latent_dist.mean  # deterministic (no .sample())
    # AutoencoderOobleck returns [B, C_latent, T_latent]; transpose to [B, T, C]
    latent = latent.transpose(1, 2).contiguous()
    return latent

def _register_layer_hooks(decoder, layers_sample: list[int]):
    """Attach forward hooks on decoder.layers[i] to capture outputs.

    Returns (captured dict, handles list). Each layer output is the first
    element of the tuple returned by AceStepDiTLayer.forward, shape [B, T_lat, H].
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    n_layers = len(decoder.layers)
    valid = []
    for idx in layers_sample:
        if 0 <= idx < n_layers:
            valid.append(idx)
        else:
            warnings.warn(
                f"[{MODEL_KEY}] layers_sample idx {idx} OOR (n_layers={n_layers}); dropping.",
                RuntimeWarning,
            )

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            # AceStepDiTLayer returns (hidden_states, [self_attn_weights], [cross_attn_weights])
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
        return hook

    for idx in valid:
        h = decoder.layers[idx].register_forward_hook(make_hook(idx))
        handles.append(h)
    return captured, handles, valid

def _run_dit_forward(
    model,
    latent: torch.Tensor,        # [1, T_lat, C_lat]
    device: torch.device,
    dtype,
    t_value: float,
    config,
) -> None:
    """Run one DiT forward with null text conditioning at small t.

    hidden_states = x_t = t*noise + (1-t)*x0   (with x0 = VAE mean)
    context_latents = cat([src_latents=0, chunk_masks=0], dim=-1)
    encoder_hidden_states = null_condition_emb  (already trained null)
    """
    B, T_lat, C_lat = latent.shape
    # Flow-matching xt: small t gives xt ≈ x0.
    noise = torch.randn_like(latent)
    t_scalar = torch.full((B,), t_value, device=device, dtype=dtype)
    t_expand = t_scalar.view(-1, 1, 1)
    xt = t_expand * noise + (1.0 - t_expand) * latent

    # context_latents: src_latents concat chunk_masks
    # src_latents has shape [B, T_lat, C_lat], chunk_masks [B, T_lat, C_lat]
    # (from upstream: context_latents = cat([src_latents, chunk_masks], dim=-1))
    src_latents = torch.zeros_like(latent)
    chunk_masks = torch.zeros_like(latent)
    context_latents = torch.cat([src_latents, chunk_masks], dim=-1)  # [B, T_lat, 2*C_lat]

    # Attention mask: all ones (no padding), shape [B, T_lat]
    attn_mask = torch.ones(B, T_lat, device=device, dtype=torch.long)

    # Encoder hidden states: null_condition_emb expanded. In generation it's
    # broadcast over a short fixed-length conditioning sequence; length 1 works
    # because downstream cross-attn handles any encoder seq length.
    null_cond = model.null_condition_emb.to(device=device, dtype=dtype)  # [1, 1, H_enc]
    encoder_hidden = null_cond.expand(B, 1, null_cond.shape[-1]).contiguous()
    encoder_attn = torch.ones(B, 1, device=device, dtype=torch.long)

    decoder = model.decoder
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        _ = decoder(
            hidden_states=xt,
            timestep=t_scalar,
            timestep_r=t_scalar,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attn,
            context_latents=context_latents,
            use_cache=False,
            return_hidden_states=None,
        )

def extract_paradigm(
    model,
    vae,
    device: torch.device,
    dtype,
    paradigm_dir: Path,
    target_sr: int,
    layers_sample: list[int],
    vae_mode: str,
    t_value: float,
    config,
) -> dict[str, np.ndarray]:
    wav_paths = sorted(p for p in paradigm_dir.glob("*.wav"))
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files found; skipping")
        return {}

    out: dict[str, np.ndarray] = {}
    done = 0
    for wav_path in wav_paths:
        audio = load_wav_mono(wav_path, target_sr)
        latent = _encode_audio_to_latent(audio, vae, device, dtype, vae_mode)

        captured, handles, valid_layers = _register_layer_hooks(
            model.decoder, layers_sample
        )
        try:
            _run_dit_forward(model, latent, device, dtype, t_value, config)
        finally:
            for h in handles:
                h.remove()

        # Stack captured in layers_sample order
        arrs = []
        for idx in valid_layers:
            h = captured[idx]  # [1, T_lat, H]
            arrs.append(h.to(torch.float32).cpu().numpy().squeeze(0))  # [T_lat, H]
        stim_arr = np.stack(arrs, axis=0).astype(np.float32)  # [n_layers, T, H]
        out[wav_path.stem] = stim_arr

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
    ap.add_argument("--t_value", type=float, default=0.01,
                    help="Flow-matching time for DiT forward (small t ≈ clean latent)")
    ap.add_argument("--output_tag", type=str, default="",
                    help="Optional filename tag inserted between OUT_PREFIX and "
                         "paradigm: {OUT_PREFIX}_{tag}_{paradigm}.npz.")
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
    dit_subfolder = model_cfg.get("dit_subfolder", "acestep-v15-turbo")
    vae_hf_id = model_cfg.get("vae_hf_id", hf_id)
    vae_hf_revision = model_cfg.get("vae_hf_revision", revision)
    vae_subfolder = model_cfg.get("vae_subfolder", "vae")
    vae_mode = model_cfg.get("vae_mode", "mean")
    layers_sample = list(model_cfg["layers_sample"])

    # Only the deterministic `.mean` path is supported.
    assert vae_mode == "mean", (
        f"[{MODEL_KEY}] Config `vae_mode` must be 'mean' (deterministic probe). "
        f"Got '{vae_mode}'. Fix config/checkpoints.yaml."
    )

    print(f"[extract_{MODEL_KEY}] device=cuda hf_id={hf_id} revision={revision[:12]}")
    print(f"[extract_{MODEL_KEY}] target_sr={target_sr} layers_sample={layers_sample}")
    print(f"[extract_{MODEL_KEY}] vae_mode={vae_mode} t_value={args.t_value}")
    print(f"[extract_{MODEL_KEY}] paradigms={paradigms}")

    device = torch.device("cuda")
    dtype = torch.float16

    # Resolve snapshot dirs
    print(f"[extract_{MODEL_KEY}] snapshot_download: DiT repo={hf_id} rev={revision[:12]}")
    dit_repo_dir = snapshot_download_pinned(hf_id, revision)
    dit_dir = dit_repo_dir / dit_subfolder
    if not dit_dir.is_dir():
        raise FileNotFoundError(
            f"[{MODEL_KEY}] DiT subfolder not found: {dit_dir}"
        )

    print(f"[extract_{MODEL_KEY}] snapshot_download: VAE repo={vae_hf_id} rev={vae_hf_revision[:12]}")
    vae_repo_dir = snapshot_download_pinned(vae_hf_id, vae_hf_revision)
    vae_dir = vae_repo_dir / vae_subfolder
    if not vae_dir.is_dir():
        raise FileNotFoundError(
            f"[{MODEL_KEY}] VAE subfolder not found: {vae_dir}"
        )

    print(f"[extract_{MODEL_KEY}] loading VAE from {vae_dir}")
    vae = _load_vae(vae_dir, device, dtype)

    print(f"[extract_{MODEL_KEY}] loading DiT model from {dit_dir}")
    model = _load_dit_model(dit_dir, device, dtype)
    n_layers_actual = len(model.decoder.layers)
    print(f"[extract_{MODEL_KEY}] decoder.layers count = {n_layers_actual}")

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY, seed=seed)

    for paradigm in paradigms:
        paradigm_dir = args.stim_root / paradigm
        if not paradigm_dir.is_dir():
            warnings.warn(f"paradigm dir not found: {paradigm_dir}", RuntimeWarning)
            continue
        print(f"[extract_{MODEL_KEY}] paradigm={paradigm}")
        stim_dict = extract_paradigm(
            model=model,
            vae=vae,
            device=device,
            dtype=dtype,
            paradigm_dir=paradigm_dir,
            target_sr=target_sr,
            layers_sample=layers_sample,
            vae_mode=vae_mode,
            t_value=args.t_value,
            config=config,
        )
        if not stim_dict:
            continue
        name_tag = f"_{args.output_tag}" if args.output_tag else ""
        out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
        atomic_savez(out_path, _meta=run_meta, **stim_dict)
        print(f"[extract_{MODEL_KEY}] wrote {out_path} ({len(stim_dict)} stims)")

    print(f"[extract_{MODEL_KEY}] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
