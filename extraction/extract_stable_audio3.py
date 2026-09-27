#!/usr/bin/env python3
"""Extractor: Stable Audio 3 Medium (stabilityai/stable-audio-3-medium).

Forward:
  SAME-L `ae.encode` -> deterministic mean latent z0 [B, 256, T_lat]
  -> x_t = t*eps + (1-t)*z0 at t = 0.01 (eps from the seed-42 CUDA stream,
     drawn in filename-sorted order; the same DiT convention as ACE-Step and DiffRhythm)
  -> one forward of the 1.4B rf_denoiser DiT with null conditioning
     (zero cross-attention context, zero global cond, zero inpaint local cond,
      no CFG)
  -> hook the 24 ContinuousTransformer blocks; the 64 prepended memory tokens
     are stripped so the token axis is pure time (DiT patch_size = 1, so no
     2-D flattening is needed).

Usage: BCS_STIM_ROOT=<stimulus root> extract_stable_audio3.py --out <dir> [--paradigms cue1,cue2]
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

from grid_common import (OUT, DEFAULT_PARADIGMS, Progress, T_SEQ, VramPeak,  # noqa
                       ALL_PARADIGMS,
                       frame_diagnostics, iter_paradigm, layer_ids,
                       load_wav_mono, save_npz, sha256_file)

MODEL_KEY = "stableaudio3_medium"
HF_ID = "stabilityai/stable-audio-3-medium"
HF_REV = "27b5a21b791b1b033d193a9e1e3ce78493f102f9"
SR = 44100
T_VALUE = 0.01
SEED = 42
N_MEMORY_TOKENS = 64          # config: model.diffusion.config.num_memory_tokens


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def resolve_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=HF_ID, revision=HF_REV,
                                  local_files_only=True))


def pin_deterministic_attention() -> str:
    """Force the SAME-L sliding-window attention onto a non-autotuned kernel.

    The autoencoder's windowed self-attention prefers `flex_attention` compiled
    with `mode="max-autotune-no-cudagraphs"`, i.e. a kernel chosen by runtime
    benchmarking. Autotuning is off so reruns are bit-identical.
    """
    import stable_audio_3.models.transformer as T
    was = T.flex_attention_compiled is not None
    T.flex_attention_compiled = None
    return f"flex_attention_compiled disabled (was_available={was})"


def build_model(device, dtype):
    """SAME-L autoencoder + the DiT, both read off the single checkpoint.

    The T5Gemma prompt conditioner is not instantiated: the
    forward zeroes the cross-attention context, and `to_cond_embed` is bias-free, so a
    zero context is numerically identical to the conditioner's output being
    dropped. Building only the DiT keeps the probe free of the text tower.
    """
    from stable_audio_3.loading_utils import load_autoencoder
    from stable_audio_3.models.diffusion import DiTWrapper
    from safetensors.torch import load_file

    attn_pin = pin_deterministic_attention()
    root = resolve_snapshot()
    cfg_path = root / "model_config.json"
    ckpt_path = root / "model.safetensors"
    config = json.loads(cfg_path.read_text())

    # --- SAME-L autoencoder (keys `pretransform.model.*` in the checkpoint)
    ae = load_autoencoder(str(cfg_path), str(ckpt_path), device="cpu")
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    # Deterministic mean latent. The released encoder carries the
    # training-time regulariser `mask_noise = 1e-3`, which is applied at
    # inference too (it does not depend on self.training) and would make ae.encode
    # stochastic. It is switched off on the encoder side only, as the
    # PupuJEPA probe disables that model's random masking.
    n_off = 0
    for m in ae.encoder.modules():
        if getattr(m, "mask_noise", 0):
            m.mask_noise = 0.0
            n_off += 1

    # --- DiT (keys `model.model.*` -> DiTWrapper `model.*`)
    dcfg = config["model"]["diffusion"]
    dit = DiTWrapper(diffusion_objective=dcfg.get("diffusion_objective", "v"),
                     modular_local_cond_configs=dcfg.get(
                         "modular_local_cond_configs", []),
                     **dcfg["config"])
    sd = load_file(str(ckpt_path))
    dit_sd = {k[len("model."):]: v for k, v in sd.items()
              if k.startswith("model.model.")}
    missing, unexpected = dit.load_state_dict(dit_sd, strict=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    del sd, dit_sd
    dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)

    meta = {
        "checkpoint_file": str(ckpt_path),
        "config_file": str(cfg_path),
        "encoder_mask_noise_disabled_blocks": n_off,
        "n_dit_blocks": len(dit.model.transformer.layers),
        "hidden_dim": int(dcfg["config"]["embed_dim"]),
        "latent_dim": int(config["model"]["io_channels"]),
        "downsampling_ratio": int(
            config["model"]["pretransform"]["config"]["downsampling_ratio"]),
        "cross_attn_tokens": 257,     # 256 prompt tokens + 1 seconds_total
        "attention_pin": attn_pin,
    }
    return ae, dit, meta


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_latent(ae, wav: torch.Tensor, device, dtype) -> torch.Tensor:
    """mono [1, T] -> stereo [1, 2, T] -> SAME-L latent [1, 256, T_lat]."""
    x = wav.repeat(2, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return ae.encode_audio(x, chunked=False)


@torch.no_grad()
def run_clip(dit, z0: torch.Tensor, ids: list[int], device, dtype, n_cross: int):
    B, C, T_lat = z0.shape
    eps = torch.randn(z0.shape, device=device, dtype=dtype)      # seed-42 stream
    t = torch.full((B,), T_VALUE, device=device, dtype=torch.float32)
    x_t = t.view(-1, 1, 1).to(dtype) * eps + (1.0 - t.view(-1, 1, 1).to(dtype)) * z0

    # Conditioning is null / zero throughout, CFG off.
    cross = torch.zeros(B, n_cross, dit.model.cond_token_dim,
                        device=device, dtype=dtype)
    glob = torch.zeros(B, 768, device=device, dtype=dtype)
    local = torch.zeros(B, 257, T_lat, device=device, dtype=dtype)

    caught: dict[int, torch.Tensor] = {}
    handles = []
    blocks = dit.model.transformer.layers
    for i in ids:
        handles.append(blocks[i].register_forward_hook(
            lambda m, inp, out, i=i: caught.__setitem__(i, out.detach())))
    try:
        dit(x_t, t, cross_attn_cond=cross, cross_attn_mask=None,
            global_cond=glob, local_add_cond=local,
            cfg_scale=1.0, cfg_dropout_prob=0.0, use_checkpointing=False)
    finally:
        for h in handles:
            h.remove()

    stack = []
    for i in ids:
        h = caught[i].to(torch.float32)                 # [1, 64 + T_lat, H]
        assert h.shape[1] == N_MEMORY_TOKENS + T_lat, (h.shape, T_lat)
        stack.append(h[0, N_MEMORY_TOKENS:, :])         # strip memory tokens
    return torch.stack(stack).cpu().numpy()             # [n_layers, T_lat, H]


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paradigms", default=",".join(ALL_PARADIGMS[:4]),
                    help="comma-separated cues (default: the four cues of the paper)")
    args = ap.parse_args()
    if not os.environ.get("BCS_STIM_ROOT"):
        ap.error("set BCS_STIM_ROOT to the stimulus root written by stimuli/generate_stimuli.py")

    torch.manual_seed(SEED)          # seeds the CUDA stream too
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    dtype = torch.bfloat16           # fixed compute dtype

    ae, dit, mm = build_model(device, dtype)
    n_blocks = mm["n_dit_blocks"]
    hidden = mm["hidden_dim"]
    ids = layer_ids(n_blocks)
    print(f"[{MODEL_KEY}] L={n_blocks} H={hidden} layer_ids={ids} "
          f"(mask_noise off in {mm['encoder_mask_noise_disabled_blocks']} "
          f"encoder blocks)", flush=True)

    outdir = Path(args.out)
    diag = {"model_key": MODEL_KEY, "hf_id": HF_ID, "hf_revision": HF_REV,
            "architecture": "dit",
            "checkpoint_file": mm["checkpoint_file"],
            "checkpoint_sha256": sha256_file(mm["checkpoint_file"]),
            "layers_total": n_blocks, "hidden_dim": hidden, "layer_ids": ids,
            "dtype": "bfloat16",
            "captured_module": "model.transformer.layers[i] "
                               "(64 memory tokens stripped)",
            "conditioning": "null: zero cross-attn context (257 tokens), zero "
                            "global cond, zero inpaint local cond, no CFG, "
                            "t=0.01",
            "sample_rate": SR,
            "front_end": "SAME-L autoencoder latent (learned)",
            "latent_dim": mm["latent_dim"],
            "downsampling_ratio": mm["downsampling_ratio"],
            "encoder_mask_noise_disabled_blocks":
                mm["encoder_mask_noise_disabled_blocks"],
            "attention_pin": mm["attention_pin"],
            "t_value": T_VALUE, "eps_seed": SEED,
            "paradigms": {}}

    with VramPeak() as vp:
        for pdm in args.paradigms.split(","):
            reps, nframes, shapes = {}, {}, None
            items = iter_paradigm(pdm)
            prog = Progress(f"{MODEL_KEY}/{pdm}", len(items))
            for stem, path in items:
                wav = load_wav_mono(path, SR)
                z0 = encode_latent(ae, wav, device, dtype)
                arr = run_clip(dit, z0, ids, device, dtype, mm["cross_attn_tokens"])
                reps[stem] = arr
                nframes[stem] = arr.shape[1]
                prog.tick()
                if shapes is None:
                    # DiT patch_size = 1, so the token axis is already pure
                    # time and no 2-D flattening is needed: the
                    # before/after shapes are identical by construction. The
                    # only reshaping is stripping the 64 prepended memory
                    # tokens, recorded separately as shape_hooked.
                    tl = int(z0.shape[2])
                    shapes = {"audio_samples": int(wav.shape[-1]),
                              "latent_shape": list(z0.shape),
                              "n_memory_tokens": N_MEMORY_TOKENS,
                              "shape_hooked": [1, N_MEMORY_TOKENS + tl, hidden],
                              "shape_before_flatten": [1, tl, hidden],
                              "shape_after_flatten": [1, tl, hidden]}
            secs = prog.done()
            npz = outdir / f"{MODEL_KEY}_{pdm}.npz"
            save_npz(npz, reps, diag)
            fr = float(np.mean([n for n in nframes.values()]))
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
