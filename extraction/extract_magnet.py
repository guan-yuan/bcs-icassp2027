#!/usr/bin/env python3
"""Extract MAGNeT-S / MAGNeT-M hidden states for the stimulus cues.

MAGNeT (Meta AudioCraft) is a non-autoregressive masked parallel decoder
operating on EnCodec 32 kHz tokens. This is a *representational probe*:
we forward clean (unmasked) EnCodec codes through the bidirectional
StreamingTransformer with null cross-attention (text=None) and capture
per-layer hidden states; nothing is generated.

Pipeline per stimulus:
  1. Load wav (mono) -> resample to 32 kHz
  2. `compression_model.encode(audio)` -> EnCodec codes [B, K, T_codes]
  3. Sum codebook embeddings via `lm.emb[k]` -> inputs_embeds [B, T, D]
  4. `transformer(inputs_embeds, cross_attention_src=null_cond)` with hooks
     on `transformer.layers[li]` for li in layers_sample
  5. Stack captured layers -> [n_layers, T, D] npz entry

Runs on a GPU with the seeds of the config, one clip per forward. The
Hugging Face repo is downloaded at the revision in config/checkpoints.yaml and
audiocraft.MAGNeT.get_pretrained() loads state_dict.bin and
compression_state_dict.bin from that snapshot (audiocraft 1.3.0's
get_pretrained takes no revision argument). The LM runs in float16, the EnCodec
compression model in float32.

The model key defaults to magnet_medium; MODEL_KEY_OVERRIDE=magnet_small
selects MAGNeT-S (config/checkpoints.yaml).

Output: {out}/<model key>_{paradigm}.npz
  {stim_name: array[n_layers, T, H]}  plus  _meta: run_meta
"""
from __future__ import annotations

import argparse
import os
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
    snapshot_download_pinned,
)

MODEL_KEY = os.environ.get("MODEL_KEY_OVERRIDE", "magnet_medium")
OUT_PREFIX = MODEL_KEY

def _build_null_cross_attention_src(
    batch_size: int, hidden_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Null text conditioning: zero tensor of shape [B, 1, D].

    MAGNeT's StreamingTransformer has cross-attention layers that expect a
    (B, T_cond, D) tensor as `cross_attention_src`. We pass all-zeros to
    provide a valid shape without any textual bias, the representational
    probe is a property of the audio token stream only.
    """
    return torch.zeros(batch_size, 1, hidden_dim, device=device, dtype=dtype)

def _forward_capture(
    model,
    audio: torch.Tensor,
    layers_sample: list[int],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], int]:
    """Run MAGNeT forward with hooks on `transformer.layers[li]`.

    Returns (captured: {layer_idx: [T, D] cpu tensor}, n_layers).
    """
    lm = model.lm
    transformer = lm.transformer
    compression_model = model.compression_model
    n_layers = len(transformer.layers)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for li in layers_sample:
        if li >= n_layers:
            continue

        def make_hook(idx):
            def hook_fn(_module, _input, output):
                t = output[0] if isinstance(output, tuple) else output
                # [B, T, D] -> [T, D] since batch=1
                captured[idx] = t.detach().squeeze(0).to(torch.float32).cpu()

            return hook_fn

        hooks.append(transformer.layers[li].register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            # [1, T] -> [1, 1, T] for EnCodec
            audio_in = audio.unsqueeze(0).to(device)
            encoded = compression_model.encode(audio_in)
            # audiocraft EnCodec returns List[(codes, scale)]; take first frame
            codes = encoded[0][0]
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)  # [1, K, T_codes]
            B, K, _T = codes.shape

            # Sum codebook embeddings (shared pattern w/ MusicGen & MAGNeT LM)
            if not hasattr(lm, "emb"):
                raise RuntimeError("MAGNeT LM missing `emb` ModuleList")
            n_emb = min(K, len(lm.emb))
            inputs_embeds = lm.emb[0](codes[:, 0, :])
            for k in range(1, n_emb):
                inputs_embeds = inputs_embeds + lm.emb[k](codes[:, k, :])
            # inputs_embeds: [1, T, D]

            hidden_dim = inputs_embeds.shape[-1]
            null_src = _build_null_cross_attention_src(
                B, hidden_dim, device, inputs_embeds.dtype
            )

            # StreamingTransformer takes (x, cross_attention_src=...)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = transformer(inputs_embeds, cross_attention_src=null_src)
    finally:
        for h in hooks:
            h.remove()

    return captured, n_layers

def extract_paradigm(
    model,
    device: torch.device,
    paradigm_dir: Path,
    target_sr: int,
    layers_sample: list[int],
) -> dict[str, np.ndarray]:
    wav_paths = iter_wavs(paradigm_dir)
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files; skip")
        return {}

    clamped: list[int] | None = None
    out: dict[str, np.ndarray] = {}
    for i, wav_path in enumerate(wav_paths):
        audio = load_wav_mono(wav_path, target_sr, as_tensor=True)
        captured, n_layers = _forward_capture(model, audio, layers_sample, device)
        if clamped is None:
            clamped = clamp_layers(layers_sample, n_layers, MODEL_KEY)
        # Stack the clamped layers in requested order
        stacked = np.stack(
            [captured[li].numpy() for li in clamped if li in captured], axis=0
        )  # [n_layers, T, D]
        out[wav_path.stem] = stacked.astype(np.float32)

        if (i + 1) % 10 == 0 or (i + 1) == len(wav_paths):
            print(f"  [{paradigm_dir.name}] processed {i + 1}/{len(wav_paths)}")
    return out

def _assert_model_key(config, out_dir):
    """Print the model key in force and check that it is a key of the config."""
    src = ("MODEL_KEY_OVERRIDE" if os.environ.get("MODEL_KEY_OVERRIDE")
           else "script default")
    print(f"[model-key] MODEL_KEY={MODEL_KEY!r} (source: {src})", flush=True)
    if MODEL_KEY not in config["models"]:
        raise SystemExit(f"[model-key] {MODEL_KEY!r} is not a key of "
                         f"config.models {list(config['models'])}")
    print(f"[model-key] outputs -> {out_dir}/{MODEL_KEY}_<paradigm>.npz", flush=True)

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
    _assert_model_key(config, args.out)
    target_sr = int(model_cfg["sample_rate"])
    hf_id = model_cfg["hf_id"]
    revision = model_cfg["hf_revision"]
    layers_sample = list(model_cfg["layers_sample"])

    print(f"[extract_magnet] device=cuda hf_id={hf_id} revision={revision[:12]}")
    print(f"[extract_magnet] target_sr={target_sr} layers_sample={layers_sample}")
    print(f"[extract_magnet] paradigms={paradigms}")

    # audiocraft 1.3.0's get_pretrained takes no revision argument, and given a
    # repo id it would fetch the latest files. Download the pinned revision
    # first and pass the local snapshot directory: audiocraft then reads
    # state_dict.bin and compression_state_dict.bin from that directory.
    from audiocraft.models import MAGNeT

    local_dir = snapshot_download_pinned(hf_id, revision)
    print(f"[extract_magnet] snapshot {local_dir}")
    model = MAGNeT.get_pretrained(str(local_dir))
    device = torch.device("cuda")
    model.lm = model.lm.half().to(device).eval()
    model.compression_model = model.compression_model.to(device).eval()

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
            print(f"[extract_magnet] paradigm={paradigm}")
            stim_dict = extract_paradigm(
                model=model,
                device=device,
                paradigm_dir=paradigm_dir,
                target_sr=target_sr,
                layers_sample=layers_sample,
            )
            if not stim_dict:
                continue
            name_tag = f"_{args.output_tag}" if args.output_tag else ""
            out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
            if out_path.exists():
                print(f"[extract_magnet] SKIP {paradigm} (exists: {out_path.name})")
                continue
            atomic_savez(out_path, _meta=run_meta, **stim_dict)
            print(f"[extract_magnet] wrote {out_path} ({len(stim_dict)} stims)")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[extract_magnet] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
