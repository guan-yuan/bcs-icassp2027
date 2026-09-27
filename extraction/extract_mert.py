#!/usr/bin/env python3
"""Extract MERT-v1-330M / MERT-v1-95M hidden states for the stimulus cues.

Model: HuggingFace AutoModel (HubertModel-like SSL encoder).
Probe: encode-forward with output_hidden_states=True.
The model key defaults to mert (MERT-v1-330M); MODEL_KEY_OVERRIDE=mert_95m
selects MERT-v1-95M.
Output: {out}/<model key>_{paradigm}.npz  with {stim_name: array[n_layers, T, H]}

All model settings come from config/checkpoints.yaml.
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
    batched,
    build_metadata,
    clamp_layers,
    get_stim_paradigms,
    iter_wavs,
    load_config,
    load_wav_mono,
    require_gpu,
    set_deterministic_seeds,
)

MODEL_KEY = os.environ.get("MODEL_KEY_OVERRIDE", "mert")

def extract_paradigm(
    model,
    processor,
    device: torch.device,
    paradigm_dir: Path,
    target_sr: int,
    layers_sample: list[int],
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Run MERT forward on every WAV in paradigm_dir.

    Returns dict {stim_name: ndarray[n_layers, T, H]}.
    """
    wav_paths = iter_wavs(paradigm_dir)
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files; skipping")
        return {}

    out: dict[str, np.ndarray] = {}
    clamped: list[int] | None = None
    done = 0

    for batch_paths in batched(wav_paths, batch_size):
        # Load audio as numpy arrays for the Wav2Vec2FeatureExtractor
        audios = [load_wav_mono(p, target_sr) for p in batch_paths]
        inputs = processor(
            audios,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            res = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers + 1) tensors, each [B, T, H]
        # Index 0 = embedding output; index k = output of transformer layer k
        hs = res.hidden_states
        if clamped is None:
            clamped = clamp_layers(layers_sample, len(hs), MODEL_KEY)

        # Stack selected layers → [n_sel, B, T, H] → float32 numpy
        stacked = torch.stack([hs[i] for i in clamped], dim=0)
        stacked_np = stacked.to(torch.float32).cpu().numpy()

        for b_idx, wav_path in enumerate(batch_paths):
            out[wav_path.stem] = stacked_np[:, b_idx, :, :]  # [n_layers, T, H]

        done += len(batch_paths)
        if done % 50 == 0 or done == len(wav_paths):
            print(f"  [{paradigm_dir.name}] {done}/{len(wav_paths)}")

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
    ap.add_argument("--paradigms", nargs="*", default=None)
    ap.add_argument("--stim_root", type=Path, required=True,
                    help="stimulus root written by stimuli/generate_stimuli.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_tag", type=str, default="",
                    help="Optional filename tag inserted between MODEL_KEY and "
                         "paradigm: {MODEL}_{tag}_{paradigm}.npz.")
    args = ap.parse_args()

    require_gpu()
    config = load_config()
    model_cfg = config["models"][MODEL_KEY]
    set_deterministic_seeds(int(config["random_seeds"]["stimuli_base"]))

    paradigms = args.paradigms or get_stim_paradigms(config)
    _assert_model_key(config, args.out)
    target_sr = int(model_cfg["sample_rate"])
    hf_id = model_cfg["hf_id"]
    revision = model_cfg["hf_revision"]
    layers_sample = list(model_cfg["layers_sample"])

    print(f"[{MODEL_KEY}] hf_id={hf_id} rev={revision[:12]} sr={target_sr}")
    print(f"[{MODEL_KEY}] layers_sample={layers_sample} paradigms={paradigms}")

    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True,
    )
    if int(processor.sampling_rate) != target_sr:
        warnings.warn(
            f"processor.sampling_rate={processor.sampling_rate} != config "
            f"target_sr={target_sr}; using processor value",
            RuntimeWarning,
        )
        target_sr = int(processor.sampling_rate)

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True,
    ).to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY)

    for paradigm in paradigms:
        pdir = args.stim_root / paradigm
        if not pdir.is_dir():
            warnings.warn(f"paradigm dir not found: {pdir}", RuntimeWarning)
            continue
        name_tag = f"_{args.output_tag}" if args.output_tag else ""
        out_path = args.out / f"{MODEL_KEY}{name_tag}_{paradigm}.npz"
        if out_path.exists():
            print(f"  SKIP {paradigm} (exists)")
            continue

        stim_dict = extract_paradigm(
            model, processor, device, pdir, target_sr, layers_sample,
            batch_size=args.batch_size,
        )
        if stim_dict:
            atomic_savez(out_path, _meta=run_meta, **stim_dict)
            print(f"  wrote {out_path.name} ({len(stim_dict)} stims)")

    print(f"[{MODEL_KEY}] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
