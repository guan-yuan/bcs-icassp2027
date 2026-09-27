#!/usr/bin/env python3
"""Extract MuQ-large-msd-iter hidden states for the stimulus cues.

Runs on a GPU with the seeds of the config. The weights are downloaded at
the pinned revision and loaded with MuQ.from_pretrained; the forward runs in
float32 because MuQ produces NaN under float16 autocast.

MuQ API note:
  from muq import MuQ
  muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
  output = muq(wavs, output_hidden_states=True)
  # wavs is raw waveform tensor shape [batch, samples] @ 24 kHz
  # output.hidden_states: list/tuple of all layer outputs
  # output.last_hidden_state: final layer output

Output: {out}/muq_{paradigm}.npz  with {stim_name: array[n_layers, T, 1024]}
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

MODEL_KEY = "muq"
OUT_PREFIX = "muq"

def _extract_hidden_states(output) -> tuple:
    """MuQ may return a ModelOutput with .hidden_states or a tuple. Normalize."""
    hs = getattr(output, "hidden_states", None)
    if hs is not None:
        return tuple(hs)
    if isinstance(output, (list, tuple)):
        # Some builds: (last_hidden, hidden_states) or just hidden_states tuple
        last = output[-1]
        if isinstance(last, (list, tuple)):
            return tuple(last)
    raise RuntimeError(
        "MuQ output has no .hidden_states attribute; check output_hidden_states=True "
        "and upstream API compatibility."
    )

def _pad_and_stack(audios: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
    """Right-pad mono float32 audios to max length; return [B, L] tensor + lengths."""
    lens = [a.shape[0] for a in audios]
    max_len = max(lens)
    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        batch[i, : a.shape[0]] = a
    return torch.from_numpy(batch), lens

def extract_paradigm(
    model,
    device: torch.device,
    paradigm_dir: Path,
    target_sr: int,
    layers_sample: list[int],
    batch_size: int,
) -> dict[str, np.ndarray]:
    wav_paths = iter_wavs(paradigm_dir)
    out: dict[str, np.ndarray] = {}
    if not wav_paths:
        print(f"  [{paradigm_dir.name}] no .wav files found; skipping")
        return out

    clamped_layers: list[int] | None = None
    done = 0
    for batch in batched(wav_paths, batch_size):
        audios = [load_wav_mono(p, target_sr) for p in batch]
        wavs, _lens = _pad_and_stack(audios)
        wavs = wavs.to(device, non_blocking=True)

        # MuQ produces NaN under fp16 autocast, run in fp32.
        # ~300M params fits comfortably in 24 GB VRAM at fp32.
        with torch.no_grad():
            output = model(wavs, output_hidden_states=True)

        hs = _extract_hidden_states(output)
        if clamped_layers is None:
            clamped_layers = clamp_layers(layers_sample, len(hs), MODEL_KEY)

        # hs[i] shape: [B, T, H]
        stacked = torch.stack([hs[i] for i in clamped_layers], dim=0)
        stacked_np = stacked.to(torch.float32).cpu().numpy()
        # [n_layers, B, T, H]
        for b_idx, wav_path in enumerate(batch):
            stim_name = wav_path.stem
            out[stim_name] = stacked_np[:, b_idx, :, :].astype(np.float32)
        done += len(batch)
        if done % 10 == 0 or done == len(wav_paths):
            print(f"  [{paradigm_dir.name}] processed {done}/{len(wav_paths)}")

    return out

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

    seed = int(config["random_seeds"]["stimuli_base"])
    set_deterministic_seeds(seed)

    paradigms = args.paradigms or get_stim_paradigms(config)
    target_sr = int(model_cfg["sample_rate"])
    hf_id = model_cfg["hf_id"]
    revision = model_cfg["hf_revision"]
    layers_sample = list(model_cfg["layers_sample"])

    print(f"[extract_muq] device=cuda hf_id={hf_id} revision={revision[:12]}")
    print(f"[extract_muq] target_sr={target_sr} layers_sample={layers_sample}")
    print(f"[extract_muq] paradigms={paradigms}")

    from muq import MuQ  # official upstream API per HF model card
    from huggingface_hub import snapshot_download

    # Pin revision at the download layer (HF hub), since MuQ.from_pretrained
    # signature does not pass `revision=` to hf_hub_download. This enforces
    # reproducibility without patching third-party source.
    local_path = snapshot_download(repo_id=hf_id, revision=revision)
    muq = MuQ.from_pretrained(local_path)

    device = torch.device("cuda")
    muq = muq.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = build_metadata(config, MODEL_KEY, seed=seed)

    for paradigm in paradigms:
        paradigm_dir = args.stim_root / paradigm
        if not paradigm_dir.is_dir():
            warnings.warn(f"paradigm dir not found: {paradigm_dir}", RuntimeWarning)
            continue
        print(f"[extract_muq] paradigm={paradigm}")
        name_tag = f"_{args.output_tag}" if args.output_tag else ""
        out_path = args.out / f"{OUT_PREFIX}{name_tag}_{paradigm}.npz"
        if out_path.exists():
            print(f"[extract_muq] SKIP {paradigm} (exists: {out_path.name})")
            continue
        stim_dict = extract_paradigm(
            model=muq,
            device=device,
            paradigm_dir=paradigm_dir,
            target_sr=target_sr,
            layers_sample=layers_sample,
            batch_size=args.batch_size,
        )
        if not stim_dict:
            continue
        atomic_savez(out_path, _meta=run_meta, **stim_dict)
        print(f"[extract_muq] wrote {out_path} ({len(stim_dict)} stims)")

    print("[extract_muq] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
