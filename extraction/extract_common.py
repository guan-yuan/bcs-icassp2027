"""Shared utilities for the extraction scripts.

Responsibilities:
  - Config & paradigm enumeration (config/stimuli.yaml + config/checkpoints.yaml).
  - Audio I/O: load WAV as mono, resample to target SR.
  - Metadata I/O: load metadata.json → list of per-stim dicts.
  - Atomic file writes: npz and JSON (write to .tmp then rename).
  - Deterministic seeds: torch + numpy + cudnn.
  - Run metadata: seed + git SHA + hf_revision + timestamp + config hash.
  - GPU enforcement.
  - Layer clamping, WAV iteration, batching.
  - HuggingFace snapshot download with revision pinning.
  - State-dict safety check for strict=False loads.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Union[Path, str, None] = None) -> dict:
    """Load the run configuration.

    With no argument and no EXPERIMENT_CONFIG variable, merge the two release
    files config/checkpoints.yaml (``checkpoints``, ``models``) and
    config/stimuli.yaml (``stimuli``, ``audio``, ``random_seeds``).
    EXPERIMENT_CONFIG (or ``config_path``) names one YAML file holding all of
    these keys instead.
    """
    import yaml

    if config_path or os.environ.get("EXPERIMENT_CONFIG"):
        with open(Path(config_path or os.environ["EXPERIMENT_CONFIG"])) as f:
            return yaml.safe_load(f)
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    merged: dict = {}
    for name in ("stimuli.yaml", "checkpoints.yaml"):
        with open(cfg_dir / name) as f:
            merged.update(yaml.safe_load(f))
    return merged

def get_stim_paradigms(config: dict) -> List[str]:
    """Return the canonical paradigm name list (order = config key order)."""
    return list(config["stimuli"].keys())

def get_stim_dirs(config: dict, stimuli_root: Path) -> List[Path]:
    """Resolve on-disk stimulus directories that exist for each config paradigm."""
    stimuli_root = Path(stimuli_root)
    return [stimuli_root / p for p in get_stim_paradigms(config)
            if (stimuli_root / p).is_dir()]

# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_wav_mono(
    wav_path: Union[Path, str],
    target_sr: int,
    *,
    as_tensor: bool = False,
) -> Any:
    """Load a WAV file as mono, resample to *target_sr*.

    Parameters
    ----------
    wav_path : path to WAV file
    target_sr : desired sample rate (Hz)
    as_tensor : if True, return torch.Tensor [1, T] float32;
                if False, return np.ndarray [T] float32.
    """
    import soundfile as sf

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if as_tensor:
        import torch
        import torchaudio

        wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav  # [1, T] float32
    else:
        if sr != target_sr:
            import torchaudio
            import torch

            wav = torch.from_numpy(audio).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.squeeze(0).numpy()
        return audio  # [T] float32

# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def load_metadata(stim_root: Union[Path, str], paradigm: str) -> List[dict]:
    """Load metadata.json for a paradigm → list of per-stim dicts.

    The generated metadata.json has structure:
        {"_info": {...}, "config": {...}, "stimuli": [{...}, ...]}
    This function returns only the "stimuli" list.
    """
    p = Path(stim_root) / paradigm / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"metadata.json missing: {p}")
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        return raw.get("stimuli", [])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unexpected metadata.json format in {p}: {type(raw)}")

def load_metadata_raw(stim_root: Union[Path, str], paradigm: str) -> dict:
    """Load the full metadata.json dict (including _info and config).

    Use this only when you need the header or config info from the metadata.
    For stimulus iteration, use load_metadata() instead.
    """
    p = Path(stim_root) / paradigm / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"metadata.json missing: {p}")
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        return raw
    return {"stimuli": raw}

def stim_key(meta_entry: dict) -> str:
    """Map a metadata entry → npz key (== WAV filename stem)."""
    return Path(meta_entry["file"]).stem

# ---------------------------------------------------------------------------
# Iteration utilities
# ---------------------------------------------------------------------------

def iter_wavs(paradigm_dir: Union[Path, str]) -> List[Path]:
    """Return sorted list of .wav files in a paradigm directory."""
    return sorted(Path(paradigm_dir).glob("*.wav"))

def batched(items: Sequence, batch_size: int) -> Iterator:
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------

def clamp_layers(
    layers_sample: List[int],
    n_available: int,
    model_key: str,
) -> List[int]:
    """Clamp layer indices to [0, n_available-1], warning on out-of-range.

    Parameters
    ----------
    layers_sample : requested layer indices from config
    n_available : actual number of layers available from the model
    model_key : for diagnostic messages
    """
    clamped = []
    for li in layers_sample:
        if li < 0 or li >= n_available:
            import warnings
            warnings.warn(
                f"[{model_key}] layers_sample contains {li} but model has "
                f"{n_available} layers (0-{n_available-1}); clamping.",
                RuntimeWarning,
            )
            clamped.append(max(0, min(n_available - 1, li)))
        else:
            clamped.append(li)
    return sorted(set(clamped))

# ---------------------------------------------------------------------------
# HuggingFace snapshot
# ---------------------------------------------------------------------------

def snapshot_download_pinned(hf_id: str, revision: str) -> Path:
    """Download a HuggingFace repo snapshot at a pinned revision.

    Returns the local directory path. Caches via huggingface_hub.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=hf_id, revision=revision))

# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------

def set_deterministic_seeds(seed: int) -> None:
    """Pin determinism across torch, numpy, and Python random."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Run metadata & git
# ---------------------------------------------------------------------------

def git_commit() -> str:
    """Return current git HEAD SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

def _config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def build_metadata(
    config: dict,
    model_name: str,
    seed: int | None = None,
) -> Dict:
    """Build the metadata dict embedded into every output npz as ``_meta``."""
    model_cfg = config["models"].get(model_name, {})
    return {
        "model_name": model_name,
        "full_name": model_cfg.get("full_name", model_name),
        "hf_id": model_cfg.get("hf_id", ""),
        "hf_revision": model_cfg.get("hf_revision", ""),
        "seed": seed if seed is not None else config["random_seeds"]["stimuli_base"],
        "config_hash": _config_hash(config),
        "config_version": config.get("version", "unknown"),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "paradigm_list": get_stim_paradigms(config),
    }

# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------

def atomic_savez(path: Union[Path, str], /, **arrays) -> None:
    """Atomic npz write: write to .tmp then os.rename (POSIX atomic).

    Uses uncompressed np.savez (not np.savez_compressed) for speed.
    The ``_meta`` value, if a dict, is JSON-serialized to a 0-d object array.

    ``BCS_NPZ_DTYPE=float16`` switches to float16 storage
    (halves the representation footprint). It is a storage cast only: the
    forward pass still runs in the model's own dtype.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Serialize _meta dict to JSON string stored as 0-d array
    if "_meta" in arrays and isinstance(arrays["_meta"], dict):
        arrays = dict(arrays)
        arrays["_meta"] = np.array(json.dumps(arrays["_meta"]), dtype=object)

    _dtype = os.environ.get("BCS_NPZ_DTYPE", "").strip().lower()
    if _dtype in ("float16", "fp16", "half"):
        cast, n_overflow = {}, 0
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.dtype in (np.float32, np.float64):
                h = v.astype(np.float16)
                # fp16 tops out at 65504: a model whose activations exceed that
                # would be silently turned into inf. Keep float32 for such
                # arrays rather than losing data (float16 is a storage
                # optimisation only).
                if np.isfinite(v).all() and not np.isfinite(h).all():
                    n_overflow += 1
                    cast[k] = v
                else:
                    cast[k] = h
            else:
                cast[k] = v
        if n_overflow:
            print(f"  [WARN] BCS_NPZ_DTYPE=float16: {n_overflow} array(s) exceed "
                  f"the fp16 range in {path.name}; kept float32 for those")
        arrays = cast

    np.savez(tmp, **arrays)
    # np.savez may add .npz suffix; handle both cases
    tmp_actual = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    tmp_actual.rename(path)

def atomic_json_dump(path: Union[Path, str], obj) -> None:
    """Atomic JSON write: write to .tmp then os.rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.rename(path)

# ---------------------------------------------------------------------------
# State-dict safety
# ---------------------------------------------------------------------------

def assert_state_dict_safe(
    load_result,
    expected_missing_prefixes: Iterable[str] = (),
    expected_unexpected_prefixes: Iterable[str] = (),
    label: str = "model",
) -> None:
    """Check PyTorch load_state_dict(strict=False) results.

    Raises AssertionError if any missing/unexpected key is not in the whitelist.
    """
    miss = [k for k in load_result.missing_keys
            if not k.startswith(tuple(expected_missing_prefixes))]
    extra = [k for k in load_result.unexpected_keys
             if not k.startswith(tuple(expected_unexpected_prefixes))]
    if miss:
        raise AssertionError(
            f"[{label}] Unexpected missing keys (first 10): {miss[:10]}")
    if extra:
        raise AssertionError(
            f"[{label}] Unexpected unexpected keys (first 10): {extra[:10]}")
    print(f"  load_state_dict [{label}]: "
          f"{len(load_result.missing_keys)} expected-missing, "
          f"{len(load_result.unexpected_keys)} expected-unexpected")

# ---------------------------------------------------------------------------
# GPU enforcement
# ---------------------------------------------------------------------------

def require_gpu() -> None:
    """Assert CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. GPU is required for model inference; "
            "CPU fallback is disabled. Check the GPU driver (nvidia-smi)."
        )
