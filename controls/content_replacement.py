#!/usr/bin/env python3
"""Content replacement for the temporal-proximity set (noise / shuffle / clicks).

The stimulus loop of stimuli/generate_stimuli.py is replayed note by note with
its own seeds, so every onset and duration lands on exactly the same sample
as in the original stimulus, and the metadata (ratio, cluster, onsets) is
copied unchanged. Only the content of each note is replaced:
    cd_noise    white-noise burst, energy matched to the replaced tone
    cd_shuffle  a random timbre at a random carrier frequency per note
    cd_silence  a click of peak 1e-3 (1 ms) at the note onset ("near-silent clicks")
--selftest replays the loop with the original sine content and checks that
the WAVs are byte-identical to the original set (onsets are then sample-exact).

Usage
  BCS_STIM_ROOT=<original stimuli> python3 controls/content_replacement.py --all --out <dir>
  BCS_STIM_ROOT=<original stimuli> python3 controls/content_replacement.py --check --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np



class P:
    """Paths (set from the command line / environment) and fixed constants."""
    HERE = Path(__file__).resolve().parent
    SCRIPTS = HERE.parent / "stimuli"
    SRC_STIM = Path(os.environ.get("BCS_STIM_ROOT", ""))   # the original stimulus set (required)
    SRC_TEMP_META = SRC_STIM / "temp_proximity/metadata.json"
    PARADIGM = "temp_proximity"
    ARMS = ("cd_noise", "cd_shuffle", "cd_silence")
    ARM_ID = {"cd_noise": 1, "cd_shuffle": 2, "cd_silence": 3}
    ROOT_SEED = 20260919
    SILENCE_CLICK_PEAK = 1e-3            # click peak relative to full scale
    SILENCE_CLICK_MS = 1.0

sys.path.insert(0, str(P.SCRIPTS))
sys.path.insert(0, str(P.HERE.parent / "extraction"))
import generate_stimuli as G      # noqa: E402  original renderer
import stimuli_rules as SR        # noqa: E402  original seed/name rules
import soundfile as sf            # noqa: E402

SR_HZ = G.SAMPLE_RATE
CFG = G.CONFIG["stimuli"]["temp_proximity"]
TIMBRES = G.CONFIG["stimuli"]["timbre_ablation"]["timbres"]
ANCHORS_HZ = G.CONFIG["stimuli"]["freq_proximity"]["anchors_hz"]


def _sine_note(amp: float, phase: float, dur: float) -> np.ndarray:
    """The note this substitution replaces: the original generator's own line."""
    return G.generate_tone(CFG["freq_hz"], dur, amplitude=amp, phase=phase)


def _match_energy(w: np.ndarray, amp: float, phase: float,
                  dur: float) -> np.ndarray:
    """Scale `w` so its RMS equals that of the sine it replaces (post-fade).

    A scalar gain: the envelope the fade imposes is untouched, so only the
    content of the note changes, never its level or its shape in time.
    """
    ref = float(np.sqrt(np.mean(_sine_note(amp, phase, dur) ** 2)))
    cur = float(np.sqrt(np.mean(w ** 2)))
    return w * (ref / cur) if cur > 0 else w


def _render_note(arm: str, amp: float, phase: float, dur: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = int(SR_HZ * dur)
    if arm == "cd_noise":
        w = G._apply_fade(rng.standard_normal(n), SR_HZ, G.FADE_MS)
        return _match_energy(w, amp, phase, dur), {"kind": "white_noise"}
    if arm == "cd_shuffle":
        ti = int(rng.integers(len(TIMBRES)))
        fi = int(rng.integers(len(ANCHORS_HZ)))
        spec, f = TIMBRES[ti], float(ANCHORS_HZ[fi])
        timbre_rng = np.random.RandomState(int(rng.integers(0, 2 ** 31 - 1)))
        w = G._render_timbre(spec, f, dur, timbre_rng).astype(np.float64)
        return _match_energy(w, amp, phase, dur), {"kind": "shuffled",
                                                   "timbre": spec["id"],
                                                   "freq_hz": f}
    if arm == "_identity":
        # the original sine note; --selftest checks that this replay reproduces
        # the original WAVs byte for byte
        return _sine_note(amp, phase, dur), {"kind": "sine"}
    if arm == "cd_silence":
        w = np.zeros(n, dtype=np.float64)
        k = max(1, int(SR_HZ * P.SILENCE_CLICK_MS / 1000.0))
        k = min(k, n)
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(k) / max(1, k - 1)))
        c = rng.standard_normal(k)
        c /= (np.max(np.abs(c)) + 1e-12)
        w[:k] = P.SILENCE_CLICK_PEAK * c * win
        return w, {"kind": "click", "peak": P.SILENCE_CLICK_PEAK}
    raise ValueError(arm)


def render_arm(arm: str, out_root: Path, limit: int | None = None) -> dict:
    """Replay the original temporal-proximity grid, substituting note content. Returns metadata."""
    base_iois_ms = CFG["base_ioi_ms_anchors"]
    ratios = CFG["ioi_ratios"]
    freq = CFG["freq_hz"]
    tone_dur = CFG["tone_duration_s"]
    seq_dur = CFG["sequence_duration_s"]
    n_inst = CFG["n_instances"]
    target_rms = CFG["target_rms"]

    src = json.loads(P.SRC_TEMP_META.read_text())
    src_idx = {e["file"]: e for e in src["stimuli"]}

    out_dir = out_root / arm / P.PARADIGM
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"_info": {
                "arm": arm, "paradigm": P.PARADIGM,
                "derived_from": str(P.SRC_TEMP_META),
                "description": "content replacement at original onsets",
                "root_seed": P.ROOT_SEED, "arm_id": P.ARM_ID.get(arm, 0),
                "generator": "content_replacement.py (replays the temporal loop of generate_stimuli.py)",
                "sample_rate": SR_HZ, "fade_ms": G.FADE_MS,
                "source_info": src["_info"]},
            "config": CFG, "stimuli": []}

    n_done = 0
    for bi, base_ms in enumerate(base_iois_ms):
        base_ioi = float(base_ms) / 1000.0
        for ri, ratio in enumerate(ratios):
            if not SR.temp_eligible(base_ioi, ratio):
                continue
            long_ioi = base_ioi * ratio
            trial_dur = 2 * base_ioi + long_ioi
            n_trials = max(1, int(seq_dur / trial_dur))
            for inst in range(n_inst):
                if limit is not None and n_done >= limit:
                    meta["_info"]["limited_to"] = limit
                    _write(meta, out_dir)
                    return meta
                seed = SR.seed_for("temp_proximity", bi, ri, 0, inst, G.SEED_BASE)
                orig_rng = np.random.RandomState(seed)          # original stream
                content_rng = np.random.default_rng(
                    [P.ROOT_SEED, P.ARM_ID.get(arm, 0), seed])    # new stream
                audio = np.zeros(int(SR_HZ * seq_dur), dtype=np.float64)
                onsets, ledger = [], []
                cursor = 0.0
                for _ in range(n_trials):
                    for k, dt in enumerate([0, base_ioi, base_ioi]):
                        cursor += dt if k > 0 else 0
                        if cursor + tone_dur > seq_dur:
                            break
                        amp = G.AMP_BASELINE + orig_rng.uniform(
                            -G.AMP_JITTER, G.AMP_JITTER)
                        phase = orig_rng.uniform(0, 2 * np.pi)
                        note, info = _render_note(arm, amp, phase, tone_dur,
                                                  content_rng)
                        idx = int(cursor * SR_HZ)
                        audio[idx:idx + len(note)] += note
                        onsets.append({"t": round(cursor, 4)})
                        info.update({"sample": idx, "amp": amp})
                        ledger.append(info)
                    cursor += long_ioi
                if arm != "cd_silence":
                    audio = G._rms_normalize(audio, target_rms)
                    peak = np.max(np.abs(audio))
                    if peak > 0.99:
                        audio = audio * (0.99 / peak)
                fname = SR.fname_temp(bi, ri, inst)
                G.atomic_sf_write(out_dir / fname, audio.astype(np.float32), SR_HZ)
                e = dict(src_idx[fname])
                e["onsets"] = onsets
                e["rms"] = round(G._rms(audio), 6)
                e["content_ledger"] = ledger
                e["arm"] = arm
                meta["stimuli"].append(e)
                n_done += 1
    _write(meta, out_dir)
    return meta


def _write(meta: dict, out_dir: Path) -> None:
    from extract_common import atomic_json_dump
    atomic_json_dump(out_dir / "metadata.json", meta)
    print(f"[content_replacement] {meta['_info']['arm']}: "
          f"{len(meta['stimuli'])} stimuli -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Checks: integrity and destruction effectiveness
# ─────────────────────────────────────────────────────────────────────────────

def _note_windows(entry: dict, n: int) -> list[tuple[int, int]]:
    d = int(SR_HZ * CFG["tone_duration_s"])
    return [(int(round(o["t"] * SR_HZ)), int(round(o["t"] * SR_HZ)) + d)
            for o in entry["onsets"] if int(round(o["t"] * SR_HZ)) + d <= n]


def _mean_pairwise_xcorr(x: np.ndarray, wins: list[tuple[int, int]],
                         max_notes: int = 6) -> float:
    """Mean over note pairs of max_lag normalised cross-correlation."""
    segs = [x[a:b] for a, b in wins[:max_notes]]
    segs = [s - s.mean() for s in segs if s.size and np.any(s)]
    vals = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            den = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if den <= 0:
                continue
            vals.append(float(np.max(np.abs(np.correlate(a, b, "full"))) / den))
    return float(np.mean(vals)) if vals else float("nan")


def checks(out_root: Path, sample_n: int = 200) -> dict:
    src = json.loads(P.SRC_TEMP_META.read_text())
    src_idx = {e["file"]: e for e in src["stimuli"]}
    rep: dict = {"integrity": {}, "destruction": {}, "status": "PASS", "failures": []}

    # reference: the original stimuli
    rng = np.random.default_rng(0)
    files = sorted(src_idx)
    pick = [files[i] for i in rng.choice(len(files), size=min(sample_n, len(files)),
                                         replace=False)]
    ref_x, ref_rms = [], []
    for f in pick:
        x, sr = sf.read(P.SRC_STIM / P.PARADIGM / f, dtype="float64")
        ref_x.append(_mean_pairwise_xcorr(x, _note_windows(src_idx[f], len(x))))
        ref_rms.append(float(np.sqrt(np.mean(x ** 2))))
    ref_xcorr = float(np.nanmedian(ref_x))
    ref_rms_m = float(np.median(ref_rms))
    rep["destruction"]["reference_original"] = {"median_pairwise_xcorr": ref_xcorr,
                                       "median_rms": ref_rms_m,
                                       "n_sampled": len(pick)}
    limits = {"cd_noise": 0.5, "cd_shuffle": 0.8}

    for arm in P.ARMS:
        d = out_root / arm / P.PARADIGM
        md_p = d / "metadata.json"
        if not md_p.exists():
            rep["failures"].append(f"{arm}: no metadata.json")
            continue
        md = json.loads(md_p.read_text())
        ents = {e["file"]: e for e in md["stimuli"]}
        g1 = {"n_stimuli": len(ents), "n_expected": len(src_idx),
              "onset_mismatches": [], "length_mismatches": [], "sr": set()}
        for f, e in ents.items():
            s = src_idx[f]
            if [o["t"] for o in e["onsets"]] != [o["t"] for o in s["onsets"]]:
                g1["onset_mismatches"].append(f)
        xs, rmss, nsamp = [], [], None
        for f in pick:
            if f not in ents:
                continue
            x, sr = sf.read(d / f, dtype="float64")
            g1["sr"].add(int(sr))
            xsrc, _ = sf.read(P.SRC_STIM / P.PARADIGM / f, dtype="float64")
            if len(x) != len(xsrc):
                g1["length_mismatches"].append(f)
            # sample-exact onset check: every note window of the source must
            # begin where the ledger says, and the ledger sample must equal
            # int(t*sr) of the source metadata
            for o, led in zip(src_idx[f]["onsets"], ents[f]["content_ledger"]):
                if abs(led["sample"] - int(round(o["t"] * SR_HZ))) > 1:
                    g1["onset_mismatches"].append(f"{f}@{o['t']}")
                    break
            xs.append(_mean_pairwise_xcorr(x, _note_windows(ents[f], len(x))))
            rmss.append(float(np.sqrt(np.mean(x ** 2))))
        g1["sr"] = sorted(g1["sr"])
        g1["ok"] = (g1["n_stimuli"] == g1["n_expected"]
                    and not g1["onset_mismatches"] and not g1["length_mismatches"]
                    and g1["sr"] == [SR_HZ])
        rep["integrity"][arm] = g1
        med_x = float(np.nanmedian(xs)) if xs else float("nan")
        med_r = float(np.median(rmss)) if rmss else float("nan")
        g2 = {"median_pairwise_xcorr": med_x,
              "xcorr_ratio_vs_original": med_x / ref_xcorr if ref_xcorr else None,
              "median_rms": med_r, "rms_ratio_vs_original": med_r / ref_rms_m}
        if arm == "cd_silence":
            g2["limit"] = {"rms_ratio_max": 1e-3}
            g2["ok"] = g2["rms_ratio_vs_original"] <= 1e-3
        else:
            g2["limit"] = {"xcorr_ratio_max": limits[arm]}
            g2["ok"] = g2["xcorr_ratio_vs_original"] <= limits[arm]
        rep["destruction"][arm] = g2
        if not g1["ok"]:
            rep["failures"].append(f"integrity {arm}")
        if not g2["ok"]:
            rep["failures"].append(f"destruction {arm}")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="content_selftest_") as d:
        rep["integrity"]["identity_replay"] = selftest(Path(d), 64)
    if not rep["integrity"]["identity_replay"]["ok"]:
        rep["failures"].append("integrity identity_replay")
    rep["status"] = "PASS" if not rep["failures"] else "FAIL"
    return rep


def selftest(tmp: Path, limit: int | None = None) -> dict:
    """Integrity core: replaying the original loop with sine content must reproduce the
    original WAV bit for bit. If it does, every arm is sample-aligned by
    construction (only `_render_note` differs between them)."""
    import hashlib
    meta = render_arm("_identity", tmp, limit)
    rep = {"n": 0, "identical": 0, "different": []}
    for e in meta["stimuli"]:
        a = (tmp / "_identity" / P.PARADIGM / e["file"]).read_bytes()
        b = (P.SRC_STIM / P.PARADIGM / e["file"]).read_bytes()
        rep["n"] += 1
        if hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest():
            rep["identical"] += 1
        else:
            rep["different"].append(e["file"])
    rep["ok"] = rep["n"] > 0 and not rep["different"]
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=P.ARMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="identity replay vs the original WAVs (byte for byte)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not os.environ.get("BCS_STIM_ROOT"):
        ap.error("set BCS_STIM_ROOT to the original stimulus root (stimuli/generate_stimuli.py)")
    if a.out is None and not a.selftest:
        ap.error("--out is required")
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="content_selftest_") as d:
            rep = selftest(Path(d), a.limit)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    if a.check:
        rep = checks(a.out)
        (a.out / "stimulus_checks.json").write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "integrity"}, indent=2))
        print(f"status={rep['status']} failures={rep['failures']}")
        return 0 if rep["status"] == "PASS" else 1
    arms = list(P.ARMS) if a.all else [a.arm]
    if not arms or arms == [None]:
        ap.error("need --arm, --all or --check")
    for arm in arms:
        render_arm(arm, a.out, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
