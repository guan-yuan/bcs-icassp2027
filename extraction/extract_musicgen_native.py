#!/usr/bin/env python3
"""MusicGen-S/M/L temporal-proximity features under audiocraft's native token schedule.

The main set's MusicGen features (extract_musicgen.py) use synchronized tokens: the four
codebooks aligned, no delay pattern, no start token. This extractor reads the same
checkpoints the way MusicGen itself is trained and run: audiocraft 1.3.0, the delay
pattern (delays 0,1,2,3) with the special start token, built by the checkpoint's own
pattern provider; teacher-forced on the clean EnCodec codes of the model's own
compression model; null text condition through LMModel.compute_predictions; forward
under the model's own autocast (fp16).

Captures: outputs of lm.transformer.layers[j] for the retained ids; the last retained
id is read at lm.out_norm's output (as the transformers `hidden_states[-1]` is the
final layer norm). Frame t <- delayed step s = t + 1, the state that has just consumed
codebook-0 frame t (codebook k up to frame t-k).

Differences from the synchronized features other than the schedule: audiocraft vs
transformers implementation; the null condition enters through cross-attention to
zeroed T5 states; codes come from audiocraft's float32 codec; fp16 autocast.

Usage
  python3 extraction/extract_musicgen_native.py --size small \
      --ckpt $HF_HOME/hub/models--facebook--musicgen-small/snapshots/<revision> \
      --stim-root <stimulus root> --out <new dir> [--sync-codes <codes npz>]
Scored by controls/native_schedule.py.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_common import load_metadata, load_wav_mono, stim_key, atomic_savez, set_deterministic_seeds

REV = {'small': '4c8334b02c6ec4e8664a91979669a501ec497792',
       'medium': 'd3bd7b00761b78ad7a8a05145ee31e7832e9916c',
       'large': '15ccdc92099879e47b6da12c350cdb71d4eab3ca'}
LAYERS = {'small': [0, 3, 6, 9, 12, 15, 18, 21, 23],
          'medium': [0, 6, 12, 18, 24, 30, 36, 42, 47],
          'large': [0, 6, 12, 18, 24, 30, 36, 42, 47]}
N_LAYERS = {'small': 24, 'medium': 48, 'large': 48}
DIM = {'small': 1024, 'medium': 1536, 'large': 2048}
SR = 32000
T_EXPECT = 150
FORWARD_VARIANT = 'audiocraft-delay-pattern-start-token-null-text'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fp16_storage(v):
    """Writer rule of extract_common.atomic_savez (BCS_NPZ_DTYPE=float16), per array."""
    h = v.astype(np.float16)
    if np.isfinite(v).all() and not np.isfinite(h).all():
        return v, True
    return h, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size', choices=tuple(REV), required=True)
    ap.add_argument('--ckpt', type=Path, required=True, help='snapshot dir at the pinned revision')
    ap.add_argument('--stim-root', type=Path, required=True)
    ap.add_argument('--sync-codes', type=Path,
                    help='optional: decoder codes npz of the synchronized features '
                         '(extract_musicgen.py); records how often the codes agree')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N clips only')
    a = ap.parse_args()
    if a.out.exists():
        ap.error('output directory exists')
    assert torch.cuda.is_available()
    assert a.ckpt.name == REV[a.size], (a.ckpt, REV[a.size])
    set_deterministic_seeds(42)
    import audiocraft
    from audiocraft.models import MusicGen
    from audiocraft.modules.conditioners import ConditioningAttributes
    from audiocraft.modules.codebooks_patterns import LayoutCoord
    import audiocraft.models.lm as lm_mod, audiocraft.modules.codebooks_patterns as pat_mod

    t_load = time.time()
    model = MusicGen.get_pretrained(str(a.ckpt), device='cuda')
    lm = model.lm.eval(); cm = model.compression_model.eval()
    load_s = time.time() - t_load
    K = lm.num_codebooks
    assert K == 4 and cm.num_codebooks == 4, (K, cm.num_codebooks)
    assert list(lm.pattern_provider.delays) == [0, 1, 2, 3], lm.pattern_provider.delays
    assert type(lm.pattern_provider).__name__ == 'DelayedPatternProvider'
    assert lm.pattern_provider.flatten_first == 0 and lm.pattern_provider.empty_initial == 0
    assert len(lm.transformer.layers) == N_LAYERS[a.size], len(lm.transformer.layers)
    assert lm.out_norm is not None
    assert cm.sample_rate == SR and abs(cm.frame_rate - 50) < 1e-9, (cm.sample_rate, cm.frame_rate)
    import collections
    dtypes = collections.Counter(f"{n.split('.')[0]}:{p.dtype}" for n, p in lm.named_parameters())
    assert all(p.dtype == torch.float16 for p in lm.transformer.parameters()), dtypes
    print('lm param dtypes (audiocraft loader)', dict(dtypes), flush=True)
    layers = LAYERS[a.size]; last = N_LAYERS[a.size] - 1
    assert layers[-1] == last

    cap = {}
    hooks = []
    for j in layers:
        if j == last:
            continue
        hooks.append(lm.transformer.layers[j].register_forward_hook(
            lambda m, i, o, j=j: cap.__setitem__(j, o.detach())))
    hooks.append(lm.out_norm.register_forward_hook(lambda m, i, o: cap.__setitem__(last, o.detach())))

    pattern = lm.pattern_provider.get_pattern(T_EXPECT)
    for t in range(T_EXPECT):
        assert LayoutCoord(t, 0) in pattern.layout[t + 1], t
        for k in range(1, K):
            assert LayoutCoord(t, k) in pattern.layout[t + 1 + k], (t, k)
    assert pattern.layout[0] == []

    sync = np.load(a.sync_codes) if a.sync_codes else None
    null = [ConditioningAttributes(text={'description': None})]
    meta = load_metadata(a.stim_root, 'temp_proximity')
    if a.limit:
        meta = meta[:a.limit]
    a.out.mkdir(parents=True)
    feats, codes_saved = {}, {}
    ce_sum = np.zeros(K); ce_n = np.zeros(K)
    agree = 0; mism = []; kept32 = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i, item in enumerate(meta):
        stem = stim_key(item)
        wav = load_wav_mono(a.stim_root / 'temp_proximity' / item['file'], SR, as_tensor=True)
        wav = wav.unsqueeze(0).to('cuda')                       # [1,1,N]
        with torch.no_grad():
            codes, scale = cm.encode(wav)
            assert scale is None
            assert codes.shape == (1, K, T_EXPECT), codes.shape
            c_np = codes[0].cpu().numpy()
            if sync is not None:
                s_np = sync[stem]
                assert s_np.shape == c_np.shape, (s_np.shape, c_np.shape)
                if np.array_equal(s_np, c_np):
                    agree += 1
                else:
                    mism.append([stem, int((s_np != c_np).sum())])
            # schedule check on this clip (the sequence compute_predictions builds)
            seq, _, _ = pattern.build_pattern_sequence(codes, lm.special_token_id, keep_only_valid_steps=True)
            assert seq.shape[-1] >= T_EXPECT + 1
            assert bool((seq[0, :, 0] == lm.special_token_id).all())
            assert torch.equal(seq[0, 0, 1:T_EXPECT + 1], codes[0, 0])
            for k in range(1, K):
                assert torch.equal(seq[0, k, 1 + k:T_EXPECT + 1], codes[0, k, :T_EXPECT - k])
                assert bool((seq[0, k, 1:1 + k] == lm.special_token_id).all())
            cap.clear()
            with model.autocast:
                out = lm.compute_predictions(codes, null)
            S = seq.shape[-1]
            logits = out.logits.float(); mask = out.mask
            for k in range(K):
                m = mask[0, k]
                ce = F.cross_entropy(logits[0, k][m], codes[0, k][m].long(), reduction='sum')
                ce_sum[k] += float(ce); ce_n[k] += int(m.sum())
            stack = []
            for j in layers:
                h = cap[j]
                assert h.shape == (1, S, DIM[a.size]), (j, h.shape)
                stack.append(h[0, 1:T_EXPECT + 1].float().cpu().numpy())
        arr = np.stack(stack)                                   # [L, T, D]
        if not np.isfinite(arr).all():
            raise ValueError(f'nonfinite features {stem}')
        arr, k32 = fp16_storage(arr); kept32 += int(k32)
        if stem in feats:
            raise ValueError('duplicate stem')
        feats[stem] = arr
        codes_saved[stem] = c_np.astype(np.int16)
        if (i + 1) % 50 == 0 or i + 1 == len(meta):
            el = time.time() - t0
            print(f'{a.size} {i+1}/{len(meta)} {el:.1f}s {el/(i+1):.3f}s/clip peak_mem_GB='
                  f'{torch.cuda.max_memory_allocated()/2**30:.2f}', flush=True)
    for h in hooks:
        h.remove()
    fwd_s = time.time() - t0
    prov = dict(model_name=f'musicgen_{a.size}', hf_id=f'facebook/musicgen-{a.size}', hf_revision=REV[a.size],
                forward_variant=FORWARD_VARIANT, audiocraft=audiocraft.__version__, torch=torch.__version__,
                lm_py_sha256=sha(lm_mod.__file__), patterns_py_sha256=sha(pat_mod.__file__),
                state_dict_sha256=sha(a.ckpt / 'state_dict.bin'),
                compression_state_dict_sha256=sha(a.ckpt / 'compression_state_dict.bin'),
                output_layers=layers, last_layer_read_at='lm.out_norm',
                frame_mapping='frame t <- delayed step s=t+1 (after start token); codebook k at s holds frame s-1-k',
                delays=list(lm.pattern_provider.delays), special_token_id=int(lm.special_token_id),
                conditioning="ConditioningAttributes(text={'description': None}) via lm.compute_predictions",
                autocast='model.autocast (cuda float16)', lm_param_dtypes=dict(dtypes),
                storage='float16 (writer rule: overflow arrays kept float32)', seed=42,
                partial=bool(a.limit), n_clips=len(meta))
    atomic_savez(a.out / f'musicgen_{a.size}_temp_proximity.npz', _meta=prov, **feats)
    np.savez_compressed(a.out / f'musicgen_{a.size}_temp_proximity_codes.npz', **codes_saved)
    diag = dict(size=a.size, n_clips=len(meta), code_agreement_with_sync=(agree if sync is not None else None), code_mismatch=mism[:50],
                n_code_mismatch=len(mism), teacher_forced_ce_nats=(ce_sum / ce_n).tolist(),
                ce_valid_tokens=ce_n.tolist(), uniform_ce=float(np.log(2048)),
                kept_float32_arrays=kept32, peak_gpu_mem_GB=torch.cuda.max_memory_allocated() / 2**30,
                load_s=load_s, forward_s=fwd_s, s_per_clip=fwd_s / len(meta),
                feature_shape=list(next(iter(feats.values())).shape),
                feature_dtype=str(next(iter(feats.values())).dtype))
    (a.out / 'diag.json').write_text(json.dumps(diag, indent=2) + '\n')
    print(json.dumps({k: v for k, v in diag.items() if k != 'code_mismatch'}), flush=True)


if __name__ == '__main__':
    main()
