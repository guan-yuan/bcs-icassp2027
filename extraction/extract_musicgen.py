#!/usr/bin/env python3
"""MusicGen-S/M/L: captured decoder layers and the first-block input.

Forward: teacher-forced clean EnCodec tokens through the token-id interface
(`decoder(input_ids=...)`), null text, the four codebooks synchronised with no
delay pattern and no start-token shift. transformers 4.44.0, unmodified.
(The pre-computed-embedding route of this transformers version passes the
positional module a [B, T, 1] tensor and applies the position-zero embedding
at every frame; the token-id route used here keeps the full time axis, which
position_check() checks on every clip.)

Outputs per cue: <key>_<cue>.npz (captured layers), token_embedding_<key>_<cue>.npz
(summed token embeddings, before position encoding), first_block_input_<key>_<cue>.npz
(the tensor the first block receives) and the integer decoder codes.
--init-seed replaces the decoder by a randomly initialised one of the same
configuration while the pretrained EnCodec tokenizer is kept.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_common import (atomic_savez,build_metadata,clamp_layers,load_config,
                           load_metadata,load_wav_mono,require_gpu,set_deterministic_seeds,stim_key)


# ----------------------------------------------------------- token-id forward
def decoder_ids(decoder, audio_codes):
    if audio_codes.ndim != 3:
        raise ValueError('EnCodec codes must be [batch,codebook,time]')
    batch,codebooks,time = audio_codes.shape
    required = decoder.num_codebooks
    if batch < 1 or time < 1 or codebooks < required:
        raise ValueError('Empty audio codes or insufficient configured codebooks')
    if audio_codes.dtype != torch.long:
        raise ValueError('MusicGen token ids require torch.long')
    # Use the first num_codebooks tokenizer codebooks; extra
    # tokenizer codebooks are not decoder inputs for this model configuration.
    return audio_codes[:,:required,:].contiguous().reshape(batch*required,time)


def forward_from_codes(decoder, audio_codes, **kwargs):
    return decoder(input_ids=decoder_ids(decoder,audio_codes), **kwargs)


# ------------------------------------------------------- random decoder control
def reinitialize_decoder(model, seed):
    """Replace only the complete decoder; keep the pretrained audio encoder.

    Call before model dtype/device conversion. The token-id forward is applied
    identically to trained and reinitialised models.
    """
    from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
    old = model.decoder
    config = copy.deepcopy(old.config)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    replacement = MusicgenForCausalLM(config)
    if len(old.model.decoder.layers) != len(replacement.model.decoder.layers):
        raise ValueError('Reinitialized decoder changed the configured depth')
    old_weight = old.model.decoder.layers[0].self_attn.q_proj.weight.detach()
    new_weight = replacement.model.decoder.layers[0].self_attn.q_proj.weight.detach()
    if torch.equal(old_weight, new_weight):
        raise ValueError('Reinitialization did not change the first block weights')
    model.decoder = replacement
    return model


# ------------------------------------------------------------------ extraction

FORWARD_VARIANT='token-ids-with-positions'


def position_check(decoder,codes,hidden):
    k=decoder.num_codebooks;selected=codes[:,:k,:]
    pre=sum(decoder.embed_tokens[j](selected[:,j]) for j in range(k))
    positions=decoder.embed_positions(selected)
    if positions.shape!=(codes.shape[-1],pre.shape[-1]):
        raise ValueError('Position lookup did not retain full time axis')
    if codes.shape[-1]>1 and torch.equal(positions[0],positions[1]):
        raise ValueError('Position vector is constant across initial time steps')
    expected=pre+positions.to(pre.device)
    if not torch.equal(expected,hidden):
        raise ValueError('First-block input differs from token embedding + position tensor')
    return pre


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant',choices=('small','medium','large'),required=True)
    ap.add_argument('--stim-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--paradigms',nargs='+',required=True)
    ap.add_argument('--limit',type=int,default=0,help='first N stimuli per cue only')
    ap.add_argument('--init-seed',type=int,
                    help='Random-decoder control seed; tokenizer remains pretrained')
    a=ap.parse_args()
    if a.limit<0:ap.error('--limit must be >= 0')
    if a.out.exists():ap.error('Output directory exists; choose a new one')
    require_gpu();cfg=load_config();base_key='musicgen_'+a.variant;mcfg=cfg['models'][base_key]
    key=base_key if a.init_seed is None else f'{base_key}_randinit_s{a.init_seed}'
    seed=int(cfg['random_seeds']['stimuli_base']);set_deterministic_seeds(seed)
    use_fp16=mcfg.get('dtype_override','float16')=='float16'
    from transformers import MusicgenForConditionalGeneration,__version__
    from transformers.models.musicgen import modeling_musicgen
    model=MusicgenForConditionalGeneration.from_pretrained(mcfg['hf_id'],revision=mcfg['hf_revision'],local_files_only=True)
    if a.init_seed is not None:
        model=reinitialize_decoder(model,a.init_seed)
    if use_fp16:model=model.to(dtype=torch.float16)
    model=model.to('cuda').eval();decoder=model.decoder.model.decoder
    layers=clamp_layers(list(mcfg['layers_sample']),len(decoder.layers),key)
    bandwidth=model.audio_encoder.config.target_bandwidths[-1]
    a.out.mkdir(parents=True)
    library=Path(modeling_musicgen.__file__)
    prov=build_metadata(cfg,base_key)
    prov.update(forward_variant=FORWARD_VARIANT,transformers=__version__,
                decoder_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                max_position_embeddings=decoder.config.max_position_embeddings,
                selected_codebooks=decoder.num_codebooks,output_layers=layers,
                output_model_key=key,initialization=('pretrained' if a.init_seed is None else 'random_decoder'),
                init_seed=a.init_seed,pretrained_tokenizer_retained=True)
    reports={}
    for cue in a.paradigms:
        meta=load_metadata(a.stim_root,cue)
        if a.limit:meta=meta[:a.limit]
        if not meta:raise ValueError('Empty selected stimulus roster')
        hs_saved={};pre_saved={};post_saved={};codes_saved={}
        for i,item in enumerate(meta):
            audio=load_wav_mono(a.stim_root/cue/item['file'],int(mcfg['sample_rate']),as_tensor=True).to('cuda')
            audio=audio.unsqueeze(0)
            if use_fp16:audio=audio.to(dtype=torch.float16)
            with torch.no_grad():
                codes=model.audio_encoder.encode(audio,bandwidth=bandwidth).audio_codes
                if codes.ndim==4:codes=codes.squeeze(0)
                ids=decoder_ids(decoder,codes)
                result=forward_from_codes(decoder,codes,output_hidden_states=True,return_dict=True)
                hs=result.hidden_states
                pre=position_check(decoder,codes,hs[0])
            stem=stim_key(item)
            if stem in hs_saved:raise ValueError('Duplicate stimulus identity')
            hs_saved[stem]=np.stack([hs[j+1].squeeze(0).detach().float().cpu().numpy() for j in layers])
            pre_saved[stem]=pre.detach().float().cpu().numpy()
            post_saved[stem]=hs[0].detach().float().cpu().numpy()
            codes_saved[stem]=ids.detach().cpu().numpy()
            if any(not np.isfinite(v[stem]).all() for v in (hs_saved,pre_saved,post_saved)):
                raise ValueError('Nonfinite produced representation')
            if (i+1)%100==0 or i+1==len(meta):print(key,cue,i+1,'/',len(meta),flush=True)
        atomic_savez(a.out/f'{key}_{cue}.npz',_meta=prov,**hs_saved)
        atomic_savez(a.out/f'token_embedding_{key}_{cue}.npz',_meta={**prov,'stage':'pre_positional'},**pre_saved)
        atomic_savez(a.out/f'first_block_input_{key}_{cue}.npz',_meta={**prov,'stage':'first_block_input'},**post_saved)
        np.savez_compressed(a.out/f'{key}_{cue}_decoder_codes.npz',**codes_saved)
        reports[cue]=dict(stimuli=len(meta),position_checks=len(meta),code_arrays=len(codes_saved))
    (a.out/'position_check.json').write_text(json.dumps(dict(forward_variant=FORWARD_VARIANT,partial=bool(a.limit),variant=a.variant,library_sha256=prov['decoder_library_sha256'],cues=reports),indent=2)+'\n')
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
