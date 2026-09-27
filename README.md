# Input calibration and a specificity audit of ASA cue scores

Code and results for the paper *Input Calibration and a Specificity Audit of
Auditory-Scene-Analysis Cue Scores in Music Foundation Models*.

- `bcs/`, `controls/`: the cue score (BCS) and its controls.
- `results/`: the result files behind every result number in the paper.
- `make_paper_numbers.py`: recomputes the paper's numbers from `results/` on a CPU.
- `config/`: the 16 checkpoints (Hugging Face id, pinned revision, captured
  layers, front-end reference) and the stimulus settings.
- `stimuli/`: the stimulus generator.
- `extraction/`: one feature-extraction script per model family and front-end.

## Recompute the paper's numbers

```
python3 make_paper_numbers.py            # result macros, Tables 1 and 3, Fig. 2, the MusicGen case study
python3 make_paper_numbers.py --macros   # macros only: name, value, where in the paper
```

Standard library only. The script checks the result files for consistency
(for example, every stored flag against its interval) and exits with status 1
if a check fails.

| Paper item | Result file(s) |
|---|---|
| Table 1, Fig. 2, Sec. 3.1 | `primary_map.json`, `static_tables.json`, `offsets.json`, `offset_counts.json` |
| Table 2 frame rates; Sec. 3.1 rate re-score | `frame_rates.json`; `rate_rescore.json`, `rate_common12.json` |
| Shared trajectory, all-zero input, first-block input | `fixed_trajectory.json`, `zero_waveform.json`, `first_block_input.json` |
| Model-free clock (range 0.989-1.000) | `clock_reference.json`, `frame_rates.json` |
| Sec. 3.3 MusicGen case study | `primary_map.json`, `offsets.json`, `offset_counts.json`, `musicgen_native_schedule.json`, `index_selectivity_native.json`, `index_selectivity.json`, `content_replacement.json`, `fixed_trajectory.json`, `zero_waveform.json`, `first_block_input.json` |
| Sec. 3.3 size and random-decoder controls | `rerender_random.json`, `rerender_random_intervals.json`, `rerender_size_contrast.json` |
| Table 3 | `index_selectivity.json`, `content_replacement.json` |
| Coarse frame grid (dagger in Fig. 2 and Table 3) | `frame_diagnostic.json` |
| Stimulus inventory (Sec. 2.1) | `static_tables.json` |

In `primary_map.json` keys are `<read-out>|<cue>`. Cues: `freq_proximity`, `temp_proximity`,
`harmonicity`, `onset_sync`. Front-end keys start with `fe_`: the eight
learned codec/VAE outputs and `fe_pupujepa_logmel` (PupuM2D-L's
log-spectrogram); `cqt`, `gammatone` and `logmel` are the fixed DSP
transforms. Content-replacement arms: `cd_noise`, `cd_shuffle`, `cd_silence`
(clicks).

## How the score is computed

1. Each stimulus is passed through a checkpoint or front-end, and the hidden
   states of the captured layers are stored as `[layers, frames, dim]`.
2. Per cue, a per-stimulus statistic is computed from cosine distances
   (`bcs.score.build_score_table`).
3. The Spearman correlation with the rendered cue value is computed within
   each anchor and averaged over anchors.
4. The exact area under the depth profile gives the score; controls use the
   nine-point summary rho_9.
5. Delta_E is the checkpoint score minus its front-end score. Intervals come
   from a two-stage cluster bootstrap (10,000 draws, seed 42), shared by both
   sides of each contrast.

## Regenerate stimuli and features

```
pip install -r requirements.txt
python3 bcs/score.py --help    # how to score one feature file
python3 stimuli/generate_stimuli.py --all --out <stimulus root>
python3 stimuli/generate_rerender.py --out <re-rendered root>   # Sec. 3.3 re-render controls
```

Checkpoint feature extraction needs a GPU, the model weights at the
revisions pinned in `config/checkpoints.yaml`, and each model family's own
environment (versions listed in `requirements.txt`). `--help` on each
extractor lists its arguments; the extractors built on
`extraction/grid_common.py` and `extraction/frontend_common.py` read the
stimulus root from `BCS_STIM_ROOT` (required), the others from `--stim_root`
or `--stim-root`, and DiffRhythm, PupuM2D-L (PupuJEPA code)
and Magenta-RT2 also need a clone of their code (`DIFFRHYTHM_DIR`,
`PUPUJEPA_SRC`, `MAGENTA_RT_SRC`).

## License

All rights reserved (see `LICENSE`).
