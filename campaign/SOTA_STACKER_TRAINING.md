# SOTA HydroRecurrentStacker — Training & Tuning Guide

Reproduces the project's current SOTA on `Combined_IARA_Deepship_1s`:
**test macroF1 ≈ 0.65 raw / 0.70+ with per-class iso + MSP-iso selective**.

## Pipeline overview

```
WAVs (1s @ 5120 Hz)
  │
  ▼
[6-ckpt ensemble probs cache]   ← campaign/probs_combined_recurrent_stacker.npz
       (5 Hydra + complete-044 cargo_confirm pool)
  │
  ▼
campaign/train_recurrent_stacker.py
  - reads cache (frozen ensemble) + audio
  - HydroRecurrentStacker = _ClipEncoder ⊕ ensemble token-proj → multi-scale GRU
  - GamblerCE + deep-gambler abstain logit
  ▼
lightning_logs/<out_dir>/best.pt
  │
  ▼
campaign/eval_recurrent_stacker.py
  - raw test macroF1 ≈ 0.65
  - MSP-isotonic selective@cov0.85 → 0.715
  │
  ▼ (optional, this session's add-on)
campaign/calibrate_sota_stacker.py
  - per-class isotonic + MSP-isotonic selective → 0.7046 raw test
```

## Shared CLI surface

Both `train_recurrent_stacker.py` and `autotune_recurrent.py` now accept the
same data/model knobs:

| Arg            | Purpose                                                         | Default |
|---|---|---|
| `--data_dir`   | Dataset root with `Train/Val/Test` class folders               | `Combined_IARA_Deepship_1s` |
| `--loader`     | One of `dali`, `dali_split`, `threaded`, `threaded_split`. Stacker uses internal threaded I/O; DALI variants are accepted for uniformity but treated as `threaded`. | `threaded` |
| `--class_depth`| Model depth knob — scales `gru_layers` and repeats `tcn_dilations`. 1=baseline, 2/3=deeper. | `1` |
| `--ens_npz`    | Frozen ensemble probability cache (only train script).         | `campaign/probs_combined_recurrent_stacker.npz` |

The dataset/loader layer is **not** parameterised by `class_depth` —
classes are read from the folders under `data_dir/Train/`. The stacker is
hard-coded to 4 classes (Cargo / Passenger / Tanker / Tug).

## Training command

```
python -m campaign.train_recurrent_stacker \
  --data_dir /var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s \
  --loader threaded --class_depth 1 \
  --ens_npz campaign/probs_combined_recurrent_stacker.npz \
  --out_dir lightning_logs/hydro_recurrent_stacker_combined \
  --steps 8000 --K_train 30 --K_eval 60 \
  --scales 1,3,10 --gambler_w 0.1 --abstain_o 2.2 \
  --focal_gamma 2.0 --deep_aux_w 0.2 --lr 2e-3
```

Wall ~3 h on RTX 5090. Saves `best.pt` keyed on val macro-F1, ~39k params at
`class_depth=1`.

## Autotune command

```
python -m campaign.autotune_recurrent \
  --data_dir /var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s \
  --loader threaded --class_depth 1 \
  --study_name hydro_recurrent_autotune_v1 \
  --hours 12 --steps 2500 --val_every 250 --K_eval 48
```

Optuna TPE study, sqlite-backed. The harness uses every CLI knob above to
construct trials; per-trial hyperparams (lr, dropout, etc.) are sampled by
the TPE sampler with `startup_trials=8` random warmup trials.

> Important: per CLAUDE.md, the autotune run hit val 0.6321 → test 0.540 with
> a big val→test gap — overfit search. Prefer hand-tuned schedules + the
> calibration stack below over autotune for this dataset.

## Eval + calibration

After training, run:

```
python -m campaign.eval_recurrent_stacker \
  --data_dir <data> \
  --ens_npz campaign/probs_combined_recurrent_stacker.npz \
  --ckpt lightning_logs/hydro_recurrent_stacker_combined/best.pt
```

Then for the extra (per-class iso + temperature + iso+selective@cov0.85):

```
python -m campaign.calibrate_sota_stacker \
  --data_dir <data> --ckpt <best.pt> --K_eval 60
```

Expected on combined: **per-class iso + selective@cov0.85 → test macroF1 ≈ 0.704**
(this session's confirmed run, cov 0.89).

## Class-depth notes

| class_depth | gru_layers | tcn_dilations               | params (approx) |
|---|---|---|---|
| 1 (default) | 1          | (1, 4, 16, 64)              | 39k             |
| 2           | 2          | (1, 4, 16, 64, 1, 4, 16, 64)| ~70k            |
| 3           | 3          | (1, 4, 16, 64) × 3          | ~110k           |

Depth-1 already saturates this dataset. Use depth-2/3 only as an ablation —
the SOTA result is at depth-1.

## Loader notes

The stacker reads waveforms via `read_batch_threaded` (in `campaign/ast_common`),
not the DALI / threaded DataModules wired in `data/loader_factory.py`. The
`--loader` arg is accepted across all four IDs for symmetry with
`train_moe.py` / `moe_sweep.py`, but `dali*` choices are logged as "recorded
only" and execution falls through to the internal threaded reader.

If you need DALI semantics for the stacker, train a HydroComplete /
HydroHydra base first (which does respect `--loader`), dump per-clip probs
into a new cache npz, and re-run the stacker on top of that.
