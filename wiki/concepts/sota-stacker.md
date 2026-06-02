# SOTA Stacker — HydroRecurrentStacker

Project's current SOTA on `Combined_IARA_Deepship_1s`.

## Numbers
- Raw test macroF1: **0.6515**
- MSP-isotonic selective @ cov0.85: **0.715**
- Per-class iso + MSP-iso selective @ cov0.85 (this session): **0.7046** (cov 0.89)

## Per-class @ best variant (per-class iso + selective)
Cargo 0.697 · Passenger 0.864 · Tanker 0.746 · Tug 0.512

## Pipeline
1. Cached 6-ckpt cargo_confirm ensemble probs (`campaign/probs_combined_recurrent_stacker.npz`) — 5 Hydra + complete-044.
2. `HydroRecurrentStacker` (39k params at `class_depth=1`): `_ClipEncoder` ⊕ frozen ensemble token-proj → multi-scale GRU (1/3/10 s) + deep-gambler abstain logit.
3. Loss: GamblerCE (focal CE + 0.1·deep-gambler aux, `abstain_o=2.2`, `deep_aux_w=0.2`).
4. Eval applies MSP-isotonic on val → cut at cov target → selective test.
5. Optional add-on: per-class isotonic on val before selective.

## Scripts
- Train: `campaign/train_recurrent_stacker.py`
- Tune: `campaign/autotune_recurrent.py` (Optuna TPE)
- Eval (MSP-iso selective): `campaign/eval_recurrent_stacker.py`
- Extra calibration (per-class iso + temperature + iso+selective): `campaign/calibrate_sota_stacker.py` ← added this session

## Shared CLI surface
Both train + tune scripts accept:
- `--data_dir` — dataset root
- `--loader {dali|dali_split|threaded|threaded_split}` — for parity with `train_moe.py`; stacker uses internal threaded I/O, DALI variants are recorded only
- `--class_depth N` — model knob; scales `gru_layers` (=max(default, N)) and repeats `tcn_dilations` N times. Default 1.

Loaders/datasets are **not** parameterised by class_depth.

## Why MoE on top failed
See `campaign/MOE_SWEEP_FINDINGS.md`. Binary OvR experts are not calibrated to the gate's softmax; product-of-experts amplifies the mismatch. Multi-mode experts only added +0.007 over their own weak gate.

## Full guide
`campaign/SOTA_STACKER_TRAINING.md`
