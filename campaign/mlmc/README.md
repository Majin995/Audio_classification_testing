# MLMC — multi-label / multi-class variants of the top-5 models

Alternate training / inference scripts for the five performant setups in the
project TL;DR, each adapted to predict **independently per class** and emit a
**one-hot / multi-hot** prediction (0/1 vector of length `num_classes`) instead
of a single argmax label.

The single-label datasets in this repo are the degenerate multi-label case
(exactly one positive bit per source), so these scripts run on the existing
data unchanged. To use genuine multi-label ground truth, pass a
`labels_map = {abs_path: [class, ...]}` to the dataloader.

## Self-splitting dataloader

`data/mlmc_windowed_loader.py` — `MLMCWindowedDataModule`. Point it at a folder
of arbitrary-length recordings (`<split>/<class>/*.wav`); it windows each file
into fixed `window_sec` clips **itself** (header-only enumeration, lazy
per-window reads) and yields `(wav[fixed_len], multihot[num_classes])`.
Preprocessing is bit-identical to `ExtendedThreadedAudioDataModule`.

Shared inference helpers (one honest path for every script):
`tune_thresholds_per_class` (val-only per-class F1-max threshold),
`probs_to_onehot` (`force_one` keeps rows non-empty → valid one-hot fallback),
`aggregate_source_logmean`, `multilabel_report` (macro/micro-F1, subset-acc,
Hamming).

## The five scripts

| # | Script | Setup | Multi-label change |
|---|--------|-------|--------------------|
| ① | `cargo_confirm_5base_mlmc.py` | 5-Hydra base + complete-044 voucher | per-class thresholds; Cargo bit keeps the voucher-confirm OR-rule |
| ② | `zero_fit_ensemble_mlmc.py` | zero-fit log-mean (top-5 by val_p) | per-class thresholds → multi-hot |
| ③ | `rapid_lr_stacker_mlmc.py` | rapid_1s 5-ckpt LR stacker | `OneVsRestClassifier(LogisticRegression)` |
| ④ | `train_ast2_mlmc.py` | AST-10s-B fine-tune | sigmoid head + `BCEWithLogitsLoss(pos_weight)` |
| ⑤ | `train_recurrent_stacker_mlmc.py` | HydroRecurrentStacker | `gambler=False` (4 logits) + BCE deep supervision |

`mlmc_common.py` holds the cached-prob loading + source grouping shared by ①②.

## Honest contract (unchanged)

Per-class thresholds / τ / epoch are selected on **val** (or an independent
train split for ③); the **test** split is touched exactly once. ①②③ reuse the
existing cached per-clip softmax dumps; ④⑤ train fresh.

## Single-label mode (`--single_label`)

Every script accepts `--single_label`. The data in this repo is single-label
(one class per source), so by default the multi-label decision rule can still
emit 2 bits on the rare ambiguous source. Pass `--single_label` for the **strict
one-hot** decision — pure argmax, exactly one bit per row — which faithfully
reproduces the original argmax models. In this mode ① reproduces the
single-label winner (test macroF1 ≈ 0.804) and the voucher *replaces* the
predicted class with Cargo (the original `np.where(mask, CARGO, ŷ)`), rather
than OR-adding a Cargo bit. Omit the flag for true multi-label / multi-hot
output. The shared `probs_to_onehot(..., single_label=True)` is the single
switch behind all of them.

## Running (from the repo root)

```bash
# ① ② — cached-prob ensembles, CPU, seconds
python campaign/mlmc/cargo_confirm_5base_mlmc.py
python campaign/mlmc/zero_fit_ensemble_mlmc.py

# ③ — LR stacker on rapid cached probs
python campaign/mlmc/rapid_lr_stacker_mlmc.py

# ④ — AST multi-label fine-tune (GPU)
python campaign/mlmc/train_ast2_mlmc.py \
    --data_root /var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_10s \
    --clip_len 160000 --src_sr 16000 --loss focal --class_weight --specaug \
    --mixup 0 --lr_head 3e-4 --seed 2024 --out lightning_logs/mlmc/ast10b

# ⑤ — recurrent stacker multi-label (GPU; needs the ensemble NPZ)
python campaign/mlmc/train_recurrent_stacker_mlmc.py --steps 8000 \
    --K_train 30 --K_eval 60 --scales 1,3,10 --lr 2e-3 \
    --out_dir lightning_logs/mlmc/recurrent_stacker
```

Each run writes `test_onehot.npy` / `val_onehot.npy` (the one-hot/multi-hot
predictions), the matching `*_y_multihot.npy`, `thresholds.json`, and
`metrics.json` under its `--out` / `--out_dir`.
