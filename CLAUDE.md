# CLAUDE.md — Audio Classification Testing

Project-specific guidance for Claude Code working in this repo.

---

## TL;DR — Performant models (honest holdout)

| Rank | Setup | Dataset | Test F1 | Test macroP | Script |
|---|---|---|---|---|---|
| ① | **cargo_confirm_5base** (5 Hydra + complete-044 voucher, τ=0.4) | Classifier_Dataset (n=83) | **0.8043** | **0.8911** | `campaign/cargo_confirm_5base.py` |
| ② | zero-fit log-mean (baseline+069+042, 5 Hydra ckpts) | Classifier_Dataset | 0.7895 | 0.9060 | `campaign/zero_fit_ensemble.py` |
| ③ | 5-ckpt LR stacker on rapid (val n=64) | Classification_rapid_testing_1s | **0.8658** | **0.8862** | `lightning_logs/rapid_1s_honest_winner/` |
| ④ | **AST-10s-B** (focal+class-wt+SpecAugment, ep8) | Combined IARA+Deepship 1s (n=236) | **0.680** | — | `lightning_logs/ast10b_combined/` |
| ⑤ | HydroRecurrentStacker + MSP-isotonic gate (val cov 0.85) | Combined IARA+Deepship (n=197 kept) | **0.715** | 0.716 | `campaign/eval_recurrent_stacker.py` |

**Honest F1 ceiling on Classifier_Dataset = 0.8043** (5 stuck Cargo + 1 stuck
Tug confused by every base ckpt — no stacker reweighting can break this).

**Honest F1 ceiling on Combined IARA+Deepship ≈ 0.70-0.75** — Tug covariate
shift in the holdout (train = small harbor tugs median 30 m; test = large
offshore supply vessels median 81 m). Full writeup: `campaign/SOTA_REPORT.md`.

### Models that DIDN'T pay off (kept here so we don't redo them)

- **cargo_confirm (precise-013 voucher, τ=0.55)** — test F1 0.7913, beaten by ①.
- **HydroSpark / HydroSetSpark cross-ensemble** — Deepship_1s test F1 0.659; Tug bottleneck, not architecture.
- **cargo_confirm_5base zero-shot to IARA** — IARA Option 1 test F1 0.229 (vs 0.804 in-domain). Cross-domain fine-tune is required; τ tuning does not help.
- **AST ensemble blend** (AST-10s×2 + AST-1s + complete_dombal) — 0.674, worse than the single AST-10s-B (0.680). Ensemble hurts here.
- **HydroRecurrent** (single-scale GRU, 35k) — test F1 0.446. Useful only as a sequential-vs-bag baseline.
- **HydroRecurrentMS** (multi-scale GRU, 35k) — test F1 0.455. +0.07 IARA over single-scale but still below all stackers.
- **HydroRecurrentStacker raw** (no gate) — test F1 0.653; needs the MSP-isotonic gate (⑤) to be competitive.
- **HydroRecurrent autotuned** (Optuna 1449 trials, 12h) — val 0.6321 → test 0.540. Big val→test gap from search overfit; built-in abstain gate gave zero selective lift.
- **HydroRecurrentStacker without IARA Glider partitions (F+G)** — −0.015 test F1; Glider removal HURTS, not helps. Don't redo.
- **rich_stacker_v1 / v3 / LR-bagged / prior-corrected / n_clips-feature** — all worse on test than zero-fit log-mean. Fitted stackers overfit val on this ckpt pool.
- **Per-clip temperature calibration under log-mean** — softer probs erase per-clip signal, hurts source-level macro-F1.
- **`class_weight='balanced'` on val** — overpredicts Tug because val:test Tug ratio differs (6.5% vs 2.4%).
- **HydroComplete-v2 + Optuna sweep** (auto_tune/, 11 trials) — clip-level val plateau ~0.61, no higher than original HydroComplete trunk; multi-scale mel + freq-attn only +0.7pp on top of `adamw + deep_ln`.

---

## How the top model works — cargo_confirm_5base

```
                  Classifier_Dataset source (≥1 clips @ 5120 Hz)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
  BASE POOL (5 Hydra ckpts, all 1D)              VOUCHER (HydroComplete-044)
   hydra-{026,032,031,069,042}                    Gabor⊕Scat⊕Sinc⊕TDSBE⊕CQT⊕DEMON
        │ per-clip softmax                                  │ per-clip softmax
        ▼ log-mean over clips → P_base (4)                 ▼ log-mean → P_voucher (4)
        ▼ argmax → ŷ_base                                  ▼ argmax → ŷ_voucher
                          ╲                                ╱
                            ▼  Cargo-confirm rule (τ=0.4 on val OOF)
                  if  ŷ_voucher = Cargo  AND  P_base[Cargo] ≥ τ:  ŷ = Cargo
                  else:                                              ŷ = ŷ_base
```

HydroComplete-044 has high Cargo recall (11/16) but only 50% Cargo precision.
The 5-Hydra pool has high overall precision but unanimously misses 7 Cargo→Tanker.
The voucher rule lets HydroComplete "vote in" Cargo **only** when the base
pool already gives Cargo ≥ τ — recovers 3 Cargos without paying the FPs.

### HydroComplete (the voucher ckpt)

Branches: ① LearnableGabor 1D · ② Kymatio Scattering1D · ③ SincNet bandpass ·
④ TDSBE subband envelope · ⑤ DEMON env → 1D CNN → GRU · ⑥ CQT → 2D CNN →
freq-collapse → cross-attention fuser → attentive statistics pool → margin head
(+ gambler abstain). Loss: `LargeMarginFocalLoss(γ=2.0, m=0.3, smoothing=0.05)
+ 0.1·gambler_aux`. Files: `models/hydro_complete.py`; cache dump via
`campaign/dump_complete_aligned.py`.

### HydroHydra (5 base ckpts) — strictly 1D, no STFT/Mel/CQT

A. LearnableGabor · B. Scattering1D · C. SincNet · D. TDSBE → concat → 1D
backbone → attentive stats pool → margin head. File: `models/hydro_hydra.py`.
Cached probs: `campaign/probs_classifier_dataset_v{1,2,3}/`.

## How AST-10s-B works (Combined IARA+Deepship SOTA)

`MIT/ast-finetuned-audioset-10-10-0.4593` (86 M, AudioSet-pretrained ViT)
fine-tuned on `Combined_IARA_Deepship_10s` (68 k clips, 16 kHz, 10 s).

```
wav (10s @ 16kHz) → Kaldi fbank (128 mel, 25/10 ms) → AudioSet norm →
AST ViT → 4-class head (re-init)
focal γ=2.0 + inverse-freq class wts + smoothing 0.1 + balanced WRS +
SpecAugment (no mixup), AdamW diff-LR (backbone 1e-5, head 3e-4), cosine, warmup 10%
```

**Select epoch on CLIP-level val macro-F1, not per-source** — per-source (252
src, 20 Tug) overfit a noisy minority and cost ~0.2 F1. Scripts:
`campaign/train_ast2.py`, `campaign/ast_common.py`, `campaign/dump_ast2.py`,
`campaign/final_ensemble2.py`. 10s builder: `data/_make_combined_10s.py`.

Run: `train_ast2.py --data_root Combined_IARA_Deepship_10s --clip_len 160000
--src_sr 16000 --loss focal --class_weight --specaug --mixup 0 --lr_head 3e-4
--seed 2024`.

## How HydroRecurrentStacker works (⑤'s base model)

Shares `_ClipEncoder` from `models/hydro_set_spark.py` (HydroSpark trunk →
48-dim per-clip embedding). Concatenates each clip's encoder embedding with
the *frozen* 6-ckpt cargo_confirm ensemble's per-clip softmax (5 Hydra +
complete-044, 24 dims in log space), projects back to 48-dim, runs a
multi-scale (1/3/10 s) causal GRU with deep-gambler abstain.

```
K 1-s clips → _ClipEncoder → z_1..z_K (N,K,D)
            → multi-scale pool (1/3/10 s) + scale-id embed
            → causal GRU(D→D_h)
            → per-step head (K_cls + 1 abstain logit)
            → source pred = logits at last 1-s valid step
```

Loss = `GamblerCE` (smoothed focal CE + 0.1·deep-gambler abstain aux,
`abstain_o=2.2`) with `--deep_aux_w 0.2` supervising ALL GRU steps. 39k params.

Per-clip ensemble dump: `campaign/dump_ensemble_combined.py` writes
`campaign/probs_combined_recurrent_stacker.npz` (6×350120×4, ~8 min); trainer
mmaps `{abs_path → row_index}` for O(1) lookup.

Run: `train_recurrent_stacker.py --steps 8000 --K_train 30 --K_eval 60
--scales 1,3,10 --gambler_w 0.1 --abstain_o 2.2 --focal_gamma 2.0
--deep_aux_w 0.2 --lr 2e-3`. Honest MSP-isotonic gate at val cov 0.85 lives
in `eval_recurrent_stacker.py`.

### Deep-gambler abstain — calibration gotcha

The K+1 abstain logit saturates on small models (0.97/0.72/0.07 on
HydroRecurrent / MS / Stacker). **Threshold on abstain prob is uninformative.**
Fix: fit isotonic regression on val mapping `max-softmax-prob → P(correct)`
(Hendrycks–Gimpel 2017), pick the p_correct cut on val with coverage ≥ 0.85,
apply once to test. Lifted Stacker 0.653 → 0.715. Code:
`campaign/calibrate_abstain.py` and the calibration block inside
`campaign/eval_recurrent_stacker.py`.

### HydroComplete fine-tuned on IARA length-stratified

Best ckpt: `lightning_logs/hydro_complete_iara_lengthstrat/version_0/checkpoints/complete-009-p0.4340.ckpt`
(trained on `IARA_length_stratified_1s`, stopped early ep 14, best ep 9
val/macro_P 0.4340). Intended use: drop into the 6-ckpt cargo_confirm pool as
the 7th model, re-dump per-clip probs on Combined_IARA_Deepship_1s, retrain
HydroRecurrentStacker. Lever to lift the 0.33 IARA test F1 bottleneck
without disturbing Deepship.

## Datasets

- **rapid**: `/var/mnt/5A009BF8009BD8F9/Data/Classification_rapid_testing_1s` (1s @ 5120 Hz).
- **full**: `/var/mnt/5A009BF8009BD8F9/Data/Classifier_Dataset` — 254 val src / 83 test src. Per-class test: Cargo=16, Pass=27, Tanker=38, **Tug=2**.
- **Deepship_main**: 557 vessel-pass sessions, 32 kHz. Manifest `deepship_manifest_v2.jsonl` (vessel-aware 5-fold).
- **IARA** (`/var/mnt/5A009BF8009BD8F9/Data/IARA/Main`): 1825 recordings, 48 kHz, 150–300 s, partitions A–H. Platforms: A/B/C/D = Offshore Station, F/G = Wave Glider, E/H = background. Manifests under `…/IARA/manifests/`. Key option: `iara_option_1_shiptype4.json` (4-class Cargo/Tanker/Tug/Passenger). Length-strat split: `iara_opt1_lengthstratified.json`.
- **Combined_IARA_Deepship_1s** — 350 k clips, 5120 Hz; 810 train / 252 val / 236 test sources.
- **Combined_IARA_Deepship_10s** — 68 k clips, 16 kHz, 10 s (5 s hop); SAME splits as 1s build.

### Per-class breakdown — current Classifier_Dataset winner (cargo_confirm_5base)
```
            Cargo Passenger Tanker Tug
Cargo (16)    11     0       5    0    P=0.786 R=0.688 F1=0.733
Pass  (27)     0    26       1    0    P=0.929 R=0.963 F1=0.945
Tanker(38)     2     2      34    0    P=0.850 R=0.895 F1=0.872
Tug   (2)      1     0       0    1    P=1.000 R=0.500 F1=0.667
                                       macro F1=0.804  macroP=0.891
```

## Conventions

- `processing/`, `features/`, `models/`, `training/`, `inference/` — library code.
- `campaign/` — experiments; one self-contained script per experiment; outputs to `lightning_logs/<run_name>/`.
- `lightning_logs/` — all training & artifact outputs (do not delete).
- **Honest contract:** never pick model-selection knobs (subset, hparam, threshold, calibration) using test metrics. Val OOF only. Test touched once at final eval.

## Repo gotchas

- **`training/` namespace shadowing.** Site-packages `training` (OpenCLIP) shadows local `training/`. Keep `training/__init__.py` (empty marker).
- **Capitalized split dirs.** `Classifier_Dataset/{Train,Val,Test}` is capitalized; `DALIAudioDataModule._scan_split` expects lowercase. Use symlink `Classifier_Dataset_lc` for legacy trainers; `campaign/train_complete_v2.py` is case-insensitive.
- **Combined holdout Tug covariate shift** caps macro-F1 ~0.70-0.75. Train Tug = small harbor tugs (median 30 m, 6 large-supply examples); test Tug = large offshore supply (`Tug/Supply Vessel`, median 81 m). IARA Passenger has only 11 train / 6 test src — test F1 = 0. Don't chase 0.90 on this holdout under RULE 1.
- **Glider partitions F+G are NOT the IARA bottleneck.** Removing them hurt overall test F1 −0.015. The bottleneck is the supply-vessel-size mismatch, not platform.

## Compute notes

- Python: `/var/home/damo/Documents/miniconda3/miniconda3_install/bin/python`.
- HGB fits fast (<1s); subset × cfg × booster grids blow up combinatorially — keep selection budgets disciplined.
- `auto_tune/study.db` — Optuna sqlite store; mark stuck RUNNING trials as FAIL before resuming.

## When in doubt

- Honest contract first; trust val OOF only.
- Architectural improvement = structural change, not more stacker tuning.
- If val OOF moves and test doesn't, you're overfitting val.
