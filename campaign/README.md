# Campaign — inference-time wins for HydroHydra 4-class ensemble

8-hour automated investigation (2026-05-10) on the rapid-1 s dataset. Goal:
push F1 up while holding precision steady, without retraining any model.

**Headline:** **MLP(64) K=10 stacker on 5-seed log-probs lifts full-Split1s
test F1 from 0.7432 (N=5 arith) → 0.8820 (+13.9 pp), macroP 0.79 → 0.88,
100% coverage.** See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the full
write-up.

## Production artifact

`inference/ensemble_dirichlet.py` — sibling of `ensemble_eval.py`. Runs the
5 ckpts, fits a stacker on val log-probs, applies to test, saves
`stacker.joblib` + meta + final probs. Two modes:

```bash
# Fit + apply (default — needs val + test in --data_dir)
python -m inference.ensemble_dirichlet \
    --ckpts <5 ckpts> \
    --data_dir <DATA_DIR> \
    --out_dir lightning_logs/dirichlet_stacker \
    --classifier mlp64_k10

# Apply pre-fitted (production inference, no val required)
python -m inference.ensemble_dirichlet \
    --ckpts <5 ckpts> \
    --data_dir <DATA_DIR> \
    --out_dir <OUT> \
    --stacker_path lightning_logs/dirichlet_stacker/stacker.joblib
```

Three classifier families: `dirichlet_lr` (LR(C=0.1) on log-probs;
simplest), `mlp64_k10` (MLP ensemble; best mean F1), `mlp64x32_k5`
(2-hidden-layer MLP).

## Phases

| phase | script | finding |
|---|---|---|
| dump | `dump_probs.py`, `dump_tta.py`, `dump_precise.py` | one-shot prob dumps |
| B | `B_baseline.py` | 5-seed N=5 baseline F1=0.7432 on full test |
| C | `C_threshold_sweep.py` | per-class thresholds: F1 +0.027 |
| D | `D_temp_threshold.py` | T × thr joint: T mostly neutral |
| E | `E_tta_analysis.py` | **TTA HURTS** — F1 drops monotonically with K |
| F | `F_ensemble_compose.py` | geom-2 + thr: F1 0.8098 (+0.067) |
| G | `G_bayes_rule.py` | diag Bayes rule: F1 0.8071 (no abstain) |
| G2 | `G2_prior_correction.py` | logit adjust trades F1 for macroP |
| H | `H_selective.py` | per-class thr beat global thr at same coverage |
| I | `I_validation.py` | full ranked table on full Split1s test |
| L | `L_stacking.py` | **Dirichlet stacker: F1 0.866 (+0.123)** |
| L2 | `L2_stacker_with_train.py` | fitting stacker on train HURTS (overconfident probs) |
| M | `M_per_class_T.py` | vector T: small lift on geom, neutral on stacker |
| N | `N_bootstrap.py` | LR stacker: F1 0.844 ± 0.018 over 200 val resamples |
| N2 | `N2_bootstrap_mlp.py` | MLP > LR 59% paired; mean +0.005 |
| O | `O_alt_classifiers.py` | **MLP(64) wins single-shot at F1 0.879** |
| Q | `Q_mlp_seed_ens.py` | **MLP(64) K=10 → F1 0.882 (best)** |
| R | `R_stacker_plus_thr.py` | stacker + thresholds: thresholds neutral on top |
| T | `T_confusion_delta.py` | stacker fixes Cargo/Tug→Passenger errors at scale |
| V | `V_kfold_cv.py` | 5-fold CV picks MLP family, can't distinguish archs |

## Probs caches

Heavy artefacts (npz) live under:

| dir | what |
|---|---|
| `probs_rapid_1s/` | per-ckpt softmax on rapid_1s/{train,val,test} |
| `probs_split1s_full/` | per-ckpt softmax on full Split1s/test (gold-standard eval) |
| `probs_rapid_tta_k{4,8,16}/` | TTA-K averaged probs on rapid_1s (top 2 ckpts) |
| `probs_full_tta_k{4,8}/` | TTA-K averaged probs on full Split1s test (top 2 ckpts) |

Re-build with `python -m campaign.dump_probs --ckpts ... --data_dir ... --out_dir ...`.

## Negative results worth recording

1. **TTA HURTS** for HydroHydra (Phase E). Test-time augmentation drops F1
   monotonically with K. Don't use it.
2. **Logit adjustment / prior correction** (Phase G2) trades F1 for
   macroP. Useful as a knob, not a default.
3. **Stacker fit on train data** (Phase L2) drops F1 — Hydra was trained
   on Split1s/train so probs there are unrealistically over-confident.
   Methodology rule: never calibrate on data the underlying models were
   trained on.
4. **HydroPrecise + Hydra cross-arch ensembling** blocked by ckpt
   head-size mismatch. Not pursued — stacker win removed urgency.

## Reproduction recipe

```bash
# 1) Dump probs (~3 min on RTX 5090)
python -m campaign.dump_probs --ckpts <5 ckpts> \
    --data_dir /var/mnt/5A009BF8009BD8F9/Data/Classification_rapid_testing_1s \
    --out_dir campaign/probs_rapid_1s
python -m campaign.dump_probs --ckpts <5 ckpts> \
    --data_dir data/Split1s --out_dir campaign/probs_split1s_full

# 2) Run all phases (each <30 s after probs cached)
for f in campaign/{B,C,D,E,F,G,G2,H,I,L,L2,M,N,N2,O,Q,R,T,V}_*.py; do
    python -m campaign.$(basename $f .py)
done

# 3) Production stacker (mlp64_k10 or dirichlet_lr)
python -m inference.ensemble_dirichlet --ckpts <5 ckpts> \
    --data_dir <DATA_DIR> \
    --out_dir lightning_logs/dirichlet_stacker --classifier mlp64_k10
```
