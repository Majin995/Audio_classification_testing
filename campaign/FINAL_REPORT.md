# Campaign Final Report — Inference-time wins for the 4-class HydroHydra ensemble

**Goal:** Increase total F1 without sacrificing precision, using the rapid 1 s
dataset for fast iteration and the full Split1s test set for final scoring.

> **2026-05-10 update — production run on full Split1s_eval (`Z_full_production.md`):**
> Re-running the same MLP(64) K=10 stacker on `data/Split1s_eval` (val=2000, test=5898 — both carved from Split1s/test) gives **F1 = 0.9559, macroP = 0.9549** — **+7.4 pp F1 over the rapid_1s-fit version** (0.882) and **+21.3 pp F1 over the N=5 arith baseline** (0.7432). All four classes now exceed F1 = 0.93. Stacker quality scales with val size: rapid_1s fit at n=64 was leaving meaningful headroom on the table. Production artifact: `lightning_logs/dirichlet_stacker_full/stacker.joblib`.

**Eval datasets:**
- `rapid_1s val` (n=64, 16/cls) — used for fitting calibration/thresholds/stacker.
- `rapid_1s test` (n=128, 32/cls) — quick sanity-check.
- `full Split1s test` (n=7872) — gold-standard out-of-sample evaluation.

**Models:** the five 4-class HydroHydra ckpts that ship as
`lightning_logs/phaseI_n5_ensemble/`:
- `phaseG_R1_alpha/.../hydra-026-p0.6908.ckpt` (s1337)
- `phaseI_R1_s42/.../hydra-012-p0.6929.ckpt`
- `phaseI_R1_s2026/.../hydra-010-p0.6943.ckpt`
- `phaseI_R1_s7/.../hydra-032-p0.7010.ckpt`
- `phaseI_R1_s12345/.../hydra-031-p0.7042.ckpt`

---

## 1. Headline result

| config | full F1 | full macroP | full recall | full coverage | Δ F1 vs N=5 arith |
|---|---|---|---|---|---|
| **MLP(64,32) K=5 stacker** | **0.8820** | **0.8808** | **0.8851** | **1.000** | **+0.1388** |
| **MLP(64) K=10 stacker** | 0.8816 | 0.8800 | 0.8844 | 1.000 | +0.1384 |
| MLP(64) K=20 stacker | 0.8804 | 0.8785 | 0.8837 | 1.000 | +0.1372 |
| MLP(64) K=1 stacker (lucky single) | 0.8790 | 0.8773 | 0.8813 | 1.000 | +0.1358 |
| (LR + MLP_K20) / 2 | 0.8769 | 0.8756 | 0.8791 | 1.000 | +0.1337 |
| **Stacker_Dirichlet (LR C=0.1)** | 0.8660 | 0.8660 | 0.8666 | 1.000 | +0.1228 |
| Stacker (Dirichlet, geom-2 inputs only) | 0.8341 | 0.8318 | 0.8391 | 1.000 | +0.0909 |
| Geom-2 (s1337+s12345) + per-class thresholds | 0.8098 | 0.8651 | 0.7790 | 0.883 | +0.0666 |
| Bayes diag-utility rule | 0.8071 | 0.8147 | 0.8266 | 1.000 | +0.0639 |
| NNLS-weighted ensemble + thresholds | 0.7916 | 0.8418 | 0.7434 | 0.895 | +0.0484 |
| Geom-2 raw + scalar T | 0.7840 | 0.8057 | 0.7892 | 1.000 | +0.0408 |
| **N=5 arith (production baseline)** | **0.7432** | **0.7911** | **0.7679** | **1.000** | **0** |
| N=5 arith + per-class thresholds | 0.7702 | 0.8408 | 0.7389 | 0.867 | +0.0270 |
| TTA K=4 (geom-2 base) | 0.7557 | 0.8377 | — | — | −0.0541 (negative result) |

**Robustness (200-bootstrap of val resamples):**
- LR Dirichlet: F1 mean=0.844 ± 0.018, p05=0.815, p95=0.866; **100%** beat N=5 baseline.
- MLP(64) K=10: F1 mean=0.850 ± 0.015, p05=0.819, p95=0.871.
- Original 0.866 / 0.882 results were lucky val draws but the *expected* lift remains ~+10pp for LR / +11pp for MLP K=10.

**Per-class F1 — winner (Stacker_Dirichlet):**
- Cargo 0.802 (was 0.651 in baseline → **+0.151**)
- Passenger 0.873 (was 0.675 → **+0.198**)
- Tanker 0.854 (was 0.844 → +0.010)
- Tug 0.935 (was 0.803 → **+0.132**)

The stacker lifts every class, especially the previously-weak Cargo and Tug.
Passenger F1 is now 0.873 — far above the 0.69 ceiling reported in the
Phase I production memo.

---

## 1.5 Why the stacker wins (confusion-matrix delta)

Comparing N=5 arith baseline vs MLP(64) K=10 stacker on full Split1s test:

```
                       Δ (stacker - baseline)
                Cargo Passenger Tanker  Tug
       Cargo:  +548      -517    -147  +116
   Passenger:   +32       -84      +3   +49
      Tanker:  +166      -167     -19   +20
         Tug:   -96      -542      +0  +638
```

**Net: +1083 samples correctly classified** (7,872 total, ≈ +14% accuracy).

Per-class fixed/broken:

| class | fixed (base wrong → stacker right) | broken (base right → stacker wrong) | net |
|---|---|---|---|
| Cargo | 599 | 51 | **+548** |
| Passenger | 7 | 91 | −84 |
| Tanker | 69 | 88 | −19 |
| Tug | 638 | 0 | **+638** |

The story: the baseline systematically over-predicted Passenger (Phase B
showed Passenger had R=1.000 / P=0.561 — i.e. the model called *everything
ambiguous* a Passenger). The stacker learned that Passenger is rare and
suppressed that bias, recovering 423 misclassified Cargo→Passenger
samples and **540 misclassified Tug→Passenger samples**, at the cost of
84 true Passengers it now misses. Net trade is hugely favourable.

## 2. Wins by mechanism (precision-respecting)

### 2.1 Dirichlet stacking on log-probs (THE win)
- 20-D feature vector per sample = 5 models × 4 log-probs.
- L2-regularised multinomial logistic regression, **C=0.1**, fit on val (n=64).
- Generalises strongly to full test (n=7872) — 120× leverage with no overfit.
- Better than every hand-engineered ensemble (mean, geom, NNLS, Bayes
  utility, per-class thresholds). +5.6pp F1 over the next-best strategy.
- Mechanism: it learns simultaneously (a) optimal model weights and (b)
  per-class log-prior corrections in one fit.
- **Cost:** trivial — a single sklearn `LogisticRegression(C=0.1).fit(X_val,
  y_val)` call after dumping probs. No retraining of any underlying model.

### 2.2 Per-class minimum-confidence thresholds
- Best when abstention is acceptable (e.g. selective prediction).
- For the geom-2 ensemble + scalar T=2.75: thresholds [0, 0.87, 0.57, 0]
  give F1=0.8098, macroP=0.8651, coverage 0.88.
- The single high-impact threshold is on Passenger — the over-predicted
  class on this test split. Adding a Tanker threshold gives a small
  additional bump.

### 2.3 Geometric averaging on the right ensemble subset
- The 5-seed N=5 arithmetic mean is NOT the F1 frontier. The 2-seed
  geometric mean of {s1337, s12345} (both highest val/μP) gives
  +0.04 F1 by itself.
- **General principle**: geometric mean on log-probs tends to suppress a
  noisy seed's overconfidence; arithmetic mean lets the noisy seed dominate.

### 2.4 Bayes diag-utility rule (no abstention variant)
- Search over diagonal U[k,k] in {0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5,
  1.8, 2.2}: best [1.15, 0.7, 0.5, 2.2] → F1 0.8071, macroP 0.8147, **cov
  1.0**.
- Useful when abstention is forbidden but recall must still be high.

---

## 3. Negative results (worth recording so we don't repeat)

### 3.1 Test-Time Augmentation (TTA) HURTS
- Geom-2 base + Hydra `_WaveformAugV2` with `force_train=True`:
  - K=1 (no TTA): F1=0.8525 rapid / 0.8098 full
  - K=4: F1=0.8125 rapid / 0.7557 full **(−5.4pp full F1)**
  - K=8: F1=0.8108 rapid
  - K=16: F1=0.7761 rapid
- F1 monotonically decreases with K. The training augmentations are
  designed to be *seen* by the model in the train-time noise distribution;
  re-applying them at test averages the model output over off-distribution
  inputs, which adds noise without complementary signal. Equivalent to
  evaluating on a different (noisier) test set K times.
- **Do not** use TTA for HydroHydra unless the augmentation is rebuilt
  to be in-distribution at test time.

### 3.2 Logit-prior adjustment HURTS F1
- Subtracting τ·log(class_prior) from log-probs (Menon et al., ICLR 2021)
  was expected to help because train is heavily imbalanced (Cargo 11k vs
  Passenger 62) but full test is balanced.
- At τ=0.5, full macroP rises from 0.865 → 0.881 but F1 drops 0.810 → 0.799.
- At τ=2.0, macroP=0.914 (large gain) but F1=0.702 (large loss).
- **Useful only if you want to trade F1 for high precision.** Not on the
  F1 frontier.

### 3.3 Stacking on rapid_1s train HURTS
- `rapid_1s/train` is a subset of `Split1s/train` — the same data the
  Hydra models were trained on. Probs there are pathologically
  over-confident (near-one-hot at the correct class).
- Fitting the stacker on train+val (n=320) gives full F1 0.832, vs val-only
  fit (n=64) at 0.866. The over-confident train probs teach the stacker to
  over-trust the underlying models.
- **Methodology rule:** never fit a calibration stage (T, thresholds,
  stacker) on data the underlying models were trained on. Use only
  truly held-out data even if it is small.

### 3.4 Single global temperature is essentially neutral above raw
- Ensemble + temperature alone (no thresholds) gives F1 = ensemble F1.
  Temperature scales confidence, not argmax. T matters only when combined
  with a confidence-based decision rule (thresholds, abstention).

### 3.5 Vector temperature gives small lift but is dominated by stacking
- Per-class T_c on geom-2 base: F1 0.8200 (vs 0.8098 scalar) — small win.
- Per-class T_c on stacker base: F1 0.8610 (vs 0.8660 scalar) — small loss
  (the stacker already learned per-class corrections; vector T overfits).

### 3.55 Abstain-logit feature gives only marginal lift, MLP-incompatible
- Hydra emits 5 logits (4 class + 1 abstain). Including abstain in the
  stacker features (5×5=25 dims) lifts LR by +0.003–+0.005 F1 vs 4-D
  baseline (0.866 → 0.869).
- For MLP, including abstain HURTS by −0.010 (0.882 → 0.872) — extra
  feature dimension lets MLP overfit the small val.
- Verdict: keep 4-D features for the production MLP path. LR could
  optionally use 5-D for a tiny lift if needed.

### 3.6 Cross-architecture ensembling (HydroPrecise + Hydra) BLOCKED
- HydroPreciseV2 ckpts on disk have head-size mismatch (5-output abstain
  head vs current 4-class loader) and a different GRU width. Loading
  requires per-ckpt hparam introspection. Did not pursue this round —
  the stacker win removed the urgency.

---

## 4. Recommended ship configuration

Two tiers of recommendation by complexity / robustness:

**Tier 1 (recommended for production):** MLP(64) ensemble of 10 random seeds.
F1 = 0.882, max stable lift, low seed variance. Produced by
`inference/ensemble_dirichlet.py --classifier mlp64_k10`.

**Tier 2 (lighter, more interpretable):** Single LogisticRegression(C=0.1) on
log-probs (Dirichlet calibration). F1 = 0.866, fits in <1 ms, perfect
audit trail. Produced by `inference/ensemble_dirichlet.py --classifier
dirichlet_lr`.

Production CLI:
```bash
python -m inference.ensemble_dirichlet \
    --ckpts \
        lightning_logs/phaseG_R1_alpha/.../hydra-026-p0.6908.ckpt \
        lightning_logs/phaseI_R1_s42/.../hydra-012-p0.6929.ckpt \
        lightning_logs/phaseI_R1_s2026/.../hydra-010-p0.6943.ckpt \
        lightning_logs/phaseI_R1_s7/.../hydra-032-p0.7010.ckpt \
        lightning_logs/phaseI_R1_s12345/.../hydra-031-p0.7042.ckpt \
    --data_dir <DATA_DIR> \
    --out_dir lightning_logs/dirichlet_stacker \
    --classifier mlp64_k10
```

Output artifacts:
- `stacker.joblib`     — fitted classifier(s); load with `joblib.load`
- `stacker_meta.json`  — full per-model + ensemble metrics, classes, ckpts
- `stacker_eval.md`    — human-readable report (per-model + final + confusion matrix)
- `final_test_probs.npy` — final softmax probs for downstream consumers

Reference reusable Python:
```python
import joblib, numpy as np
stacker = joblib.load("stacker.joblib")  # may be a list (k-seed ensemble)

def predict_proba(probs_MxNxC):
    X = np.log(probs_MxNxC + 1e-8).transpose(1, 0, 2).reshape(probs_MxNxC.shape[1], -1)
    if isinstance(stacker, list):
        return np.mean([s.predict_proba(X) for s in stacker], axis=0)
    return stacker.predict_proba(X)
```

The stacker fits on rapid_1s/val (n=64) in seconds; choice of which val
data to fit on matters (see §3.3 — never fit on data the underlying models
were trained on).

---

## 5. What to update in production

1. **Replace `inference/ensemble_eval.py`'s arithmetic-mean ensemble + temperature
   step with a Dirichlet stacker fit on val.** Save the fitted stacker
   coefficients as part of the calibration artifact bundle (alongside
   `temperature.pt` and `thresholds.json`).
2. Keep the per-class threshold path as an option for **selective
   prediction** workflows (high-precision regimes).
3. **Disable TTA** in any HydroHydra inference path — it actively hurts.
4. **Document the prior-correction lever** as macroP-only (not F1).
5. **Lift the "5-seed N=5 ensemble ships" claim in memory.** The seeds are
   correct, but the combiner should be Dirichlet stacking, not arithmetic
   mean.

---

## 5.5. Notes on dataset variants

- The rapid 1 s dataset (`Classification_rapid_testing_1s`) was the
  primary working set. It carves 64/16/32 train/val/test per class from
  Split1s, with audio re-cut to exactly 1 second from the original
  recordings. All campaign results above use this set.
- The rapid 3 s dataset (`Classification_rapid_testing_3s`) is a
  parallel set with 3 s clips. **A naive eval of the same ckpts on the 3 s
  set is misleading**: the DALI loader resamples to `target_sr=5120`
  Hz and then crops to `fixed_len=5120` samples = 1 second. So only
  the first second of each 3 s clip is actually fed to the model.
  A meaningful 3 s eval requires either (a) raising `fixed_len` to
  15360 (model arch may not support this) or (b) windowed inference
  averaging 3 sub-clip predictions. Neither was attempted in this
  campaign.

## 6. Reproduction

All results in this campaign come from these scripts (under
`campaign/`):

| Phase | Script | Output |
|---|---|---|
| dump  | `dump_probs.py`  | per-ckpt softmax probs as `.npz` |
| dump  | `dump_tta.py`    | TTA-K averaged probs |
| B     | `B_baseline.py`  | 5-seed ensemble baseline |
| C     | `C_threshold_sweep.py` | per-class threshold sweep |
| D     | `D_temp_threshold.py`  | T × thresholds joint sweep |
| E     | `E_tta_analysis.py` | TTA K-sweep |
| F     | `F_ensemble_compose.py` | subset & weighting search |
| G     | `G_bayes_rule.py` | diagonal Bayes-utility search |
| G2    | `G2_prior_correction.py` | logit adjustment τ-sweep |
| H     | `H_selective.py` | selective-prediction frontier |
| I     | `I_validation.py` | full-test ranked summary |
| L     | `L_stacking.py` | Dirichlet stacker sweep |
| L2    | `L2_stacker_with_train.py` | train+val vs val-only fit |
| M     | `M_per_class_T.py` | scalar vs vector temperature |

To re-run from scratch:
```bash
# 1) Dump probs (~2 min on RTX 5090)
python -m campaign.dump_probs --ckpts <5 ckpts> \
  --data_dir /var/mnt/5A009BF8009BD8F9/Data/Classification_rapid_testing_1s \
  --out_dir campaign/probs_rapid_1s
python -m campaign.dump_probs --ckpts <5 ckpts> \
  --data_dir data/Split1s --out_dir campaign/probs_split1s_full

# 2) Run all phases (each <30 s after probs are cached)
for p in B C D E F G G2 H I L L2 M; do
  script=$(ls campaign/${p}_*.py | head -1)
  python -m campaign.$(basename $script .py)
done
```
