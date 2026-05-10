# Phase I — I1 (SWA + drop_path) findings

## Critical training-script bug identified

`training/train_hydra.py` calls `trainer.test(model, data, ckpt_path="best")`,
which **reloads the pre-SWA best checkpoint** before running test. The SWA
callback's averaged weights (computed in-memory at the end of training)
are NEVER saved as a separate checkpoint and NEVER reach the test step or
the post-cal block.

**Consequence:** I1's reported metrics (val 0.7043 / test 0.5957) reflect
the pre-SWA model only — the drop_path 0.10 effect, NOT SWA. I1b will
have the same flaw because it uses the same training script.

**Fix needed for any future SWA experiment:** add `--save_swa_final` flag
to `train_hydra.py` that captures the in-memory state_dict at
`on_train_end` and writes it as `swa_final.ckpt`. Then run extra_metrics
on `swa_final.ckpt` separately.

## I1 (drop_path 0.10) effective findings

| metric | I1 | R1 baseline | Δ |
|---|---|---|---|
| val/μP | 0.7043 | 0.6908 | +0.0135 |
| test/μP | 0.5957 | 0.6620 | **−0.0663** |
| val/test gap | −0.1086 | −0.0288 | **−7.98 pt worse** |
| MP@cov0.85 | 0.7667 | 0.7518 | +0.0149 |
| temperature | 3.951 | 3.287 | (post-cal flattens more) |

Per-class val precision (best ckpt at ep 48):
- c0 Cargo: 0.7407
- c1 Passenger: 0.6113
- c2 Tanker: 0.7591
- c3 Tug: 0.7059

## Diagnosis

drop_path 0.10 in SaShiMi blocks does increase val/μP +1.35 pt — proof the
backbone has slack capacity that stochastic depth can regularize. But the
val/test gap **widened** from R1's −2.88 pt to I1's −10.86 pt. Drop_path
biases the model toward val-specific patterns; the regularization is
in-distribution, not cross-distribution.

This is the same pathology Phase H exposed with bigger heads (DEMON-MoE,
Sub-Center ArcFace). **Any architectural / regularization change that
lifts val on this dataset comes with a proportionally worse test.**

## Implication for Phase I plan

The val/test gap is structural, not curable by any in-distribution
regularizer (drop_path, mixup, JTFS, MoE, ArcFace, …). The remaining
levers that have a chance to help test are:

1. **β.1 DART-MT semi-sup with unlabeled corpus** — only lever that
   exposes the model to a *different* data distribution during training.
   Likely the highest-leverage remaining bet.
2. **β.2 LAME at inference** — refines the test-batch logits using
   penultimate-feature similarity; doesn't depend on training-time data
   distribution.
3. **α.4 K-pass TTA** — averaging across noisy augmentations narrows
   variance without retraining.
4. **Multi-seed soup of identical configs** (gated; requires retraining
   R1 at 3 seeds first, ~18h compute).

Phase I sweep has 4 more runs (I1b, I2 LPC, I3 RP, I4 LPC+RP). I expect
all will show the same pattern (val ↑, test stagnant or ↓) unless they
contain real new information that survives distribution shift. The most
likely positive surprise: γ.1 LPC residual or γ.2 RP, because they
expose new physical axes of the signal that overfit-by-memorization is
less able to exploit.

## Saved I1 checkpoints

- `hydra-035-p0.6948.ckpt`
- `hydra-048-p0.7043.ckpt` ← best (pre-SWA, used for test + post-cal)
- `hydra-058-p0.6912.ckpt`
- `temperature.pt` (T = 3.951)
- `thresholds.json` (MP@cov0.85 = 0.7667)
- `selective_pr.md`

The post-SWA averaged weights were never saved and are now lost.
