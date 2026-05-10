# Phase I — I2b LPC-fix retry findings

## Outcome (vs R1 baseline)

| metric | I2b LPC-fix | R1 baseline | Δ |
|---|---|---|---|
| val/μP | 0.6918 (ep 26) | 0.6908 | +0.0010 |
| test/μP | 0.6025 | 0.6620 | **−0.0595** |
| val/test gap | −0.0893 | −0.0288 | **−6.05 pt worse** |
| MP@cov0.85 | 0.7335 | 0.7518 | −0.0183 |
| AUROC | 0.808 | (n/a) | — |
| F1 | 0.592 | (n/a) | — |
| MCC | 0.476 | (n/a) | — |
| temperature | 3.168 | 3.287 | — |
| recall_floor 0.6 | **infeasible** | feasible | — |

Best ckpt: `hydra-026-p0.6918.ckpt`. EarlyStopping fired at ep 41 (15 patience past
best at ep 26).

## Per-class precision at coverage 0.85

| class | precision |
|---|---|
| Cargo | 0.6473 |
| Passenger | 0.7867 |
| Tanker | 0.8446 |
| Tug | 0.6659 |

Tug at 0.6659 just barely meets the 0.65 plan gate; Cargo at 0.6473 underperforms
R1's class-0 baseline.

## What the tanh-squash fix proved

The Levinson-Durbin reflection-coefficient tanh squash *did* cure the I2 NaN
collapse. Training proceeded cleanly through 41 epochs with no NaN gradients,
no loss spikes, and a sensible val curve (0.605 → 0.692 over 26 epochs). The
LPC stream is now a numerically stable contributor to the model.

## What the I2b run *also* proved

LPC residual + AR-coefficient stream is **not** the orthogonal-information win
the plan hoped for. It exhibits the same val-overfitting / test-regression
pathology as every Phase H lever and every Phase I in-distribution
regularizer:

- val ↑ +0.10 pt (essentially noise)
- test ↓ −5.95 pt (large regression)
- val/test gap widens from −2.88 to −8.93 pt (3.1× worse)

The "source/filter factorization" hypothesis was that LPC residual would expose
propeller-cavitation transients orthogonal to the existing 4 envelope-/scattering-
based branches. Empirically, the LPC features compress to roughly the same
val-discriminative subspace the rest of the model already exploits — the +0.10
pt val lift is within seed noise, and the −5.95 pt test drop indicates the
LPC-derived features are highly dataset-specific (memorize Lexar Train,
fail to generalize to Lexar Test).

## Phase I final scoreboard (all runs)

| run | val/μP | test/μP | gap | MP@0.85 | gate (≥0.6700 test) |
|---|---|---|---|---|---|
| **R1** (G ship) | 0.6908 | **0.6620** | −2.88 | **0.7518** | **PASS (held)** |
| I1 SWA+drop_path | 0.7043 | 0.5957 | −10.86 | 0.7667 | FAIL |
| I1b SWA-only | 0.7006 | 0.6051 | −9.55 | 0.7722 | FAIL |
| I2 LPC (NaN) | 0.0652 | 0.0865 | — | — | CRASH |
| I2b LPC-fix | 0.6918 | 0.6025 | −8.93 | 0.7335 | FAIL |
| I3 RP | 0.6544 | 0.5751 | −7.93 | 0.6878 | FAIL |
| I4 LPC+RP (killed) | — | — | — | — | — |

**0/5 Phase I runs beat R1 on test/μP.** Adding capacity, regularization, or
new front-end features all fail. The pattern is consistent and now overdetermined.

## Diagnosis (now overdetermined across Phase H + Phase I)

The val/test gap is a **train-vs-test distribution shift on Lexar**, not:
- a flat-minima problem (SWA failed)
- a regularization deficit (drop_path failed)
- a front-end feature deficit (LPC, RP failed; Phase H JTFS, HELIX failed)
- a head-design problem (Phase H DEMON-MoE, Sub-Center ArcFace failed)
- a loss-function problem (Phase H LDAM-DRW failed)

Every lever that adds capacity or shifts the inductive bias improves val (because
val and train share distribution properties) and damages test (because test
samples come from a different distribution and the new features don't transfer).

## Implication for the plan

Per `look-into-running-optuna-lexical-zephyr.md` decision tree:
- α (free generalization wins): all attempted, ∼0 net gain on test.
- γ.1 LPC: gate (test ≥ 0.67) **FAILED** → drop.
- γ.2 RP: gate **FAILED** → drop.
- γ.3 cyclostationary CSC: P1, gated by γ.1/γ.2 outcomes — given γ.1+γ.2 both
  fail and they were the cheaper variants, γ.3 is unlikely to behave differently.
  **Recommend defer until β succeeds.**
- β.1 DART-MT semi-sup: **only remaining lever that could survive distribution
  shift** (it sees a different distribution during training via the unlabeled
  corpus). Highest-leverage remaining bet. Proceeding.

## Next action

Implement β.1 — `HydroDARTMTHydra` mean-teacher wrapper around HydroHydra.
~250 LOC: fresh `training/train_hydra_mt.py`, EMA teacher inside HydroHydra
init, FreeMatch-style class-adaptive thresholds on pseudo-label confidence,
unlabeled loss ramp 0→1 over epochs 5–15, pair with LAME at inference for
confirmation-bias insurance.

Compute: ~9h training (vs R1's 6h) due to extra teacher forward pass + extra
unlabeled batch per step.

Expected result: if β.1 produces test/μP ≥ 0.68, the unlabeled-corpus thesis
is validated and β.1 ships as new full-coverage baseline. If β.1 *also* fails
test (showing pseudo-label confirmation bias), the conclusion will be that
Lexar Train and Lexar Test were drawn from genuinely different generative
processes (sensor / mission / depth / sea-state) and no in-domain
self-training can bridge the gap; the user's UATR distribution shift is
*structural* and only an entirely different test set or a domain-adversarial
training regime can move the number.

## Saved I2b artifacts

- `hydra-016-p0.6806.ckpt`
- `hydra-021-p0.6751.ckpt`
- `hydra-026-p0.6918.ckpt` ← best
- `hydra-033-p0.6819.ckpt`
- `temperature.pt` (T = 3.168)
- `thresholds.json` (MP@cov0.85 = 0.7335; recall_floor infeasible)
- `selective_pr.md`
