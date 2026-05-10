# Phase H — Extended metric suite (beyond F1 / precision / recall)

Generated 2026-05-06 via `python -m scripts.extra_metrics --use_temperature`.
Each row uses the per-ckpt `temperature.pt` so calibration metrics reflect
the post-cal model.

## Test split — all five best ckpts

| Run | val/μP (raw) | test/μP (raw) | macro AUROC | macro AUPR | Cohen κ | MCC | ECE↓ | Brier↓ | T |
|---|---|---|---|---|---|---|---|---|---|
| **R1 baseline** | 0.6908 | 0.6620 | **0.8393** | **0.6920** | 0.4965 | 0.4868 | **0.1212** | **0.5140** | 3.29 |
| H1 LDAM | 0.7035 | 0.6614 | 0.8164 | 0.6065 | **0.5075** | 0.4968 | 0.2434 | 0.6091 | 0.65 |
| H1b LDAM-drw15 | 0.7123 | 0.6299 | 0.7347 | 0.4989 | 0.4238 | 0.4174 | 0.1848 | 0.6346 | 0.66 |
| H5 DEMON-MoE | 0.7279 | 0.6128 | 0.8326 | 0.6512 | 0.4720 | 0.4598 | 0.1559 | 0.5573 | 3.36 |
| H7b stacked-drw15 | 0.7364 | 0.5982 | 0.7282 | 0.5149 | 0.4475 | 0.4395 | 0.2143 | 0.6335 | 0.67 |

↓ = lower-is-better.

## Headline

**R1 baseline dominates on every distribution-aware metric.** Phase H's "H1
LDAM ties R1 on test/μP" finding was misleading — H1 is +0.011 on Cohen's κ
but R1 wins by:
- **+2.29 pt on AUROC** (0.8393 vs 0.8164) — R1's rank-ordering is materially better.
- **+8.55 pt on AUPR** (0.6920 vs 0.6065) — R1 separates positives from negatives much more cleanly.
- **−12.22 pt ECE** (0.1212 vs 0.2434) — R1 is twice as well-calibrated.
- **−9.51 pt Brier** (0.5140 vs 0.6091) — R1's probabilistic accuracy is materially better.

The whole-distribution view collapses every Phase H lever back below R1 —
including H7b which had the highest val/μP of any run (0.7364) but ranks
**worst on AUROC** (0.7282) and **worst on Brier** (0.6335 tied with H1b).

## Calibration regime split

LDAM models (H1, H1b, H7b) all have **temperature ≈ 0.66** — the post-cal
fit is *sharpening*, meaning the LDAM margin pre-trains an already-overconfident
model that calibration can't fully fix.

R1 and H5 have **temperature ≈ 3.3** — *flattening*, the typical regime for
LMF / softmax-CE models, where calibration corrects garden-variety overconfidence
caused by deep nets. ECE in this regime is much lower because temperature can do
its full job.

**Implication**: LDAM-margin loss interacts poorly with post-hoc temperature
calibration. If a deployment uses calibrated probabilities (e.g. for
selective prediction or risk-aware routing), LDAM is the wrong loss
choice on this dataset.

## Per-metric winners on test

| Metric | Winner | Value |
|---|---|---|
| macro AUROC | R1 | 0.8393 |
| macro AUPR | R1 | 0.6920 |
| Cohen κ | H1 | 0.5075 (tight; R1 is 0.4965) |
| MCC | H1 | 0.4968 (R1 is 0.4868) |
| ECE (lower=better) | **R1** | 0.1212 |
| Brier (lower=better) | **R1** | 0.5140 |
| Balanced accuracy | (per-ckpt JSON) | — |

## Per-class breakdown

Detailed per-class precision / recall / F1 / FPR and confusion matrices live
in each ckpt's `extra_metrics.md` next to the checkpoint:

- `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/extra_metrics.md`
- `lightning_logs/phaseH_H1_ldam_drw/version_0/checkpoints/extra_metrics.md`
- `lightning_logs/phaseH_H1b_ldam_drw_early/version_0/checkpoints/extra_metrics.md`
- `lightning_logs/phaseH_H5_demon_moe/version_0/checkpoints/extra_metrics.md`
- `lightning_logs/phaseH_H7b_stacked_drw15/version_0/checkpoints/extra_metrics.md`

## Recommended ship config (full coverage)

**R1 Phase G baseline** is now the unambiguous full-coverage winner. Phase H's
contention that H1 LDAM "ties on test/μP" doesn't survive the wider metric
panel — R1 wins AUROC, AUPR, ECE, Brier all at once.

For gated-coverage deployments (abstain ≤15%) **H5 DEMON-MoE** still wins on
post-cal MP@cov0.85 (0.7905 vs R1's 0.7518), with respectable AUROC (0.8326)
and ECE (0.1559).

## Phase I implication

Closing the val/test gap is necessary but not sufficient — Phase I's α and γ
levers should also be measured on AUROC / Brier / ECE, not just test/μP. The
plan's `compile_phaseI_results.py` should be extended to pull these from
`extra_metrics.json` once Phase I runs land.
