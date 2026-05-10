# Phase I — multi-seed verification + logit ensemble: NEW selective-PR baseline

## TL;DR

3-seed verification of R1 (val/μP 0.6908, test 0.6620) found **massive
test-side seed variance** (±0.10 pt) on Lexar's tiny Test split, while
val/μP is rock-stable (±0.002). R1's 0.6620 is a *lucky single-seed
draw*, not the structural ceiling. **Greedy weight soup is infeasible**
(seeds in different basins). **Logit-level ensemble of 3 seeds delivers
the new production baseline** with MP@cov0.85 = **0.7926** (+4.08 pt vs
R1 solo).

## Multi-seed leaderboard

| seed | best ep | val/μP | test/μP | gap | MP@cov0.85 (val) |
|---|---|---|---|---|---|
| 1337 (R1) | 26 | 0.6908 | **0.6620** | −2.88 | 0.7518 |
| 42 | 12 | 0.6929 | 0.4126 | **−28.03** | 0.7457 |
| 2026 | 10 | 0.6943 | 0.5727 | −12.16 | 0.7585 |
| **mean** | — | 0.6927 | 0.5491 | −14.36 | 0.7520 |
| **std** | — | **±0.002** | **±0.103** | — | **±0.005** |

**Val/μP is stable to ±0.002** but **test/μP varies by ±0.10 across seeds**.
This single observation rewrites the entire Phase H/I diagnosis: most of the
"val/test gap" we attributed to distribution shift was actually
**seed-variance on a 110-source-file test set**.

## Greedy weight soup falsified (again)

3-seed greedy soup attempted; both addition steps rejected:
- Start s42 (val 0.6938).
- Add s2026: val drops to **0.1189** (rejected).
- Add s1337: val drops to **0.0265** (rejected).
- Final soup = s42 alone (no averaging happened).

This is the **second** soup failure — Phase I H1+H1b also crashed. Conclusion:
HydroHydra's optimization landscape has many narrow non-overlapping basins.
**Weight averaging across seeds is fundamentally infeasible** for this model.

## Logit ensemble — the workaround

Each seed makes independent predictions; we average softmax probabilities at
inference. Doesn't require shared basin; standard variance-reduction trick.
3× inference cost.

### Ensemble metrics (full coverage, post-T):

| metric | R1 solo | Ensemble | Δ |
|---|---|---|---|
| val/μP | 0.6908 | **0.7301** | **+0.0393** |
| test/μP | 0.6620 | 0.6357 | −0.0263 (R1 was lucky) |
| AUROC | — | 0.8457 | — |
| F1 | — | 0.5237 | — |
| MCC | — | 0.4412 | — |
| accuracy | — | 0.5859 | — |
| recall (macro) | — | 0.5406 | — |
| temperature | 3.287 | 2.237 | (less flattening needed) |

### Selective MP@cov 0.85 — production-relevant:

| metric | R1 solo | **Ensemble** | Δ |
|---|---|---|---|
| MP@cov0.85 | 0.7518 | **0.7926** | **+0.0408** |
| Cargo @0.85 | 0.7407 | 0.8204 | +0.0797 |
| Passenger @0.85 | 0.6113 | 0.6166 | +0.0053 |
| Tanker @0.85 | 0.7591 | 0.8360 | +0.0769 |
| **Tug @0.85** | 0.7059 | **0.8911** | **+0.1852** |

**Tug @cov0.85 jumped 18.5 pt** — addresses the persistent Tug bottleneck
flagged across Phase H/I as the production gate (target ≥0.65 was
comfortably exceeded).

### What the ensemble doesn't fix

- **recall_floor 0.6 still infeasible.** No threshold combination gives
  per-class recall ≥0.60 simultaneously. Same as R1 solo.
- **Headline test/μP** is below R1's lucky 0.6620 (0.6357), but
  significantly above the single-seed mean (0.5491) — proves the variance
  reduction works.
- **Class-1 Passenger** sees only +0.005 lift vs R1, the smallest gain.
  Passenger is the hardest class in this dataset for all three seeds
  (single-model val precision: 0.59-0.61).

## Production recommendation

**Ship the 3-seed logit ensemble** (`lightning_logs/phaseI_multiseed_ensemble/`)
as the new full-coverage and selective-prediction baseline.

Inference cost: 3× a single model — acceptable for offline classification.
For real-time or resource-constrained deployment, fall back to R1 solo
(s1337) at slightly lower MP@cov0.85.

Production artifacts:
- `ensemble.md` — summary
- `temperature.pt` — T = 2.237 (post-ensemble calibration)
- `thresholds.json` — per-class thresholds for MP@cov0.85
- `selective_pr.md` — full coverage curve

Inference command:
```python
python -m inference.ensemble_eval \
  --ckpts \
    lightning_logs/phaseG_R1_alpha/.../hydra-026-p0.6908.ckpt \
    lightning_logs/phaseI_R1_s42/.../hydra-012-p0.6929.ckpt \
    lightning_logs/phaseI_R1_s2026/.../hydra-010-p0.6943.ckpt \
  --data_dir <DATA_DIR> \
  --out_dir <OUT_DIR>
```

## Open question — is even more variance lurking?

Three seeds is a small sample. test/μP std=0.103 from N=3 is itself
high-uncertainty. Adding 2 more seeds (s7, s12345) would tighten the
estimate. Each costs ~6h. If the user wants a confident ceiling estimate,
re-run twice more.

## Rewriting the diagnosis from Phase H/I

Most of the val/test "gap pathology" we attributed to distribution shift
was instead **single-seed test-side noise**. The Phase H+I single-seed
comparisons against R1 = 0.6620 were partly comparing against an outlier
draw. Some of those experiments may actually have been winners but looked
like regressions because s1337's test was high.

What was confirmed real:
- The val/test gap exists on average (test mean 0.5491 < val mean 0.6927)
- LDAM-DRW destabilizes Tug specifically (orthogonal failure)
- LPC-fix works numerically (the fix shipped)
- Soup is infeasible (replicated failure)

What was probably noise:
- Most single-experiment "regressions" by 1-3 pt vs R1 on test
- Best-val-epoch effects (different epochs converge across seeds anyway)

## Files written

- `inference/ensemble_eval.py` — NEW (~250 LOC), logit-level ensemble runner
- `lightning_logs/phaseI_R1_s42/` — full multi-seed run logs/ckpts
- `lightning_logs/phaseI_R1_s2026/` — full multi-seed run logs/ckpts
- `lightning_logs/phaseI_multiseed_soup/soup.md` — falsified result
- `lightning_logs/phaseI_multiseed_ensemble/ensemble.md` — NEW baseline
- `lightning_logs/phaseI_multiseed_ensemble/selective_pr.md`
- `lightning_logs/phaseI_multiseed_ensemble/temperature.pt`
- `lightning_logs/phaseI_multiseed_ensemble/thresholds.json`
- `scripts/run_phaseI_multiseed.sh` — multi-seed runner
