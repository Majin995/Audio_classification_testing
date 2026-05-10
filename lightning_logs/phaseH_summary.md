# Phase H — Results Summary

**Started**: 2026-05-05 20:15 · **Sweep complete**: 2026-05-06 ~07:30 · **Followups in progress**: H1b, H7b

Baseline: **phaseG_R1** val/μP=0.6908  test/μP=0.6620  MP@cov0.85=0.7518.

## Headline

**No P0/P1 lever beat the baseline on test.** H1 LDAM-margin is a statistical tie. All other levers regress on test, several catastrophically.

The Phase H sweep tested 6 single-feature levers (3 P0 + 3 P1) plus one stacked combination. The headline finding mirrors Phase G: **the front-end (Gabor + Scattering + SincNet + TDSBE) is doing the heavy lifting; backbone/head/loss perturbations layered on top consistently overfit val and lose test.**

## Sweep table — sorted by test/μP

| Run | val/μP | Δval | test/μP | Δtest | c2 Tanker | c3 Tug | MP@cov0.85 |
|---|---|---|---|---|---|---|---|
| **R1 baseline** | **0.6908** | — | **0.6620** | — | — | — | 0.7518 |
| H1 LDAM-margin (DRW silently fired ep 40) | 0.7035 | +0.0127 | 0.6615 | −0.0005 | 0.7824 | 0.6340 | 0.7300 |
| H3 Global-attn (HELIX-lite) | 0.6882 | −0.0026 | 0.6474 | −0.0146 | 0.7703 | 0.7077 | 0.7437 |
| H2 JTFS-lite (Δ + ΔΔ on Scat1D) | 0.7192 | +0.0284 | 0.6158 | −0.0462 | 0.7876 | **0.7726** | 0.7865 |
| H5 DEMON-MoE (4-expert head) | **0.7279** | **+0.0371** | 0.6128 | −0.0492 | 0.7484 | 0.7391 | **0.7905** |
| H6 Manifold mixup | 0.6884 | −0.0024 | 0.6074 | −0.0546 | 0.7346 | 0.6966 | 0.7508 |
| H7 Stacked P0 (LDAM + JTFS + GA) | 0.7000 | +0.0092 | 0.5869 | −0.0751 | 0.7904 | 0.6579 | 0.7149 |
| H4 Sub-Center ArcFace K=2 | 0.6855 | −0.0053 | 0.4805 | −0.1815 | 0.7919 | 0.6782 | 0.7586 |

## Two operating regimes — different winners

**Full coverage (must classify everything):** R1 baseline (0.6620) and H1 LDAM (0.6615) are tied. All others regress.

**85% gated coverage (allowed to abstain 15%):**
1. **H5 DEMON-MoE: 0.7905** (+3.87 pt vs R1)
2. H2 JTFS-lite: 0.7865 (+3.47 pt)
3. H4 SC-ArcFace: 0.7586 (+0.68 pt)
4. R1 baseline: 0.7518

## Key findings

### Tanker class (the original Phase G motivation)

**Already solved by Phase G R1.** All Phase H runs have c2 Tanker precision in 0.73–0.79 — vs the original Phase G baseline's ~0.50. The Sinc+TDSBE front-end (already in R1) was the lift; Phase H's architecture/loss/head perturbations cannot add more to a class that's already at 0.78.

### val/test gap is the silent killer

The bigger models (H2 JTFS, H5 DEMON-MoE, H7 Stacked) all show val/test gaps of 10–11 pt, vs R1's −2.88 pt. Higher-capacity heads/branches latch onto val-specific patterns that don't transfer.

### LDAM-DRW: margin works, DRW hurts

H1 confirms LDAM's per-class margin is a real lever (+1.27 pt val, flat test). But **DRW with class-balanced weights `[0.79, 0.64, 0.64, 1.93]` actively hurt**: at epoch 40 when DRW fired (silently — log was buffered), Tug precision dropped from 0.6340 to 0.5312, opposite of the intended lift. Class-balanced weighting up-samples Tug too aggressively for this dataset where Tug isn't *under-fit* — it's *intrinsically harder*.

### DEMON-MoE: best post-cal, worst test

H5's gating (4 experts) does deliver more balanced per-class precision (all 0.71–0.75 — flatter distribution than other runs). Post-cal MP@0.85 = 0.7905 is the best of any run. But raw test/μP = 0.6128 because the gate over-commits on val patterns. Production use is gated only.

### Stacking is destructive (H7)

Stacking the 3 P0 levers (LDAM + JTFS-lite + global-attn) produced **worse val AND worse test than any single lever**. Negative compositionality. The plan's "stack all P0 wins" hypothesis is falsified.

### ArcFace family confirmed broken on this dataset

H4 Sub-Center ArcFace dropped test by −18 pt — even worse than vanilla ArcFace's Phase G regression. The angular margin geometry simply does not survive this dataset's val→test domain shift. Do not revisit.

## Bug-fixes shipped during the sweep

1. **DRW print buffered**: `print()` from `on_train_epoch_start` was buffered behind the Lightning progress bar; H1 silently fired DRW at epoch 40 but no log message appeared until process exit. Fixed: `self.print(flush=True)` + idempotent `weight is None` guard.
2. **`cls_num_list` not in checkpoint hparams**: post-cal `load_from_checkpoint` for LDAM crashed with `ValueError: loss='ldam' requires cls_num_list`. Fixed: defaults to uniform list when missing (DRW gated on training mode anyway).
3. **β=0.999 saturated** for the Lexar dataset's class counts (~15k–55k). All weights collapsed to ~1.0, making DRW a no-op. Fixed: default `--ldam_drw_beta 0.99999` + degenerate-weight detection that falls back to inverse-frequency.

## Ship recommendations

**Full-coverage classifier (no abstention)**: keep **Phase G R1** as ship config. H1 LDAM-margin is a tied alternative if you want a slightly higher val (~+1.3 pt) at no test cost; the only practical reason to swap would be downstream calibration.

**Gated classifier (85% coverage)**: ship **H5 DEMON-MoE** post-cal config — `--head_type demon_moe --moe_n_experts 4 --moe_aux_weight 0.05`. Delivers MP@0.85 = 0.7905 (+3.87 pt vs R1). Note: only gain at gated coverage; full-coverage test regresses.

## What's still being tested (followup runs)

- **H1b**: LDAM with `drw_epoch=15` + `patience=25` so DRW fires before early-stop. The original H1 had DRW at epoch 40 and patience 15 — DRW fired but at the tail end of trainable window.
- **H7b**: Stacked P0 with the same realistic DRW timing.

Both running sequentially after the main sweep. Will append numbers when available.

## Phase H deliverables

- 7 trained checkpoints in `lightning_logs/phaseH_*/version_0/checkpoints/`
- Per-run post-cal artifacts: `temperature.pt`, `thresholds.json`, `selective_pr.md`
- This summary, regeneratable via `python scripts/compile_phaseH_results.py`
- 6 new code paths in `models/hydro_hydra.py`, `models/heads.py`, `models/__init__.py`, `training/train_hydra.py` — all back-compatible (Phase G R1 ckpt still loads)
