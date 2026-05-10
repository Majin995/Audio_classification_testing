# Phase I — α.1 Model Soup findings (negative result)

## What was attempted

Greedy state-dict soup over the strongest MLP+Gambler-head Phase G/H ckpts:
1. R1 baseline (`phaseG_R1_alpha/.../hydra-026-p0.6908.ckpt`) — incompatible
   architecture (legacy `nn.Sequential` head with `head.0`/`head.3` keys).
2. H1 LDAM (`phaseH_H1_ldam_drw/.../hydra-032-p0.7035.ckpt`) — `build_head`
   MLPHead path with `head.net.0`/`head.net.3` keys.
3. H1b LDAM-drw15 (`phaseH_H1b_ldam_drw_early/.../hydra-050-p0.7123.ckpt`) —
   same head path as H1.

Compatible pair = {H1, H1b}.

## Result

| Soup composition | val/μP | test/μP | val/test gap |
|---|---|---|---|
| H1 alone | 0.7047 | (n/a) | — |
| **H1b alone** | **0.7118** | **0.6199** | **+0.0919** |
| H1 ⊕ H1b (greedy) | 0.4661 | — | — |

Greedy soup REJECTED adding H1 to the H1b base because the average crashed val/μP from
0.7118 → 0.4661. Final soup contained only H1b — no actual averaging happened.

## Interpretation

H1 and H1b were trained with the same architecture and same loss *family*
(LDAM-margin), but **different DRW timing** (epoch 40 vs 15) and **different
patience** (15 vs 25). The averaging-collapse means they ended in **different
basins of weight space**. The Wortsman 2022 "model soups" assumption (multiple
fine-tunings of the same init drift in a shared basin) does not hold across
LDAM-DRW hyperparameter variations.

## Implication for Phase I

α.1 model soup is **not viable** for HydroHydra without first running multi-seed
replicates of the *same* training config. The Phase G `hydro_hydra_s{42,1337,2026}`
runs are pre-Phase-G (older Split1s data, old architecture) and not soup-worthy.

If Phase I's α.2 SWA + α.3 drop_path do not close the gap, the next attempt
should be:
1. Run R1 ship config at three new seeds (~18h compute) — multi-seed soup of
   identical-config runs.
2. Or invoke Model Stock (ECCV 2024), which uses just 2 fine-tunings from the
   same init.

R1 itself is not eligible for soup with H1/H1b because its head architecture
differs. To soup with R1, retrain R1 with `gambler_weight=0.0` (drops the
abstention logit, switches to `build_head` path). That's a 6h re-train.

## Files

- `lightning_logs/phaseI_soup_h1_h1b/soup.md` — full soup output.
- `lightning_logs/phaseI_soup_h1_h1b/soup.ckpt` — H1b's weights (since soup
  reduced to H1b alone).
