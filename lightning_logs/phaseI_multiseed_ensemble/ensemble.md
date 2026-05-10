# Multi-seed logit-level ensemble report

## Inputs (3 ckpts)

- `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
- `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt`
- `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt`

## Per-model + ensemble metrics

| model | val/μP | test/μP |
|---|---|---|
| hydra-026-p0.6908.ckpt | 0.6874 | 0.6613 |
| hydra-012-p0.6929.ckpt | 0.6938 | 0.4133 |
| hydra-010-p0.6943.ckpt | 0.6931 | 0.5772 |
| **ensemble (mean prob)** | **0.7301** | **0.6357** |
| **ensemble + post-T** | (val same) | **0.6357** |

## Ensemble test metrics (post-T)

- macro-precision: 0.6357
- accuracy:        0.5859
- F1:              0.5237
- recall:          0.5406
- MCC:             0.4412
- AUROC:           0.8457
- P[class_0]:      0.6073
- P[class_1]:      0.5534
- P[class_2]:      0.6039
- P[class_3]:      0.7784

## Calibration
- temperature: 2.237
- MP@cov0.85: 0.7926
- recall_floor feasible: False