# Multi-seed logit-level ensemble report

## Inputs (5 ckpts)

- `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
- `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt`
- `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt`
- `lightning_logs/phaseI_R1_s7/version_1/checkpoints/hydra-032-p0.7010.ckpt`
- `lightning_logs/phaseI_R1_s12345/version_0/checkpoints/hydra-031-p0.7042.ckpt`

## Per-model + ensemble metrics

| model | val/μP | test/μP |
|---|---|---|
| hydra-026-p0.6908.ckpt | 0.6874 | 0.6613 |
| hydra-012-p0.6929.ckpt | 0.6938 | 0.4133 |
| hydra-010-p0.6943.ckpt | 0.6931 | 0.5772 |
| hydra-032-p0.7010.ckpt | 0.7040 | 0.5821 |
| hydra-031-p0.7042.ckpt | 0.7051 | 0.6415 |
| **ensemble (mean prob)** | **0.7421** | **0.6545** |
| **ensemble + post-T** | (val same) | **0.6545** |

## Ensemble test metrics (post-T)

- macro-precision: 0.6545
- accuracy:        0.6227
- F1:              0.5621
- recall:          0.5750
- MCC:             0.4888
- AUROC:           0.8491
- P[class_0]:      0.6567
- P[class_1]:      0.6335
- P[class_2]:      0.5828
- P[class_3]:      0.7449

## Calibration
- temperature: 2.012
- MP@cov0.85: 0.8034
- recall_floor feasible: False