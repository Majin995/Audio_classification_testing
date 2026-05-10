# Model Soup — 1/3 ckpts kept

## Components
- `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt` (individual val/μP=0.6938)

## Rejected
- `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt` (individual val/μP=0.6931)
- `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt` (individual val/μP=0.6874)

## Final metrics
- val/macro_P = **0.6938**
- test/macro_P = **0.4133**
- val/test gap = +0.2805

### Per-class precision
| split | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| val | 0.5696 | 0.5250 | 0.8408 | 0.8398 |
| test | 0.4766 | 0.4928 | 0.6838 | 0.0000 |
