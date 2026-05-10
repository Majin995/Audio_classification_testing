# Model Soup — 1/2 ckpts kept

## Components
- `lightning_logs/phaseH_H1b_ldam_drw_early/version_0/checkpoints/hydra-050-p0.7123.ckpt` (individual val/μP=0.7118)

## Rejected
- `lightning_logs/phaseH_H1_ldam_drw/version_0/checkpoints/hydra-032-p0.7035.ckpt` (individual val/μP=0.7047)

## Final metrics
- val/macro_P = **0.7118**
- test/macro_P = **0.6199**
- val/test gap = +0.0919

### Per-class precision
| split | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| val | 0.6827 | 0.6811 | 0.7460 | 0.7372 |
| test | 0.5483 | 0.6734 | 0.5450 | 0.7130 |
