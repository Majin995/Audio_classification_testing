# N2. Bootstrap robustness: LR vs MLP(64)

Two sources of variance: val resampling (rows 1-2) and MLP init seed (row 3).

| classifier | n | mean F1 | std | p05 | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|---|
| LR(C=0.1) | 100 | 0.8451 | 0.0175 | 0.8174 | 0.8480 | 0.8660 | 0.7790 | 0.8729 |
| MLP(64) | 100 | 0.8504 | 0.0154 | 0.8186 | 0.8511 | 0.8712 | 0.7962 | 0.8767 |
| MLP(64,seed=...) | 50 | 0.8632 | 0.0099 | 0.8488 | 0.8640 | 0.8782 | 0.8267 | 0.8808 |

## Paired comparison (same val resamples)
- MLP(64) − LR(C=0.1) F1: mean=+0.0053, std=0.0154
- P(MLP > LR) = 59.0%