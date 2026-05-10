# Phase C — 1-ckpt v1 on `precise-013-p0.7444.ckpt`

- model_version = v1
- ensemble size = 1
- best alpha (val) = 0.30  (val/micro_precision=0.6359)
- KNN: k=5, cosine, distance-weighted, fit on first ckpt's train (167872 samples)
  - ckpt: `lightning_logs/grid_precise_verify/verify_B_m0.30_g2.0_s0.05_gw0.0_seed2026/version_0/checkpoints/precise-013-p0.7444.ckpt`  (T=1.6841)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.4334 | 0.5193 | 0.3817 | 0.4334 | 0.2651 | 0.6779 |
| blend (alpha*)     |   0.30 | 0.5457 | 0.5547 | 0.5350 | 0.5457 | 0.3805 | 0.7593 |
| knn-only           |   0.00 | 0.5352 | 0.5366 | 0.5286 | 0.5352 | 0.3572 | 0.7404 |
