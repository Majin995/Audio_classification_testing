# Phase C — 1-ckpt v2 on `precisev2-008-0.7410.ckpt`

- model_version = v2
- ensemble size = 1
- best alpha (val) = 0.70  (val/micro_precision=0.7520)
- KNN: k=5, cosine, distance-weighted, fit on first ckpt's train (167872 samples)
  - ckpt: `lightning_logs/phaseE_lofar/version_0/checkpoints/precisev2-008-0.7410.ckpt`  (T=0.8544)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.6158 | 0.6080 | 0.6033 | 0.6158 | 0.4744 | 0.7921 |
| blend (alpha*)     |   0.70 | 0.6211 | 0.6365 | 0.6154 | 0.6211 | 0.4827 | 0.7939 |
| knn-only           |   0.00 | 0.6196 | 0.6702 | 0.6204 | 0.6196 | 0.4827 | 0.7679 |
