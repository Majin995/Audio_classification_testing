# Phase C — 1-ckpt v2 on `precisev2-008-0.6993.ckpt`

- model_version = v2
- ensemble size = 1
- best alpha (val) = 0.50  (val/micro_precision=0.7409)
- KNN: k=5, cosine, distance-weighted, fit on first ckpt's train (167872 samples)
  - ckpt: `lightning_logs/phaseD_canonical_demon/version_0/checkpoints/precisev2-008-0.6993.ckpt`  (T=0.9556)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.5904 | 0.5832 | 0.5706 | 0.5904 | 0.4514 | 0.7880 |
| blend (alpha*)     |   0.50 | 0.6186 | 0.6069 | 0.6062 | 0.6186 | 0.4762 | 0.8044 |
| knn-only           |   0.00 | 0.6243 | 0.6303 | 0.6161 | 0.6243 | 0.4815 | 0.7894 |
