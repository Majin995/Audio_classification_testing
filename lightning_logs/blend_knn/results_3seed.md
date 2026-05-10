# Phase C — 3-ckpt v2 on `precisev2-030-0.7305.ckpt` + 2 more

- model_version = v2
- ensemble size = 3
- best alpha (val) = 0.80  (val/micro_precision=0.7495)
- KNN: k=5, cosine, distance-weighted, fit on first ckpt's train (167872 samples)
  - ckpt: `lightning_logs/phaseB_cosine_ln2/version_0/checkpoints/precisev2-030-0.7305.ckpt`  (T=1.0115)
  - ckpt: `lightning_logs/phaseB_cosine_ln2_s1337/version_0/checkpoints/precisev2-019-0.7270.ckpt`  (T=0.9389)
  - ckpt: `lightning_logs/phaseB_cosine_ln2_s2026/version_0/checkpoints/precisev2-053-0.7311.ckpt`  (T=1.1424)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.6214 | 0.6065 | 0.6088 | 0.6214 | 0.4897 | 0.8190 |
| blend (alpha*)     |   0.80 | 0.6372 | 0.6205 | 0.6268 | 0.6372 | 0.5051 | 0.8216 |
| knn-only           |   0.00 | 0.6391 | 0.6451 | 0.6278 | 0.6391 | 0.5022 | 0.7821 |
