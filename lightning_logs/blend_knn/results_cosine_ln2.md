# Phase C — inference modes on `precisev2-030-0.7305.ckpt`

- temperature T = 1.0115
- best alpha (val) = 0.60  (val/micro_precision=0.7419)
- KNN: k=5, cosine, distance-weighted, fit on train (167872 samples)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.6253 | 0.6042 | 0.6118 | 0.6253 | 0.4908 | 0.8108 |
| blend (alpha*)     |   0.60 | 0.6441 | 0.6341 | 0.6363 | 0.6441 | 0.5112 | 0.8141 |
| knn-only           |   0.00 | 0.6392 | 0.6452 | 0.6278 | 0.6392 | 0.5023 | 0.7821 |
