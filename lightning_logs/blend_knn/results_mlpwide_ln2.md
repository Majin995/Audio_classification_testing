# Phase C — inference modes on `precisev2-023-0.7267.ckpt`

- temperature T = 0.9493
- best alpha (val) = 0.50  (val/micro_precision=0.7417)
- KNN: k=5, cosine, distance-weighted, fit on train (167872 samples)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.6287 | 0.6092 | 0.6184 | 0.6287 | 0.4924 | 0.8062 |
| blend (alpha*)     |   0.50 | 0.6412 | 0.6360 | 0.6327 | 0.6412 | 0.5064 | 0.8133 |
| knn-only           |   0.00 | 0.6411 | 0.6437 | 0.6309 | 0.6411 | 0.5065 | 0.7904 |
