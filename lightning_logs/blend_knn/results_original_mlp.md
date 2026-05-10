# Phase C — inference modes on `precisev2-035-0.8240.ckpt`

- temperature T = 1.0975
- best alpha (val) = 0.70  (val/micro_precision=0.7400)
- KNN: k=5, cosine, distance-weighted, fit on train (167872 samples)

| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |
|---|---|---|---|---|---|---|---|
| head-only          |   1.00 | 0.5987 | 0.5861 | 0.5891 | 0.5987 | 0.4585 | 0.8112 |
| blend (alpha*)     |   0.70 | 0.6254 | 0.6212 | 0.6236 | 0.6254 | 0.4848 | 0.8145 |
| knn-only           |   0.00 | 0.6294 | 0.6492 | 0.6289 | 0.6294 | 0.4887 | 0.7731 |
