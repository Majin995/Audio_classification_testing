# E. TTA sweep on geom-2 ensemble (s1337 + s12345)

Augmentation source: HydroHydra._WaveformAugV2 with force_train=True.
Probabilities averaged across K stochastic forwards per sample.

| K | T | rapid_test F1 (raw) | rapid_test F1 (cal) | full_test F1 (raw) | full_test F1 (cal) | full_test macroP (cal) |
|---|---|---|---|---|---|---|
| 1 | 2.75 | 0.8400 | 0.8525 | 0.7840 | 0.8098 | 0.8651 |
| 4 | 2.15 | 0.8005 | 0.8125 | 0.7437 | 0.7557 | 0.8377 |
| 8 | 1.40 | 0.7903 | 0.8108 | 0.7448 | 0.7783 | 0.8445 |
| 16 | 1.50 | 0.7737 | 0.7761 | — | — | — |

## Best by full_test_cal F1 (or rapid if no full)
```
K=1  T=2.750  thresholds=[0.0, 0.869, 0.566, 0.0]
full_test (cal):
  acc=0.8632  f1=0.8098  macroP=0.8651  microP=0.8632  recall=0.7790  mcc=0.8221  cov=0.883
    P0=0.821  P1=0.820  P2=0.824  P3=0.996
    R0=0.564  R1=0.910  R2=0.912  R3=0.730
    F1_0=0.669  F1_1=0.862  F1_2=0.866  F1_3=0.842
rapid_test (cal):
  acc=0.9273  f1=0.8525  macroP=0.9345  microP=0.9273  recall=0.7969  mcc=0.9058  cov=0.859
    P0=0.952  P1=0.903  P2=0.882  P3=1.000
    R0=0.625  R1=0.875  R2=0.938  R3=0.750
    F1_0=0.755  F1_1=0.889  F1_2=0.909  F1_3=0.857
```