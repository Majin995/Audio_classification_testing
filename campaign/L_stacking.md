# L. Stacking — learned probs combiner

Five models × 4-class probs = 20 features per sample.
Logistic regression fit on rapid_1s val (n=64), evaluated on test.
Compare: (a) raw probs features, (b) log-probs features (≈ Dirichlet calib).

| feat | C (L2 inv) | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |
|---|---|---|---|---|---|
| probs | 0.01 | 0.8429 | 0.8918 | 0.7757 | 0.8034 |
| probs | 0.1 | 0.8504 | 0.8952 | 0.7849 | 0.8061 |
| probs | 1.0 | 0.8658 | 0.8862 | 0.8032 | 0.8113 |
| probs | 10.0 | 0.8437 | 0.8494 | 0.7966 | 0.7965 |
| probs | 100.0 | 0.8445 | 0.8515 | 0.7872 | 0.7861 |
| logprobs | 0.01 | 0.9055 | 0.9072 | 0.8572 | 0.8560 |
| logprobs | 0.1 | 0.9144 | 0.9190 | 0.8660 | 0.8660 |
| logprobs | 1.0 | 0.9222 | 0.9270 | 0.8596 | 0.8602 |
| logprobs | 10.0 | 0.9138 | 0.9178 | 0.8623 | 0.8629 |
| logprobs | 100.0 | 0.9061 | 0.9094 | 0.8644 | 0.8641 |

## Stacking + per-class thresholds (best base + threshold search on stacker val probs)

| feat | C | T | thresholds | full_test F1 | full_test macroP | full_test cov |
|---|---|---|---|---|---|---|
| probs | 0.1 | 0.50 | [0.0, 0.81, 0.61, 0.0] | 0.7927 | 0.8533 | 0.870 |
| probs | 1.0 | 0.55 | [0.0, 0.87, 0.0, 0.0] | 0.8074 | 0.8320 | 0.947 |
| probs | 10.0 | 0.60 | [0.0, 0.0, 0.0, 0.0] | 0.7966 | 0.7965 | 1.000 |
| logprobs | 0.1 | 0.50 | [0.0, 0.0, 0.0, 0.0] | 0.8660 | 0.8660 | 1.000 |
| logprobs | 1.0 | 0.50 | [0.0, 0.0, 0.0, 0.0] | 0.8596 | 0.8602 | 1.000 |
| logprobs | 10.0 | 0.50 | [0.0, 0.0, 0.0, 0.0] | 0.8623 | 0.8629 | 1.000 |

## Best stacker (no thresholds)
```
mode=logprobs  C=0.1
rapid_test:
  acc=0.9141  f1=0.9144  macroP=0.9190  microP=0.9141  recall=0.9141  mcc=0.8868
    P0=0.829  P1=0.933  P2=1.000  P3=0.914
    R0=0.906  R1=0.875  R2=0.875  R3=1.000
    F1_0=0.866  F1_1=0.903  F1_2=0.933  F1_3=0.955
full_test:
  acc=0.8673  f1=0.8660  macroP=0.8660  microP=0.8673  recall=0.8666  mcc=0.8218
    P0=0.815  P1=0.865  P2=0.875  P3=0.910
    R0=0.790  R1=0.881  R2=0.834  R3=0.962
    F1_0=0.802  F1_1=0.873  F1_2=0.854  F1_3=0.935
```

## Best stacker + post-cal
```
mode=logprobs  C=0.1  T=0.500
thresholds=[0.0, 0.0, 0.0, 0.0]
rapid_test:
  acc=0.9141  f1=0.9144  macroP=0.9190  microP=0.9141  recall=0.9141  mcc=0.8868  cov=1.000
    P0=0.829  P1=0.933  P2=1.000  P3=0.914
    R0=0.906  R1=0.875  R2=0.875  R3=1.000
    F1_0=0.866  F1_1=0.903  F1_2=0.933  F1_3=0.955
full_test:
  acc=0.8673  f1=0.8660  macroP=0.8660  microP=0.8673  recall=0.8666  mcc=0.8218  cov=1.000
    P0=0.815  P1=0.865  P2=0.875  P3=0.910
    R0=0.790  R1=0.881  R2=0.834  R3=0.962
    F1_0=0.802  F1_1=0.873  F1_2=0.854  F1_3=0.935
```