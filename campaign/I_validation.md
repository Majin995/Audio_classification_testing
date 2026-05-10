# I. Validation — full Split1s test (n=7872) ranked

All configurations evaluated on FULL Split1s test set (n=7872) — the gold-standard out-of-sample eval.

| rank | config | calib | T | full F1 | full macroP | full recall | full cov | rapid F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Stacker_Dirichlet(C=0.1) | T+thr | 0.50 | 0.8660 | 0.8660 | 0.8666 | 1.000 | 0.9144 |
| 2 | Stacker_Dirichlet_g2(C=0.1) | T+thr | 0.50 | 0.8341 | 0.8318 | 0.8391 | 1.000 | 0.8893 |
| 3 | Geom2_+thr | T+thr | 2.75 | 0.8098 | 0.8651 | 0.7790 | 0.883 | 0.8525 |
| 4 | Stacker_LR(probs,C=1) | T+thr | 0.55 | 0.8074 | 0.8320 | 0.7985 | 0.947 | 0.8517 |
| 5 | NNLS_weighted_+thr | T+thr | 2.20 | 0.7915 | 0.8419 | 0.7678 | 0.895 | 0.8194 |
| 6 | Geom2(s1337+s12345) | T_only | 2.75 | 0.7840 | 0.8057 | 0.8092 | 1.000 | 0.8400 |
| 7 | single_031 | none | — | 0.7722 | 0.8009 | 0.7977 | 1.000 | 0.8432 |
| 8 | N=5_arith_+thr | T+thr | 1.75 | 0.7702 | 0.8408 | 0.7389 | 0.867 | 0.8219 |
| 9 | single_026 | none | — | 0.7641 | 0.7834 | 0.7896 | 1.000 | 0.7824 |
| 10 | N=5_arith | T_only | 1.75 | 0.7432 | 0.7911 | 0.7679 | 1.000 | 0.7927 |
| 11 | single_032 | none | — | 0.7278 | 0.7583 | 0.7471 | 1.000 | 0.7095 |
| 12 | single_010 | none | — | 0.6280 | 0.7236 | 0.6501 | 1.000 | 0.6368 |
| 13 | single_012 | none | — | 0.5469 | 0.7177 | 0.5643 | 1.000 | 0.5846 |

## Top 3 — detail

### Stacker_Dirichlet(C=0.1)
```
T = 0.500
thresholds = [0.0, 0.0, 0.0, 0.0]
full_test:
  acc=0.8673  f1=0.8660  macroP=0.8660  microP=0.8673  recall=0.8666  mcc=0.8218  cov=1.000
    P0=0.815  P1=0.865  P2=0.875  P3=0.910
    R0=0.790  R1=0.881  R2=0.834  R3=0.962
    F1_0=0.802  F1_1=0.873  F1_2=0.854  F1_3=0.935
```

### Stacker_Dirichlet_g2(C=0.1)
```
T = 0.500
thresholds = [0.0, 0.0, 0.0, 0.0]
full_test:
  acc=0.8336  f1=0.8341  macroP=0.8318  microP=0.8336  recall=0.8391  mcc=0.7777  cov=1.000
    P0=0.792  P1=0.835  P2=0.837  P3=0.863
    R0=0.688  R1=0.897  R2=0.864  R3=0.907
    F1_0=0.737  F1_1=0.865  F1_2=0.851  F1_3=0.884
```

### Geom2_+thr
```
T = 2.750
thresholds = [0.0, 0.869, 0.566, 0.0]
full_test:
  acc=0.8632  f1=0.8098  macroP=0.8651  microP=0.8632  recall=0.7790  mcc=0.8221  cov=0.883
    P0=0.821  P1=0.820  P2=0.824  P3=0.996
    R0=0.564  R1=0.910  R2=0.912  R3=0.730
    F1_0=0.669  F1_1=0.862  F1_2=0.866  F1_3=0.842
```

## Δ vs N=5 arith baseline (full F1, macroP)

| config | Δ F1 | Δ macroP | Δ recall |
|---|---|---|---|
| Stacker_Dirichlet(C=0.1) | +0.1228 | +0.0749 | +0.0987 |
| Stacker_Dirichlet_g2(C=0.1) | +0.0909 | +0.0407 | +0.0712 |
| Geom2_+thr | +0.0666 | +0.0739 | +0.0111 |
| Stacker_LR(probs,C=1) | +0.0642 | +0.0409 | +0.0306 |
| NNLS_weighted_+thr | +0.0484 | +0.0507 | -0.0001 |
| Geom2(s1337+s12345) | +0.0408 | +0.0146 | +0.0413 |
| single_031 | +0.0290 | +0.0098 | +0.0298 |
| N=5_arith_+thr | +0.0270 | +0.0497 | -0.0290 |
| single_026 | +0.0209 | -0.0077 | +0.0217 |
| single_032 | -0.0154 | -0.0328 | -0.0208 |
| single_010 | -0.1152 | -0.0675 | -0.1178 |
| single_012 | -0.1963 | -0.0734 | -0.2036 |