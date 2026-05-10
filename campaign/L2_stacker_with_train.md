# L2. Stacker with train+val data

- train+val n = 320 (vs val-only n = 64)

| C | fit_data | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |
|---|---|---|---|---|---|
| 0.01 | val_only | 0.9055 | 0.9072 | 0.8572 | 0.8560 |
| 0.05 | val_only | 0.9139 | 0.9157 | 0.8641 | 0.8636 |
| 0.1 | val_only | 0.9144 | 0.9190 | 0.8660 | 0.8660 |
| 0.3 | val_only | 0.9144 | 0.9190 | 0.8626 | 0.8633 |
| 1.0 | val_only | 0.9222 | 0.9270 | 0.8596 | 0.8602 |
| 3.0 | val_only | 0.9138 | 0.9178 | 0.8645 | 0.8649 |
| 10.0 | val_only | 0.9138 | 0.9178 | 0.8623 | 0.8629 |
| 0.01 | train+val | 0.8742 | 0.8788 | 0.8317 | 0.8312 |
| 0.05 | train+val | 0.8667 | 0.8704 | 0.8260 | 0.8275 |
| 0.1 | train+val | 0.8586 | 0.8637 | 0.8250 | 0.8272 |
| 0.3 | train+val | 0.8586 | 0.8637 | 0.8229 | 0.8267 |
| 1.0 | train+val | 0.8592 | 0.8651 | 0.8224 | 0.8274 |
| 3.0 | train+val | 0.8592 | 0.8651 | 0.8210 | 0.8262 |
| 10.0 | train+val | 0.8592 | 0.8651 | 0.8201 | 0.8254 |

## Top 5 by full_test F1
- C=0.1, fit_data=val_only → full F1=0.8660, macroP=0.8660, recall=0.8666
- C=3.0, fit_data=val_only → full F1=0.8645, macroP=0.8649, recall=0.8651
- C=0.05, fit_data=val_only → full F1=0.8641, macroP=0.8636, recall=0.8655
- C=0.3, fit_data=val_only → full F1=0.8626, macroP=0.8633, recall=0.8631
- C=10.0, fit_data=val_only → full F1=0.8623, macroP=0.8629, recall=0.8627

## Detail — best
```
C=0.1  fit_data=val_only
full_test:
  acc=0.8673  f1=0.8660  macroP=0.8660  microP=0.8673  recall=0.8666  mcc=0.8218
    P0=0.815  P1=0.865  P2=0.875  P3=0.910
    R0=0.790  R1=0.881  R2=0.834  R3=0.962
    F1_0=0.802  F1_1=0.873  F1_2=0.854  F1_3=0.935
```