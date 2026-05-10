# R. Stacker (MLP(64) K=10) + per-class thresholds

Sweeping per-class minimum-confidence thresholds on top of stacker softmax.

## Baseline (no thresholds)
```
rapid_test:
  acc=0.9141  f1=0.9135  macroP=0.9157  microP=0.9141  recall=0.9141  mcc=0.8864
    P0=0.875  P1=0.931  P2=0.968  P3=0.889
    R0=0.875  R1=0.844  R2=0.938  R3=1.000
    F1_0=0.875  F1_1=0.885  F1_2=0.952  F1_3=0.941
full_test:
  acc=0.8812  f1=0.8816  macroP=0.8800  microP=0.8812  recall=0.8844  mcc=0.8410
    P0=0.847  P1=0.890  P2=0.864  P3=0.919
    R0=0.777  R1=0.930  R2=0.875  R3=0.955
    F1_0=0.811  F1_1=0.910  F1_2=0.870  F1_3=0.937
```

## Threshold sweep (fit on val, eval on full)

| p_floor | thresholds | val F1 | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP | full_test cov |
|---|---|---|---|---|---|---|---|
| 0.00 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.50 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.60 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.70 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.75 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.80 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.85 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.90 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |
| 0.95 | [0.0, 0.0, 0.0, 0.0] | 1.0000 | 0.9135 | 0.9157 | 0.8816 | 0.8800 | 1.000 |

## Result
No threshold config beat baseline (F1=0.8816). Stacker alone is the F1 maximum.

Thresholding on stacker outputs is **F1-neutral or harmful** — the stacker's softmax
is already well-calibrated, and any abstention loses recall faster than it gains precision.