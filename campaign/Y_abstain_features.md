# Y. Abstain-logit feature inclusion

Hydra outputs 5 logits (4 class + 1 abstain). Compare:
- 4-D: re-normalized 4-class softmax (abstain dropped, then re-softmax)
- 5-D: full 5-class softmax (abstain kept as feature)

| classifier | 4-D full F1 | 4-D full macroP | 5-D full F1 | 5-D full macroP | Δ F1 |
|---|---|---|---|---|---|
| LR(C=0.1) | 0.8657 | 0.8658 | 0.8683 | 0.8686 | +0.0026 |
| LR(C=0.3) | 0.8632 | 0.8638 | 0.8674 | 0.8684 | +0.0042 |
| LR(C=1.0) | 0.8630 | 0.8640 | 0.8685 | 0.8700 | +0.0054 |
| MLP(64) K=10 | 0.8816 | 0.8800 | 0.8721 | 0.8716 | -0.0096 |

## Best 5-D classifier: MLP(64) K=10 → F1 = 0.8721, macroP = 0.8716

```
full_test (5-D):
  acc=0.8727  f1=0.8721  macroP=0.8716  microP=0.8727  recall=0.8729  mcc=0.8291
    P0=0.821  P1=0.871  P2=0.877  P3=0.917
    R0=0.796  R1=0.890  R2=0.854  R3=0.952
    F1_0=0.808  F1_1=0.881  F1_2=0.865  F1_3=0.934
```

**Average Δ F1 across classifiers: +0.0007**
Abstain logit is roughly neutral — no clear signal in either direction.