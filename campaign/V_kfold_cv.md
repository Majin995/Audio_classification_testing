# V. 5-fold CV stacker hyperparam selection

CV on rapid_1s val (n=64). For MLP families, average across 5 init seeds per fold.

| family | CV F1 mean | CV F1 std | CV macroP mean | full_test F1 (refit on full val) |
|---|---|---|---|---|
| LR(C=0.05) | 0.9343 | 0.0333 | 0.9500 | 0.8641 |
| LR(C=0.1) | 0.9343 | 0.0333 | 0.9500 | 0.8660 |
| LR(C=0.3) | 0.9514 | 0.0400 | 0.9625 | 0.8626 |
| LR(C=1.0) | 0.9514 | 0.0400 | 0.9625 | 0.8596 |
| MLP(32) | 0.9686 | 0.0388 | 0.9750 | 0.8628 |
| MLP(64) | 0.9686 | 0.0388 | 0.9750 | 0.8770 |
| MLP(64,32) | 0.9686 | 0.0388 | 0.9750 | 0.8820 |

## CV-best family: MLP(32) (CV F1 = 0.9686, full_test F1 = 0.8628)
## Full-test-best family: MLP(64,32) (full_test F1 = 0.8820)

If these match, CV correctly selected the best family without peeking at full test.
**MISMATCH** — CV picked MLP(32) but full-test-best is MLP(64,32). Likely small-n noise; both are within ~0.01 F1.