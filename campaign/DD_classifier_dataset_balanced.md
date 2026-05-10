# DD. Class-balanced + threshold-calibrated stacker on Classifier_Dataset

- val n=50752, test n=14208
- val class counts: Cargo=13236, Passenger=15045, Tanker=17087, Tug=5384
- Tug imbalance: 10.6% of val

| stacker | balanced? | + thresholds | test F1 | test macroP | test recall | per-class F1 |
|---|---|---|---|---|---|---|
| LR(C=0.1) | no | no | 0.5929 | 0.6082 | 0.5989 | [0.548, 0.759, 0.689, 0.376] |
| LR(C=0.1) | no | yes | 0.5950 | 0.6275 | 0.5911 | [0.545, 0.76, 0.688, 0.387] |
| LR(C=0.1) | yes | no | 0.5997 | 0.6027 | 0.6160 | [0.556, 0.746, 0.679, 0.418] |
| LR(C=0.1) | yes | yes | 0.6042 | 0.6326 | 0.5890 | [0.551, 0.746, 0.679, 0.441] |
| MLP(64) K=10 | no | no | 0.5623 | 0.5897 | 0.5681 | [0.564, 0.719, 0.678, 0.289] |
| MLP(64) K=10 | no | yes | 0.5610 | 0.5952 | 0.5599 | [0.561, 0.712, 0.682, 0.289] |
| MLP(64) K=10 | yes | no | 0.5676 | 0.5769 | 0.5726 | [0.57, 0.711, 0.686, 0.303] |
| MLP(64) K=10 | yes | yes | 0.5648 | 0.5967 | 0.5606 | [0.569, 0.712, 0.686, 0.292] |

## Best by test F1
```
LR, balanced=True, thresholds=True
test:
  acc=0.6545  f1=0.6042  macroP=0.6326  microP=0.6545  recall=0.5890  mcc=0.5244  cov=0.935
    P0=0.637  P1=0.816  P2=0.609  P3=0.468
    R0=0.485  R1=0.687  R2=0.767  R3=0.417
    F1_0=0.551  F1_1=0.746  F1_2=0.679  F1_3=0.441
```