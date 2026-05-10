# B. Baseline on rapid_1s — 5-seed ensemble

- probs_dir: `campaign/probs_rapid_1s`
- num_classes: 4
- classes: ['Cargo', 'Passenger', 'Tanker', 'Tug']
- val n: 64, test n: 128

## Per-model (raw, no temperature)

| stem | val/F1 | val/macroP | val/MCC | test/F1 | test/macroP | test/MCC |
|---|---|---|---|---|---|---|
| hydra-026-p0.6908 | 0.8878 | 0.8994 | 0.8584 | 0.7824 | 0.8259 | 0.7340 |
| hydra-012-p0.6929 | 0.6287 | 0.7999 | 0.5780 | 0.5846 | 0.8125 | 0.5214 |
| hydra-010-p0.6943 | 0.7596 | 0.8449 | 0.6976 | 0.6368 | 0.7896 | 0.5660 |
| hydra-032-p0.7010 | 0.8294 | 0.8662 | 0.7834 | 0.7095 | 0.7719 | 0.6348 |
| hydra-031-p0.7042 | 0.8564 | 0.8905 | 0.8236 | 0.8432 | 0.8728 | 0.8018 |

## Ensemble (mean prob, no temperature)

```
VAL:    acc=0.8438  f1=0.8475  macroP=0.8944  microP=0.8438  recall=0.8438  mcc=0.8079
    P0=1.000  P1=0.640  P2=0.938  P3=1.000
    R0=0.750  R1=1.000  R2=0.938  R3=0.688
    F1_0=0.857  F1_1=0.780  F1_2=0.938  F1_3=0.815
TEST:   acc=0.7891  f1=0.7927  macroP=0.8747  microP=0.7891  recall=0.7891  mcc=0.7479
    P0=1.000  P1=0.561  P2=0.938  P3=1.000
    R0=0.625  R1=1.000  R2=0.938  R3=0.594
    F1_0=0.769  F1_1=0.719  F1_2=0.938  F1_3=0.745
```

## Ensemble + temperature (T = 1.750, fitted on val)

```
VAL:    acc=0.8438  f1=0.8475  macroP=0.8944  microP=0.8438  recall=0.8438  mcc=0.8079
    P0=1.000  P1=0.640  P2=0.938  P3=1.000
    R0=0.750  R1=1.000  R2=0.938  R3=0.688
    F1_0=0.857  F1_1=0.780  F1_2=0.938  F1_3=0.815
TEST:   acc=0.7891  f1=0.7927  macroP=0.8747  microP=0.7891  recall=0.7891  mcc=0.7479
    P0=1.000  P1=0.561  P2=0.938  P3=1.000
    R0=0.625  R1=1.000  R2=0.938  R3=0.594
    F1_0=0.769  F1_1=0.719  F1_2=0.938  F1_3=0.745
```
