# Dirichlet-stacker ensemble report

- classifier: `mlp64_k10`
- data_dir: `/var/mnt/5A009BF8009BD8F9/Data/Classification_rapid_testing_1s`
- ckpts (5):
  - `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
  - `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt`
  - `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt`
  - `lightning_logs/phaseI_R1_s7/version_1/checkpoints/hydra-032-p0.7010.ckpt`
  - `lightning_logs/phaseI_R1_s12345/version_0/checkpoints/hydra-031-p0.7042.ckpt`

## Per-model metrics
| model | val/F1 | val/macroP | test/F1 | test/macroP |
|---|---|---|---|---|
| hydra-026-p0.6908.ckpt | 0.8878 | 0.8994 | 0.7824 | 0.8259 |
| hydra-012-p0.6929.ckpt | 0.6287 | 0.7999 | 0.5846 | 0.8125 |
| hydra-010-p0.6943.ckpt | 0.7596 | 0.8449 | 0.6368 | 0.7896 |
| hydra-032-p0.7010.ckpt | 0.8294 | 0.8662 | 0.7095 | 0.7719 |
| hydra-031-p0.7042.ckpt | 0.8564 | 0.8905 | 0.8432 | 0.8728 |

## Stacker test metrics
- F1:      0.9135
- macro_P: 0.9157
- recall:  0.9141
- MCC:     0.8864
- AUROC:   0.9927
- Cargo: P=0.875 R=0.875 F1=0.875
- Passenger: P=0.931 R=0.844 F1=0.885
- Tanker: P=0.968 R=0.938 F1=0.952
- Tug: P=0.889 R=1.000 F1=0.941

## Confusion matrix
(rows = true, cols = predicted)
```
            Cargo  Passenger     Tanker        Tug
 Cargo:         28          2          1          1
Passenger:          2         27          0          3
Tanker:          2          0         30          0
   Tug:          0          0          0         32
```