# Dirichlet-stacker ensemble report

- classifier: `mlp64_k10`
- data_dir: `/var/mnt/5A009BF8009BD8F9/Data/Classification_rapid_testing_3s`
- ckpts (5):
  - `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
  - `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt`
  - `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt`
  - `lightning_logs/phaseI_R1_s7/version_1/checkpoints/hydra-032-p0.7010.ckpt`
  - `lightning_logs/phaseI_R1_s12345/version_0/checkpoints/hydra-031-p0.7042.ckpt`

## Per-model metrics
| model | val/F1 | val/macroP | test/F1 | test/macroP |
|---|---|---|---|---|
| hydra-026-p0.6908.ckpt | 0.8720 | 0.8913 | 0.7959 | 0.8512 |
| hydra-012-p0.6929.ckpt | 0.5477 | 0.7545 | 0.5820 | 0.7941 |
| hydra-010-p0.6943.ckpt | 0.7108 | 0.8068 | 0.6287 | 0.7960 |
| hydra-032-p0.7010.ckpt | 0.7915 | 0.8333 | 0.7677 | 0.8566 |
| hydra-031-p0.7042.ckpt | 0.8038 | 0.8426 | 0.8064 | 0.8457 |

## Stacker test metrics
- F1:      0.9294
- macro_P: 0.9334
- recall:  0.9297
- MCC:     0.9075
- AUROC:   0.9934
- Cargo: P=0.882 R=0.938 F1=0.909
- Passenger: P=0.912 R=0.969 F1=0.939
- Tanker: P=1.000 R=0.844 F1=0.915
- Tug: P=0.939 R=0.969 F1=0.954

## Confusion matrix
(rows = true, cols = predicted)
```
            Cargo  Passenger     Tanker        Tug
 Cargo:         30          2          0          0
Passenger:          0         31          0          1
Tanker:          4          0         27          1
   Tug:          0          1          0         31
```