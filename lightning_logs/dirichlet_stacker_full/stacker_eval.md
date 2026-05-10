# Dirichlet-stacker ensemble report

- classifier: `mlp64_k10`
- data_dir: `data/Split1s_eval`
- ckpts (5):
  - `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
  - `lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt`
  - `lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt`
  - `lightning_logs/phaseI_R1_s7/version_1/checkpoints/hydra-032-p0.7010.ckpt`
  - `lightning_logs/phaseI_R1_s12345/version_0/checkpoints/hydra-031-p0.7042.ckpt`

## Per-model metrics
| model | val/F1 | val/macroP | test/F1 | test/macroP |
|---|---|---|---|---|
| hydra-026-p0.6908.ckpt | 0.7854 | 0.8125 | 0.7550 | 0.7718 |
| hydra-012-p0.6929.ckpt | 0.5587 | 0.7346 | 0.5409 | 0.7115 |
| hydra-010-p0.6943.ckpt | 0.6513 | 0.7448 | 0.6188 | 0.7154 |
| hydra-032-p0.7010.ckpt | 0.7419 | 0.7765 | 0.7214 | 0.7512 |
| hydra-031-p0.7042.ckpt | 0.7950 | 0.8294 | 0.7629 | 0.7900 |

## Stacker test metrics
- F1:      0.9559
- macro_P: 0.9549
- recall:  0.9572
- MCC:     0.9404
- AUROC:   0.9957
- Cargo: P=0.953 R=0.913 F1=0.932
- Passenger: P=0.961 R=0.977 F1=0.969
- Tanker: P=0.924 R=0.947 F1=0.935
- Tug: P=0.981 R=0.993 F1=0.987

## Confusion matrix
(rows = true, cols = predicted)
```
            Cargo  Passenger     Tanker        Tug
 Cargo:       1547         19        107         22
Passenger:          7        969          4         12
Tanker:         60         16       1350          0
   Tug:          9          4          0       1762
```