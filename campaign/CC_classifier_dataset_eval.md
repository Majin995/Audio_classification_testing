# CC. Full eval on Classifier_Dataset

- data_dir: `/var/mnt/5A009BF8009BD8F9/Data/Classifier_Dataset`
- val n: 50752, test n: 14208
- 5 ckpts (HydroHydra Phase-I seeds)

## Per-model raw metrics

| model | val/F1 | val/macroP | test/F1 | test/macroP |
|---|---|---|---|---|
| hydra-026-p0.6908 | 0.6407 | 0.6874 | 0.6073 | 0.6613 |
| hydra-012-p0.6929 | 0.5555 | 0.6938 | 0.4192 | 0.4133 |
| hydra-010-p0.6943 | 0.6140 | 0.6931 | 0.5091 | 0.5772 |
| hydra-032-p0.7010 | 0.6535 | 0.7040 | 0.5322 | 0.5821 |
| hydra-031-p0.7042 | 0.6561 | 0.7051 | 0.5683 | 0.6415 |

## Headline comparison

| stacker | fit on | test F1 | test macroP | test recall | test MCC | test AUROC |
|---|---|---|---|---|---|---|
| (none — N=5 arith) | — | 0.5621 | 0.6545 | 0.5750 | 0.4888 | 0.8506 |
| MLP(64) K=10 PRE-FIT (Split1s_eval) | (loaded) | 0.5992 | 0.6269 | 0.5926 | 0.4720 | 0.8276 |
| LR(C=0.1) Dirichlet REFIT | Classifier_Dataset/val | 0.5929 | 0.6082 | 0.5989 | 0.5046 | 0.8562 |
| **MLP(64) K=10 REFIT** | Classifier_Dataset/val | **0.5623** | **0.5897** | **0.5681** | **0.4821** | **0.8418** |

## Per-class detail — refit MLP(64) K=10 (production-grade)
```
test:
  acc=0.6248  f1=0.5623  macroP=0.5897  microP=0.6248  recall=0.5681  mcc=0.4821
    P0=0.648  P1=0.699  P2=0.579  P3=0.432
    R0=0.498  R1=0.739  R2=0.818  R3=0.217
    F1_0=0.564  F1_1=0.719  F1_2=0.678  F1_3=0.289
```

## Per-class detail — pre-fit (cross-dataset apply)
```
test:
  acc=0.6191  f1=0.5992  macroP=0.6269  microP=0.6191  recall=0.5926  mcc=0.4720
    P0=0.568  P1=0.829  P2=0.576  P3=0.534
    R0=0.565  R1=0.588  R2=0.785  R3=0.432
    F1_0=0.567  F1_1=0.688  F1_2=0.664  F1_3=0.478
```

## Lift summary
- pre-fit (apply only) Δ F1 = +0.0371
- LR refit Δ F1 = +0.0308
- **MLP refit Δ F1 = +0.0001**